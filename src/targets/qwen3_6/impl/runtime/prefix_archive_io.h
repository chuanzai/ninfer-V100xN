#pragma once

// Durable image of the host-resident prefix archive.
//
// What is written is exactly what archive_prefix() captured: the immutable KV chunk images
// (PrefixKVChunk::text / ::backend, one vector per execution rank) plus the per-frontier
// recurrent-state images (PrefixCheckpoint::image) and the token identity that authorises reuse.
// Nothing here touches CUDA -- the chunks are already host copies of device pages -- so a save is
// a memory-to-disk dump and a load is disk-to-memory, both executable while the engine is idle.
//
// A file is only ever restored into the engine that produced it: build, artifact identity, KV
// layout and rope regime are stamped, and any mismatch drops the whole file instead of warming a
// prefix whose KV no longer means what the current weights/kernels would compute. That invariant
// is stricter than "same model": a rebuilt binary can move numerics, and a restored prefix would
// then be silently stale rather than wrong-looking.
//
// Layout: a fixed 512-byte header followed by
//   chunk table : the deduplicated union of every selected branch's chunk list
//   branches    : ledger + identity + state images + chunk ids, oldest first
// All multi-byte fields are native little-endian; a big-endian host is rejected at compile time.

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <limits>
#include <span>
#include <string>
#include <vector>

#include <sys/stat.h>
#include <unistd.h>

#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ != __ORDER_LITTLE_ENDIAN__
#    error "prefix archive images are little-endian only"
#endif

namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS {
namespace {

constexpr std::array<unsigned char, 8> kPrefixArchiveMagic{{'N', 'I', 'N', 'F', 'P', 'A', '0', '1'}};
constexpr std::uint32_t kPrefixArchiveVersion    = 1;
constexpr std::size_t kPrefixArchiveHeaderBytes  = 512;
constexpr std::uint64_t kPrefixArchiveFnvOffset  = 1469598103934665603ULL;
constexpr std::uint64_t kPrefixArchiveFnvPrime   = 1099511628211ULL;
// A corrupt or hostile file must not be able to make the loader allocate unbounded memory.
constexpr std::uint32_t kPrefixArchiveMaxChunks  = 1U << 20;
constexpr std::uint32_t kPrefixArchiveMaxBranches = 1U << 16;

std::uint64_t prefix_archive_hash(std::uint64_t hash, const void* data, std::size_t bytes) {
    const auto* bytes_view = static_cast<const unsigned char*>(data);
    for (std::size_t i = 0; i < bytes; ++i) {
        hash ^= bytes_view[i];
        hash *= kPrefixArchiveFnvPrime;
    }
    return hash;
}

std::string prefix_archive_strerror(int error) {
    return std::string(std::strerror(error));
}

class PrefixArchiveHeaderWriter {
public:
    void u32(std::uint32_t value) { append(&value, sizeof value); }
    void u64(std::uint64_t value) { append(&value, sizeof value); }
    void f64(double value) { append(&value, sizeof value); }
    void bytes(const void* data, std::size_t count) { append(data, count); }
    void str(const std::string& value, std::size_t width) {
        const std::size_t kept = std::min(width - 1, value.size());
        append(value.data(), kept);
        const std::vector<unsigned char> pad(width - kept, 0);
        append(pad.data(), pad.size());
    }
    [[nodiscard]] const std::vector<unsigned char>& data() const noexcept { return bytes_; }

private:
    void append(const void* data, std::size_t count) {
        if (count == 0) { return; }
        const auto* begin = static_cast<const unsigned char*>(data);
        bytes_.insert(bytes_.end(), begin, begin + count);
    }
    std::vector<unsigned char> bytes_;
};

class PrefixArchiveHeaderReader {
public:
    explicit PrefixArchiveHeaderReader(std::span<const unsigned char> bytes) : bytes_(bytes) {}

    void bytes(void* destination, std::size_t count) { take(destination, count); }
    std::uint32_t u32() {
        std::uint32_t value = 0;
        take(&value, sizeof value);
        return value;
    }
    std::uint64_t u64() {
        std::uint64_t value = 0;
        take(&value, sizeof value);
        return value;
    }
    double f64() {
        double value = 0.0;
        take(&value, sizeof value);
        return value;
    }
    std::string str(std::size_t width) {
        if (!ok_ || offset_ + width > bytes_.size()) {
            ok_ = false;
            return {};
        }
        std::string value(reinterpret_cast<const char*>(bytes_.data() + offset_), width);
        offset_ += width;
        const std::size_t end = value.find('\0');
        if (end != std::string::npos) { value.resize(end); }
        return value;
    }
    [[nodiscard]] bool ok() const noexcept { return ok_; }

private:
    void take(void* destination, std::size_t count) {
        if (!ok_ || offset_ + count > bytes_.size()) {
            ok_ = false;
            return;
        }
        std::memcpy(destination, bytes_.data() + offset_, count);
        offset_ += count;
    }
    std::span<const unsigned char> bytes_;
    std::size_t offset_ = 0;
    bool ok_            = true;
};

class PrefixArchiveBodyWriter {
public:
    explicit PrefixArchiveBodyWriter(std::FILE* file) : file_(file) {}

    void u32(std::uint32_t value) { raw(&value, sizeof value); }
    void i32(std::int32_t value) { raw(&value, sizeof value); }
    void u64(std::uint64_t value) { raw(&value, sizeof value); }
    void f64(double value) { raw(&value, sizeof value); }
    void bulk(const void* data, std::size_t count) { raw(data, count); }
    [[nodiscard]] bool ok() const noexcept { return ok_; }
    [[nodiscard]] std::uint64_t bytes() const noexcept { return bytes_; }
    [[nodiscard]] std::uint64_t hash() const noexcept { return hash_; }

private:
    void raw(const void* data, std::size_t count) {
        if (!ok_ || count == 0) { return; }
        if (std::fwrite(data, 1, count, file_) != count) {
            ok_ = false;
            return;
        }
        hash_ = prefix_archive_hash(hash_, data, count);
        bytes_ += count;
    }
    std::FILE* file_;
    bool ok_             = true;
    std::uint64_t bytes_ = 0;
    std::uint64_t hash_  = kPrefixArchiveFnvOffset;
};

class PrefixArchiveBodyReader {
public:
    explicit PrefixArchiveBodyReader(std::FILE* file) : file_(file) {}

    void bytes(void* destination, std::size_t count) { raw(destination, count); }
    std::uint32_t u32() {
        std::uint32_t value = 0;
        raw(&value, sizeof value);
        return value;
    }
    std::int32_t i32() {
        std::int32_t value = 0;
        raw(&value, sizeof value);
        return value;
    }
    std::uint64_t u64() {
        std::uint64_t value = 0;
        raw(&value, sizeof value);
        return value;
    }
    double f64() {
        double value = 0.0;
        raw(&value, sizeof value);
        return value;
    }
    [[nodiscard]] bool ok() const noexcept { return ok_; }
    [[nodiscard]] std::uint64_t bytes() const noexcept { return bytes_; }
    [[nodiscard]] std::uint64_t hash() const noexcept { return hash_; }

private:
    void raw(void* destination, std::size_t count) {
        if (!ok_ || count == 0) { return; }
        if (std::fread(destination, 1, count, file_) != count) {
            ok_ = false;
            return;
        }
        hash_ = prefix_archive_hash(hash_, destination, count);
        bytes_ += count;
    }
    std::FILE* file_;
    bool ok_             = true;
    std::uint64_t bytes_ = 0;
    std::uint64_t hash_  = kPrefixArchiveFnvOffset;
};

std::uint32_t prefix_archive_page_count(std::uint32_t tokens) noexcept {
    return tokens == 0 ? 0 : 1 + (tokens - 1) / static_cast<std::uint32_t>(kPagedKVPageSize);
}

std::size_t prefix_archive_chunk_bytes(const PrefixKVChunk& chunk, std::uint32_t ranks) {
    std::size_t bytes = 0;
    for (std::uint32_t rank = 0; rank < ranks; ++rank) {
        bytes += chunk.text[rank].size();
        bytes += chunk.backend[rank].size();
    }
    return bytes;
}

// Rank vectors beyond the execution device count must be empty, otherwise the image would be
// written with a different rank stride than the engine that produced it.
bool prefix_archive_chunk_ranks_match(const PrefixKVChunk& chunk, std::uint32_t ranks) {
    if (ranks == 0 || ranks > kMaximumExecutionDevices) { return false; }
    for (std::uint32_t rank = ranks; rank < kMaximumExecutionDevices; ++rank) {
        if (!chunk.text[rank].empty() || !chunk.backend[rank].empty()) { return false; }
    }
    return true;
}

// Everything that decides whether a chunk image still means the same thing. `kv_payload_bytes`
// folds the whole persistent KV layout (planes, dtype, heads, layers) into one number, so a
// layout change cannot slip through as "same model id".
struct PrefixArchiveStampFields {
    std::string build_id;
    std::string target;
    std::string model_id;
    std::string weights_id;
    std::uint32_t kv_dtype             = 0;
    std::uint32_t kv_quant_group       = 0;
    std::uint32_t page_size            = 0;
    std::uint32_t tp                   = 1;
    std::uint32_t pp                   = 1;
    std::uint32_t device_count         = 1;
    std::uint32_t max_concurrency      = 1;
    std::uint64_t kv_payload_bytes     = 0;
    std::uint64_t checkpoint_bytes     = 0;
    std::uint64_t gdn_state_bytes      = 0;
    std::uint64_t text_page_bytes      = 0;
    std::uint64_t backend_page_bytes   = 0;
    std::uint32_t capacity             = 0;
    std::uint32_t kv_capacity          = 0;
    std::uint32_t rope_mode            = 0;
    std::uint32_t yarn_origin          = 0;
    double yarn_factor                 = 0.0;
    std::uint32_t effective_max_context = 0;
    std::uint32_t max_context          = 0;
    double yarn_mscale                 = 1.0;
    std::uint32_t speculative_backend  = 0;
    std::uint32_t draft_window         = 0;
    std::uint32_t vision_enabled       = 0;
    std::uint32_t volta_long_attention = 0;
};

struct PrefixArchiveHeaderValues {
    std::uint64_t file_bytes           = 0;
    std::uint64_t body_bytes           = 0;
    std::uint64_t body_hash            = 0;
    std::uint64_t payload_bytes        = 0;
    std::uint32_t branch_count         = 0;
    std::uint32_t chunk_count          = 0;
    PrefixArchiveStampFields stamp;
};

PrefixArchiveHeaderWriter prefix_archive_header_bytes(const PrefixArchiveHeaderValues& values) {
    PrefixArchiveHeaderWriter header;
    header.bytes(kPrefixArchiveMagic.data(), kPrefixArchiveMagic.size());
    header.u32(kPrefixArchiveVersion);
    header.u32(static_cast<std::uint32_t>(kPrefixArchiveHeaderBytes));
    header.u64(values.file_bytes);
    header.u64(values.body_bytes);
    header.u64(values.body_hash);
    header.u64(values.payload_bytes);
    header.u32(values.branch_count);
    header.u32(values.chunk_count);
    header.str(values.stamp.build_id, 64);
    header.str(values.stamp.target, 32);
    header.str(values.stamp.model_id, 96);
    header.str(values.stamp.weights_id, 96);
    header.u32(values.stamp.kv_dtype);
    header.u32(values.stamp.kv_quant_group);
    header.u32(values.stamp.page_size);
    header.u32(values.stamp.tp);
    header.u32(values.stamp.pp);
    header.u32(values.stamp.device_count);
    header.u32(values.stamp.max_concurrency);
    header.u64(values.stamp.kv_payload_bytes);
    header.u64(values.stamp.checkpoint_bytes);
    header.u64(values.stamp.gdn_state_bytes);
    header.u64(values.stamp.text_page_bytes);
    header.u64(values.stamp.backend_page_bytes);
    header.u32(values.stamp.capacity);
    header.u32(values.stamp.kv_capacity);
    header.u32(values.stamp.rope_mode);
    header.u32(values.stamp.yarn_origin);
    header.f64(values.stamp.yarn_factor);
    header.u32(values.stamp.effective_max_context);
    header.u32(values.stamp.max_context);
    header.f64(values.stamp.yarn_mscale);
    header.u32(values.stamp.speculative_backend);
    header.u32(values.stamp.draft_window);
    header.u32(values.stamp.vision_enabled);
    header.u32(values.stamp.volta_long_attention);
    return header;
}

// Returns the first field that differs, or an empty string when the file belongs to this engine.
std::string prefix_archive_stamp_mismatch(const PrefixArchiveStampFields& expected,
                                          const PrefixArchiveStampFields& actual) {
    const auto u32 = [](const char* name, std::uint32_t left, std::uint32_t right) {
        return left == right ? std::string() : std::string(name) + " " + std::to_string(left) +
                                                   " != " + std::to_string(right);
    };
    const auto u64 = [](const char* name, std::uint64_t left, std::uint64_t right) {
        return left == right ? std::string() : std::string(name) + " " + std::to_string(left) +
                                                   " != " + std::to_string(right);
    };
    if (expected.build_id != actual.build_id) {
        return "build '" + actual.build_id + "' != '" + expected.build_id + "'";
    }
    if (expected.target != actual.target) { return "target '" + actual.target + "'"; }
    if (expected.model_id != actual.model_id) { return "model '" + actual.model_id + "'"; }
    if (expected.weights_id != actual.weights_id) { return "weights '" + actual.weights_id + "'"; }
    for (const std::string& mismatch :
         {u32("kv_dtype", expected.kv_dtype, actual.kv_dtype),
          u32("kv_group", expected.kv_quant_group, actual.kv_quant_group),
          u32("page_size", expected.page_size, actual.page_size),
          u32("tp", expected.tp, actual.tp),
          u32("pp", expected.pp, actual.pp),
          u32("devices", expected.device_count, actual.device_count),
          u32("max_concurrency", expected.max_concurrency, actual.max_concurrency),
          u64("kv_payload_bytes", expected.kv_payload_bytes, actual.kv_payload_bytes),
          u64("checkpoint_bytes", expected.checkpoint_bytes, actual.checkpoint_bytes),
          u64("gdn_state_bytes", expected.gdn_state_bytes, actual.gdn_state_bytes),
          u32("capacity", expected.capacity, actual.capacity),
          u32("kv_capacity", expected.kv_capacity, actual.kv_capacity),
          u32("rope_mode", expected.rope_mode, actual.rope_mode),
          u32("yarn_origin", expected.yarn_origin, actual.yarn_origin),
          u32("effective_max_context", expected.effective_max_context, actual.effective_max_context),
          u32("max_context", expected.max_context, actual.max_context),
          u32("spec", expected.speculative_backend, actual.speculative_backend),
          u32("draft_window", expected.draft_window, actual.draft_window),
          u32("vision", expected.vision_enabled, actual.vision_enabled),
          u32("volta_long_attention", expected.volta_long_attention, actual.volta_long_attention)}) {
        if (!mismatch.empty()) { return mismatch; }
    }
    if (expected.yarn_factor != actual.yarn_factor || expected.yarn_mscale != actual.yarn_mscale) {
        return "yarn regime";
    }
    return {};
}

PrefixArchiveHeaderValues prefix_archive_parse_header(std::span<const unsigned char> raw,
                                                      bool* ok) {
    PrefixArchiveHeaderValues values;
    PrefixArchiveHeaderReader header(raw);
    std::array<unsigned char, 8> magic{};
    header.bytes(magic.data(), magic.size());
    const std::uint32_t version      = header.u32();
    const std::uint32_t header_bytes = header.u32();
    values.file_bytes                = header.u64();
    values.body_bytes                = header.u64();
    values.body_hash                 = header.u64();
    values.payload_bytes             = header.u64();
    values.branch_count              = header.u32();
    values.chunk_count               = header.u32();
    values.stamp.build_id            = header.str(64);
    values.stamp.target              = header.str(32);
    values.stamp.model_id            = header.str(96);
    values.stamp.weights_id          = header.str(96);
    values.stamp.kv_dtype            = header.u32();
    values.stamp.kv_quant_group      = header.u32();
    values.stamp.page_size           = header.u32();
    values.stamp.tp                  = header.u32();
    values.stamp.pp                  = header.u32();
    values.stamp.device_count        = header.u32();
    values.stamp.max_concurrency     = header.u32();
    values.stamp.kv_payload_bytes    = header.u64();
    values.stamp.checkpoint_bytes    = header.u64();
    values.stamp.gdn_state_bytes     = header.u64();
    values.stamp.text_page_bytes     = header.u64();
    values.stamp.backend_page_bytes  = header.u64();
    values.stamp.capacity            = header.u32();
    values.stamp.kv_capacity         = header.u32();
    values.stamp.rope_mode           = header.u32();
    values.stamp.yarn_origin         = header.u32();
    values.stamp.yarn_factor         = header.f64();
    values.stamp.effective_max_context = header.u32();
    values.stamp.max_context         = header.u32();
    values.stamp.yarn_mscale         = header.f64();
    values.stamp.speculative_backend = header.u32();
    values.stamp.draft_window        = header.u32();
    values.stamp.vision_enabled      = header.u32();
    values.stamp.volta_long_attention = header.u32();
    *ok = header.ok() && magic == kPrefixArchiveMagic && version == kPrefixArchiveVersion &&
          header_bytes == kPrefixArchiveHeaderBytes;
    return values;
}

} // namespace

// Writes the newest branches that fit `max_bytes` (0 means no limit). A single branch larger than
// the budget is still written: the newest branch is the one the next session is most likely to
// extend, and refusing it would make the budget self-defeating.
inline PrefixArchiveStats ProgramImplCore::save_prefix_archive(const std::string& path,
                                                              std::size_t max_bytes,
                                                              const PrefixArchiveStamp& stamp) {
    PrefixArchiveStats stats;
    if (path.empty()) {
        stats.message = "no prefix archive path configured";
        return stats;
    }
    if (prefix_cache_tokens == 0 || prefix_history.empty()) {
        stats.message = "prefix history is empty";
        return stats;
    }
    if (ram_kv.gpu_tokens != 0) {
        stats.message = "RAM-KV windows keep no complete host KV image";
        return stats;
    }
    const std::uint32_t ranks = static_cast<std::uint32_t>(device_count);
    const std::size_t checkpoint_bytes = prefix_checkpoint_bytes();
    if (checkpoint_bytes == 0) {
        stats.message = "prefix state images are unavailable";
        return stats;
    }

    // A retained lane only enters prefix_history when it is evicted or reused, so at shutdown the
    // newest conversation is still resident. Capture it first: that is exactly the prefix the next
    // session extends, and leaving it out would persist only stale branches.
    std::string capture_error;
    for (auto& sequence : sequences) {
        if (!sequence.retained) { continue; }
        try {
            archive_prefix(sequence);
        } catch (const std::exception& error) {
            capture_error = error.what();
            break;
        }
    }
    if (prefix_history.empty()) {
        stats.message = capture_error.empty()
                            ? std::string("no retained lane to archive")
                            : "cannot capture the resident prefix: " + capture_error;
        return stats;
    }

    std::vector<std::shared_ptr<ArchivedPrefix>> ordered;
    ordered.reserve(prefix_history.size());
    for (const auto& branch : prefix_history) {
        if (branch != nullptr) { ordered.push_back(branch); }
    }
    std::stable_sort(ordered.begin(), ordered.end(),
                     [](const std::shared_ptr<ArchivedPrefix>& left,
                        const std::shared_ptr<ArchivedPrefix>& right) {
                         return left->frontier > right->frontier;
                     });

    const auto known = [](const auto& list, const auto* pointer) {
        return std::find(list.begin(), list.end(), pointer) != list.end();
    };
    std::vector<std::shared_ptr<ArchivedPrefix>> chosen;
    std::vector<const PrefixKVChunk*> chunks;
    std::vector<const std::byte*> images;
    std::size_t payload = 0;
    for (const auto& branch : ordered) {
        const std::uint32_t frontier = branch->frontier;
        if (frontier == 0 || branch->ledger.size() < frontier || branch->identity.size() < frontier ||
            branch->mtp_valid > frontier) {
            continue;
        }
        std::size_t branch_bytes = 0;
        bool consistent          = true;
        for (const auto& chunk : branch->chunks) {
            if (chunk == nullptr || !prefix_archive_chunk_ranks_match(*chunk, ranks) ||
                chunk->pages == 0 ||
                chunk->first_page + chunk->pages > prefix_archive_page_count(frontier)) {
                consistent = false;
                break;
            }
            if (!known(chunks, chunk.get())) {
                branch_bytes += prefix_archive_chunk_bytes(*chunk, ranks);
            }
        }
        if (!consistent) { continue; }
        for (const auto& checkpoint : branch->checkpoints) {
            if (checkpoint.frontier == 0 || !checkpoint.image) {
                consistent = false;
                break;
            }
            if (!known(images, checkpoint.image.get())) { branch_bytes += checkpoint_bytes; }
        }
        if (!consistent) { continue; }
        if (max_bytes != 0 && !chosen.empty() && payload + branch_bytes > max_bytes) { continue; }
        for (const auto& chunk : branch->chunks) {
            if (!known(chunks, chunk.get())) { chunks.push_back(chunk.get()); }
        }
        for (const auto& checkpoint : branch->checkpoints) {
            if (!known(images, checkpoint.image.get())) { images.push_back(checkpoint.image.get()); }
        }
        payload += branch_bytes;
        chosen.push_back(branch);
        if (max_bytes != 0 && payload >= max_bytes) { break; }
    }
    if (chosen.empty()) {
        stats.message = "no consistent branch to save";
        return stats;
    }

    // Page geometry, replayed verbatim on load so a chunk whose byte count does not divide into
    // whole pages is rejected before it reaches an import that would throw mid-request.
    std::uint64_t text_page_bytes    = 0;
    std::uint64_t backend_page_bytes = 0;
    for (const PrefixKVChunk* chunk : chunks) {
        if (text_page_bytes == 0 && chunk->pages != 0 && chunk->text[0].size() % chunk->pages == 0) {
            text_page_bytes = chunk->text[0].size() / chunk->pages;
        }
        if (backend_page_bytes == 0 && chunk->backend_pages != 0 &&
            chunk->backend[0].size() % chunk->backend_pages == 0) {
            backend_page_bytes = chunk->backend[0].size() / chunk->backend_pages;
        }
    }

    const std::filesystem::path target(path);
    if (target.has_parent_path()) {
        std::error_code ignored;
        std::filesystem::create_directories(target.parent_path(), ignored);
    }
    const std::string temporary = path + ".tmp";
    std::FILE* file             = std::fopen(temporary.c_str(), "wb");
    if (file == nullptr) {
        stats.message = "cannot create " + temporary + ": " + prefix_archive_strerror(errno);
        return stats;
    }
    const std::array<unsigned char, kPrefixArchiveHeaderBytes> placeholder{};
    if (std::fwrite(placeholder.data(), 1, placeholder.size(), file) != placeholder.size()) {
        stats.message = "cannot reserve the archive header: " + prefix_archive_strerror(errno);
        std::fclose(file);
        std::remove(temporary.c_str());
        return stats;
    }

    {
        PrefixArchiveBodyWriter body(file);
        body.u32(static_cast<std::uint32_t>(chunks.size()));
        for (const PrefixKVChunk* chunk : chunks) {
            body.u32(chunk->first_page);
            body.u32(chunk->pages);
            body.u32(chunk->backend_pages);
            body.u32(ranks);
            for (std::uint32_t rank = 0; rank < ranks; ++rank) {
                body.u64(chunk->text[rank].size());
                body.bulk(chunk->text[rank].data(), chunk->text[rank].size());
                body.u64(chunk->backend[rank].size());
                body.bulk(chunk->backend[rank].data(), chunk->backend[rank].size());
            }
        }
        body.u32(static_cast<std::uint32_t>(chosen.size()));
        for (const auto& branch : chosen) {
            const std::uint32_t frontier = branch->frontier;
            body.u32(frontier);
            body.u32(branch->mtp_valid);
            body.u32(frontier);
            for (std::uint32_t index = 0; index < frontier; ++index) {
                body.i32(branch->ledger[index]);
            }
            const auto& identity = branch->identity;
            body.u32(frontier);
            body.bulk(identity.token_types().data(), frontier);
            for (const auto& axis : identity.positions()) {
                body.bulk(axis.data(), static_cast<std::size_t>(frontier) * sizeof(std::int32_t));
            }
            const auto& items = identity.vision_items();
            body.u32(static_cast<std::uint32_t>(items.size()));
            for (const VisionItem& item : items) {
                body.u32(static_cast<std::uint32_t>(item.modality));
                body.i32(item.grid.temporal);
                body.i32(item.grid.height);
                body.i32(item.grid.width);
                body.u64(item.patch_begin);
                body.u64(item.patch_count);
                body.bulk(item.content_digest.data(), item.content_digest.size());
                body.u32(static_cast<std::uint32_t>(item.timestamps.size()));
                for (const double timestamp : item.timestamps) { body.f64(timestamp); }
                body.u32(static_cast<std::uint32_t>(item.token_spans.size()));
                for (const TokenSpan& span : item.token_spans) {
                    body.u64(span.begin);
                    body.u64(span.count);
                }
            }
            body.u32(static_cast<std::uint32_t>(branch->checkpoints.size()));
            for (const auto& checkpoint : branch->checkpoints) {
                body.u32(checkpoint.frontier);
                body.bulk(checkpoint.image.get(), checkpoint_bytes);
            }
            body.u32(static_cast<std::uint32_t>(branch->chunks.size()));
            for (const auto& chunk : branch->chunks) {
                const auto found = std::find(chunks.begin(), chunks.end(), chunk.get());
                body.u32(static_cast<std::uint32_t>(std::distance(chunks.begin(), found)));
            }
        }
        if (!body.ok()) {
            stats.message = "write failed: " + prefix_archive_strerror(errno);
            std::fclose(file);
            std::remove(temporary.c_str());
            return stats;
        }
        stats.bytes = static_cast<std::size_t>(body.bytes());
        stats.chunks = chunks.size();
        stats.branches = chosen.size();

        PrefixArchiveStampFields fields;
        fields.build_id             = stamp.build_id;
        fields.target               = stamp.target;
        fields.model_id             = stamp.model_id;
        fields.weights_id           = stamp.weights_id;
        fields.kv_dtype             = static_cast<std::uint32_t>(kv_dtype);
        fields.kv_quant_group       = static_cast<std::uint32_t>(kv_quant_group);
        fields.page_size            = static_cast<std::uint32_t>(kPagedKVPageSize);
        fields.tp                   = stamp.tp == 0 ? static_cast<std::uint32_t>(device_count) : stamp.tp;
        fields.pp                   = static_cast<std::uint32_t>(pp);
        fields.device_count         = static_cast<std::uint32_t>(device_count);
        fields.max_concurrency      = max_concurrency;
        fields.kv_payload_bytes     = kv_payload_bytes;
        fields.checkpoint_bytes     = checkpoint_bytes;
        fields.gdn_state_bytes      = gdn_state_bytes;
        fields.text_page_bytes      = text_page_bytes;
        fields.backend_page_bytes   = backend_page_bytes;
        fields.capacity             = capacity;
        fields.kv_capacity          = kv_capacity;
        fields.rope_mode            = stamp.rope_mode;
        fields.yarn_origin          = stamp.yarn_origin;
        fields.yarn_factor          = stamp.yarn_factor;
        fields.effective_max_context = effective_max_context;
        fields.max_context          = stamp.max_context;
        fields.yarn_mscale          = yarn_mscale;
        fields.speculative_backend  = static_cast<std::uint32_t>(speculative_backend);
        fields.draft_window         = draft_window;
        fields.vision_enabled       = vision_enabled ? 1U : 0U;
        fields.volta_long_attention = volta_long_attention ? 1U : 0U;

        PrefixArchiveHeaderValues values;
        values.body_bytes   = body.bytes();
        values.body_hash    = body.hash();
        values.payload_bytes = static_cast<std::uint64_t>(payload);
        values.branch_count = static_cast<std::uint32_t>(chosen.size());
        values.chunk_count  = static_cast<std::uint32_t>(chunks.size());
        values.stamp        = fields;
        values.file_bytes   = kPrefixArchiveHeaderBytes + values.body_bytes;
        const PrefixArchiveHeaderWriter header = prefix_archive_header_bytes(values);
        if (header.data().size() > kPrefixArchiveHeaderBytes) {
            stats.message = "archive header overflow";
            std::fclose(file);
            std::remove(temporary.c_str());
            return stats;
        }
        if (std::fflush(file) != 0 || ::fsync(::fileno(file)) != 0 ||
            std::fseek(file, 0, SEEK_SET) != 0 ||
            std::fwrite(header.data().data(), 1, header.data().size(), file) !=
                header.data().size() ||
            std::fflush(file) != 0 || ::fsync(::fileno(file)) != 0) {
            stats.message = "finalize failed: " + prefix_archive_strerror(errno);
            std::fclose(file);
            std::remove(temporary.c_str());
            return stats;
        }
    }
    if (std::fclose(file) != 0) {
        stats.message = "close failed: " + prefix_archive_strerror(errno);
        std::remove(temporary.c_str());
        return stats;
    }
    if (std::rename(temporary.c_str(), path.c_str()) != 0) {
        stats.message = "rename failed: " + prefix_archive_strerror(errno);
        std::remove(temporary.c_str());
        return stats;
    }
    stats.ok = true;
    if (!capture_error.empty()) { stats.message = "partial capture: " + capture_error; }
    return stats;
}

inline PrefixArchiveStats ProgramImplCore::load_prefix_archive(const std::string& path,
                                                              const PrefixArchiveStamp& stamp) {
    PrefixArchiveStats stats;
    if (path.empty()) {
        stats.message = "no prefix archive path configured";
        return stats;
    }
    std::FILE* file = std::fopen(path.c_str(), "rb");
    if (file == nullptr) {
        stats.message = "no prefix archive at " + path;
        return stats;
    }
    const auto reject = [&](std::string message) {
        std::fclose(file);
        stats.ok = false;
        stats.message = std::move(message);
        return stats;
    };

    std::array<unsigned char, kPrefixArchiveHeaderBytes> raw{};
    if (std::fread(raw.data(), 1, raw.size(), file) != raw.size()) {
        return reject("truncated header");
    }
    bool header_ok = false;
    const PrefixArchiveHeaderValues values = prefix_archive_parse_header(raw, &header_ok);
    if (!header_ok) { return reject("header is not a prefix archive this build understands"); }

    PrefixArchiveStampFields expected;
    expected.build_id             = stamp.build_id;
    expected.target               = stamp.target;
    expected.model_id             = stamp.model_id;
    expected.weights_id           = stamp.weights_id;
    expected.kv_dtype             = static_cast<std::uint32_t>(kv_dtype);
    expected.kv_quant_group       = static_cast<std::uint32_t>(kv_quant_group);
    expected.page_size            = static_cast<std::uint32_t>(kPagedKVPageSize);
    expected.tp                   = stamp.tp == 0 ? static_cast<std::uint32_t>(device_count) : stamp.tp;
    expected.pp                   = static_cast<std::uint32_t>(pp);
    expected.device_count         = static_cast<std::uint32_t>(device_count);
    expected.max_concurrency      = max_concurrency;
    expected.kv_payload_bytes     = kv_payload_bytes;
    expected.checkpoint_bytes     = prefix_checkpoint_bytes();
    expected.gdn_state_bytes      = gdn_state_bytes;
    expected.capacity             = capacity;
    expected.kv_capacity          = kv_capacity;
    expected.rope_mode            = stamp.rope_mode;
    expected.yarn_origin          = stamp.yarn_origin;
    expected.yarn_factor          = stamp.yarn_factor;
    expected.effective_max_context = effective_max_context;
    expected.max_context          = stamp.max_context;
    expected.yarn_mscale          = yarn_mscale;
    expected.speculative_backend  = static_cast<std::uint32_t>(speculative_backend);
    expected.draft_window         = draft_window;
    expected.vision_enabled       = vision_enabled ? 1U : 0U;
    expected.volta_long_attention = volta_long_attention ? 1U : 0U;

    if (const std::string mismatch = prefix_archive_stamp_mismatch(expected, values.stamp);
        !mismatch.empty()) {
        return reject("stale archive rejected: " + mismatch);
    }
    if (values.chunk_count > kPrefixArchiveMaxChunks || values.branch_count > kPrefixArchiveMaxBranches) {
        return reject("archive counts are out of range");
    }
    if (values.stamp.text_page_bytes == 0 && values.chunk_count != 0) {
        return reject("archive has no KV page geometry");
    }
    // The header carries no checksum of its own, so bound everything it can steer an allocation
    // with: a page is a few MiB in this engine, and no image may exceed the file that holds it.
    if (values.stamp.text_page_bytes > (1ULL << 24) ||
        values.stamp.backend_page_bytes > (1ULL << 24)) {
        return reject("archive page geometry is implausible");
    }
    struct stat file_info {};
    if (::fstat(::fileno(file), &file_info) != 0 ||
        static_cast<std::uint64_t>(file_info.st_size) != values.file_bytes) {
        return reject("archive length does not match its header");
    }

    PrefixArchiveBodyReader body(file);
    const std::uint32_t chunk_count = body.u32();
    if (!body.ok() || chunk_count != values.chunk_count) { return reject("truncated chunk table"); }
    std::vector<std::shared_ptr<PrefixKVChunk>> chunks;
    chunks.reserve(chunk_count);
    for (std::uint32_t index = 0; index < chunk_count; ++index) {
        auto chunk        = std::make_shared<PrefixKVChunk>();
        chunk->first_page = body.u32();
        chunk->pages      = body.u32();
        const std::uint32_t backend_pages = body.u32();
        const std::uint32_t ranks         = body.u32();
        chunk->backend_pages              = backend_pages;
        if (!body.ok() || ranks != static_cast<std::uint32_t>(device_count) || chunk->pages == 0 ||
            chunk->pages > capacity) {
            return reject("chunk header is inconsistent");
        }
        for (std::uint32_t rank = 0; rank < ranks; ++rank) {
            const std::uint64_t text_bytes = body.u64();
            if (!body.ok() || text_bytes > values.file_bytes ||
                text_bytes != static_cast<std::uint64_t>(chunk->pages) * values.stamp.text_page_bytes) {
                return reject("chunk KV image size does not match the stamped page geometry");
            }
            chunk->text[rank].resize(static_cast<std::size_t>(text_bytes));
            body.bytes(chunk->text[rank].data(), chunk->text[rank].size());
            const std::uint64_t backend_bytes = body.u64();
            if (!body.ok() || backend_bytes > values.file_bytes ||
                backend_bytes !=
                    static_cast<std::uint64_t>(chunk->backend_pages) * values.stamp.backend_page_bytes) {
                return reject("chunk MTP image size does not match the stamped page geometry");
            }
            chunk->backend[rank].resize(static_cast<std::size_t>(backend_bytes));
            body.bytes(chunk->backend[rank].data(), chunk->backend[rank].size());
        }
        if (!body.ok()) { return reject("truncated chunk image"); }
        chunks.push_back(std::move(chunk));
    }

    const std::uint32_t branch_count = body.u32();
    if (!body.ok() || branch_count != values.branch_count) { return reject("truncated branch table"); }
    std::vector<std::shared_ptr<ArchivedPrefix>> loaded;
    loaded.reserve(branch_count);
    for (std::uint32_t index = 0; index < branch_count; ++index) {
        auto branch      = std::make_shared<ArchivedPrefix>();
        branch->frontier = body.u32();
        branch->mtp_valid = body.u32();
        const std::uint32_t ledger_count = body.u32();
        if (!body.ok() || ledger_count != branch->frontier || branch->mtp_valid > branch->frontier ||
            branch->frontier > capacity) {
            return reject("branch header is inconsistent");
        }
        branch->ledger.resize(ledger_count);
        body.bytes(branch->ledger.data(), branch->ledger.size() * sizeof(TokenId));
        const std::uint32_t identity_tokens = body.u32();
        if (!body.ok() || identity_tokens != branch->frontier) {
            return reject("branch identity does not cover its frontier");
        }
        std::vector<std::uint8_t> token_types(identity_tokens);
        body.bytes(token_types.data(), token_types.size());
        std::array<std::vector<std::int32_t>, 3> positions;
        for (auto& axis : positions) {
            axis.resize(identity_tokens);
            body.bytes(axis.data(), axis.size() * sizeof(std::int32_t));
        }
        const std::uint32_t item_count = body.u32();
        if (!body.ok() || item_count > (1U << 16)) { return reject("identity item count is out of range"); }
        std::vector<VisionItem> items(item_count);
        for (VisionItem& item : items) {
            item.modality = static_cast<PromptModality>(body.u32());
            item.grid.temporal = body.i32();
            item.grid.height   = body.i32();
            item.grid.width    = body.i32();
            item.patch_begin   = static_cast<std::size_t>(body.u64());
            item.patch_count   = static_cast<std::size_t>(body.u64());
            body.bytes(item.content_digest.data(), item.content_digest.size());
            const std::uint32_t timestamps = body.u32();
            if (!body.ok() || timestamps > (1U << 20)) { return reject("identity timestamps are out of range"); }
            item.timestamps.resize(timestamps);
            for (double& timestamp : item.timestamps) { timestamp = body.f64(); }
            const std::uint32_t spans = body.u32();
            if (!body.ok() || spans > (1U << 20)) { return reject("identity spans are out of range"); }
            item.token_spans.resize(spans);
            for (TokenSpan& span : item.token_spans) {
                span.begin = static_cast<std::size_t>(body.u64());
                span.count = static_cast<std::size_t>(body.u64());
            }
        }
        if (!body.ok()) { return reject("truncated branch identity"); }
        branch->identity.assign_raw(std::move(token_types), std::move(positions), std::move(items));

        const std::uint32_t checkpoint_count = body.u32();
        if (!body.ok() || checkpoint_count > (1U << 16)) {
            return reject("checkpoint count is out of range");
        }
        branch->checkpoints.resize(checkpoint_count);
        for (auto& checkpoint : branch->checkpoints) {
            checkpoint.frontier = body.u32();
            checkpoint.image =
                std::shared_ptr<std::byte[]>(new std::byte[prefix_checkpoint_bytes()]);
            body.bytes(checkpoint.image.get(), prefix_checkpoint_bytes());
            if (!body.ok() || checkpoint.frontier == 0 || checkpoint.frontier > branch->frontier) {
                return reject("checkpoint image is inconsistent");
            }
        }
        const std::uint32_t branch_chunks = body.u32();
        if (!body.ok() || branch_chunks > chunk_count) { return reject("branch chunk list is out of range"); }
        branch->chunks.reserve(branch_chunks);
        for (std::uint32_t i = 0; i < branch_chunks; ++i) {
            const std::uint32_t id = body.u32();
            if (!body.ok() || id >= chunks.size()) { return reject("branch names an unknown chunk"); }
            branch->chunks.push_back(chunks[id]);
        }
        for (const auto& chunk : branch->chunks) {
            if (chunk->first_page + chunk->pages > prefix_archive_page_count(branch->frontier)) {
                return reject("chunk reaches past its branch frontier");
            }
        }
        loaded.push_back(std::move(branch));
    }

    if (!body.ok() || body.bytes() != values.body_bytes || body.hash() != values.body_hash) {
        return reject("archive body is truncated or corrupt");
    }
    std::fclose(file);

    for (auto& branch : loaded) { prefix_history.push_back(std::move(branch)); }
    std::stable_sort(prefix_history.begin(), prefix_history.end(),
                     [](const std::shared_ptr<ArchivedPrefix>& left,
                        const std::shared_ptr<ArchivedPrefix>& right) {
                         return left->frontier < right->frontier;
                     });
    trim_prefix_history();
    stats.ok       = true;
    stats.branches = loaded.size();
    stats.chunks   = chunks.size();
    stats.bytes    = static_cast<std::size_t>(values.body_bytes);
    return stats;
}

} // namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS
