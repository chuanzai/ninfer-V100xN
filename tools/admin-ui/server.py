#!/usr/bin/env python3
"""A Strata-style admin console for a running ninfer-serve.

Everything shown by the console comes from the live service: /health,
/ninfer/model, the JSONL request log, and host telemetry (nvidia-smi, free,
systemctl). Python standard library only, so it can run on the Mac or next to
the inference process on the GPU host without touching the C++ binary.

Examples:
  # from a Mac, talking to the GPU host over SSH
  python3 tools/admin-ui/server.py

  # on the GPU host itself
  NINFER_TARGET_HOST=local NINFER_BASE_URL=http://127.0.0.1:8080 \
    python3 tools/admin-ui/server.py --host 0.0.0.0 --port 8081
"""

from __future__ import annotations

import argparse
import gzip
import http.client
import json
import os
import re
import shlex
import subprocess
import sys
import threading
import time
import urllib.request
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

BASE_URL = os.environ.get("NINFER_BASE_URL", "http://192.168.11.6:8080")
TARGET_HOST = os.environ.get("NINFER_TARGET_HOST", "192.168.11.6")
SSH_USER = os.environ.get("NINFER_SSH_USER", "chuanzai")
LOG_PATH = os.environ.get(
    "NINFER_LOG_PATH", "/home/chuanzai/ninfer-V100X2/profiles/serve.jsonl"
)
SERVICE_NAME = os.environ.get("NINFER_SERVICE", "ninfer-serve")
ADMIN_TOKEN = os.environ.get("NINFER_ADMIN_TOKEN", "")

STATIC_DIR = Path(__file__).resolve().parent
OVERVIEW_TTL = 2.0
_CACHE: dict[str, object] = {"at": 0.0, "data": None}
_CACHE_LOCK = threading.Lock()


# --------------------------------------------------------------------------- #
# Data sources
# --------------------------------------------------------------------------- #
def is_local() -> bool:
    return TARGET_HOST.lower() in {"local", "localhost", "127.0.0.1"}


def remote(command: str, timeout: float = 8.0) -> subprocess.CompletedProcess[str]:
    """Run a telemetry command on the inference host (directly or over SSH)."""
    argv = (
        ["/bin/sh", "-c", command]
        if is_local()
        else [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=4",
            f"{SSH_USER}@{TARGET_HOST}",
            command,
        ]
    )
    return subprocess.run(argv, capture_output=True, text=True, timeout=timeout)


def remote_raw(command: str, timeout: float = 8.0) -> subprocess.CompletedProcess[bytes]:
    """Same as remote(), but keeps stdout binary (used for compressed log reads)."""
    argv = (
        ["/bin/sh", "-c", command]
        if is_local()
        else [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=4",
            f"{SSH_USER}@{TARGET_HOST}",
            command,
        ]
    )
    return subprocess.run(argv, capture_output=True, timeout=timeout)


def http_json(path: str, timeout: float = 3.0) -> dict:
    url = BASE_URL.rstrip("/") + path
    request = urllib.request.Request(url, headers={"Connection": "close", "Accept": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8", "replace"))
    except Exception as exc:  # the console must stay usable when the engine is down
        return {"error": str(exc)}


def tail_log(lines: int = 500) -> list[str]:
    command = f"tail -n {int(lines)} {shlex.quote(LOG_PATH)}"
    result = remote(command)
    if result.returncode != 0:
        return []
    return [line for line in result.stdout.splitlines() if line.strip()]


def machine_info() -> dict:
    result = remote(
        "nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,"
        "temperature.gpu,power.draw,power.limit,pcie.link.gen.current,pcie.link.gen.max,"
        "pcie.link.width.current,pcie.link.width.max --format=csv,noheader,nounits"
    )
    gpus = []
    for line in result.stdout.splitlines():
        parts = [p.strip().replace("N/A", "0") for p in line.split(",")]
        if len(parts) < 7:
            continue
        try:
            gpus.append(
                {
                    "index": int(parts[0]),
                    "name": parts[1],
                    "vram_used_mib": float(parts[2]),
                    "vram_total_mib": float(parts[3]),
                    "util_pct": float(parts[4]),
                    "temp_c": float(parts[5]),
                    "power_w": float(parts[6]),
                    "power_limit_w": float(parts[7]) if len(parts) > 7 else None,
                    "pcie_gen": int(float(parts[8])) if len(parts) > 8 else None,
                    "pcie_gen_max": int(float(parts[9])) if len(parts) > 9 else None,
                    "pcie_width": int(float(parts[10])) if len(parts) > 10 else None,
                    "pcie_width_max": int(float(parts[11])) if len(parts) > 11 else None,
                }
            )
        except ValueError:
            continue

    free = remote("free -b")
    ram = {}
    for line in free.stdout.splitlines():
        if line.startswith("Mem:"):
            fields = line.split()
            if len(fields) >= 3:
                ram = {"total_bytes": int(fields[1]), "used_bytes": int(fields[2])}

    used = sum(g["vram_used_mib"] for g in gpus)
    total = sum(g["vram_total_mib"] for g in gpus)
    weakest = min(
        (g for g in gpus if g.get("pcie_width") is not None),
        key=lambda g: (g.get("pcie_gen") or 0, g.get("pcie_width") or 0),
        default=None,
    )
    host = remote("hostname").stdout.strip()
    try:
        threads = int(remote("nproc").stdout.strip())
    except ValueError:
        threads = None
    return {
        "host": host or TARGET_HOST,
        "gpus": gpus,
        "ram": ram,
        "gpu_util": max((g["util_pct"] for g in gpus), default=None),
        "gpu_mem_used": used,
        "gpu_mem_total": total,
        "gpu_temp": max((g["temp_c"] for g in gpus), default=None),
        "gpu_power": sum(g["power_w"] for g in gpus),
        "gpu_power_limit": sum(
            g["power_limit_w"] for g in gpus if g.get("power_limit_w") is not None
        ) or None,
        "gpu_pcie_gen": weakest.get("pcie_gen") if weakest else None,
        "gpu_pcie_gen_max": weakest.get("pcie_gen_max") if weakest else None,
        "gpu_pcie_width": weakest.get("pcie_width") if weakest else None,
        "gpu_pcie_width_max": weakest.get("pcie_width_max") if weakest else None,
        "gpu_pcie_bottleneck": weakest,
        "pcie": weakest,
        "cores": None,
        "threads": threads,
    }


def cpu_disk_sample(previous: tuple[int, int] | None) -> tuple[dict, tuple[int, int]]:
    """Sample CPU utilisation and disk throughput from /proc on Linux hosts."""
    command = (
        "A=$(awk '/^cpu /{t=0;for(i=2;i<=NF;i++)t+=$i;print t,$5+$6}' /proc/stat); "
        "sleep 0.2; B=$(awk '/^cpu /{t=0;for(i=2;i<=NF;i++)t+=$i;print t,$5+$6}' /proc/stat); "
        "echo $A $B; awk '{r+=$6;w+=$10} END{print r*512,w*512}' /proc/diskstats"
    )
    result = remote(command, timeout=6)
    fields = result.stdout.split()
    if len(fields) < 6:
        return {"cpu_pct": None, "disk_read_mb": None, "disk_write_mb": None}, previous or (0, 0)
    try:
        t1, i1, t2, i2 = (int(float(x)) for x in fields[:4])
        cpu_pct = 100.0 * (1 - (i2 - i1) / (t2 - t1)) if t2 > t1 else 0.0
        read, write = int(float(fields[4])), int(float(fields[5]))
        if previous:
            delta_t = 0.2
            read_mb = max(0, read - previous[0]) / 1048576 / delta_t
            write_mb = max(0, write - previous[1]) / 1048576 / delta_t
        else:
            read_mb = write_mb = 0.0
        return (
            {"cpu_pct": round(cpu_pct, 1), "disk_read_mb": round(read_mb, 2),
             "disk_write_mb": round(write_mb, 2)},
            (read, write),
        )
    except (ValueError, ZeroDivisionError):
        return {"cpu_pct": None, "disk_read_mb": None, "disk_write_mb": None}, previous or (0, 0)


class HistoryStore:
    """Keeps a small rolling window of host metrics for the per-card sparklines."""

    KEYS = (
        "gpu_util", "gpu_mem_used", "gpu_mem_total", "gpu_temp", "gpu_power",
        "pcie_gen", "pcie_width", "cpu_pct", "ram_pct", "disk_read_mb", "disk_write_mb",
    )

    def __init__(self, maxlen: int = 90, interval: float = 2.0) -> None:
        self.samples: deque[dict] = deque(maxlen=maxlen)
        self.interval = interval
        self.lock = threading.Lock()
        self.previous_disk: tuple[int, int] | None = None

    def start(self) -> None:
        def loop() -> None:
            while True:
                try:
                    machine = machine_info()
                    counters, self.previous_disk = cpu_disk_sample(self.previous_disk)
                    ram = machine.get("ram") or {}
                    ram_pct = (
                        100.0 * ram["used_bytes"] / ram["total_bytes"]
                        if ram.get("total_bytes") else None
                    )
                    pcie = machine.get("pcie") or {}
                    sample = {
                        "gpu_util": machine.get("gpu_util"),
                        "gpu_mem_used": machine.get("gpu_mem_used"),
                        "gpu_mem_total": machine.get("gpu_mem_total"),
                        "gpu_temp": machine.get("gpu_temp"),
                        "gpu_power": machine.get("gpu_power"),
                        "pcie_gen": pcie.get("pcie_gen"),
                        "pcie_width": pcie.get("pcie_width"),
                        "cpu_pct": counters.get("cpu_pct"),
                        "ram_pct": round(ram_pct, 1) if ram_pct is not None else None,
                        "disk_read_mb": counters.get("disk_read_mb"),
                        "disk_write_mb": counters.get("disk_write_mb"),
                    }
                    with self.lock:
                        self.samples.append(sample)
                except Exception:
                    pass
                time.sleep(self.interval)

        threading.Thread(target=loop, name="telemetry-history", daemon=True).start()

    def snapshot(self) -> dict[str, list]:
        with self.lock:
            samples = list(self.samples)
        return {key: [sample.get(key) for sample in samples] for key in self.KEYS}


HISTORY = HistoryStore()


def service_info() -> dict:
    result = remote(
        f"systemctl show {shlex.quote(SERVICE_NAME)} -p ActiveState -p SubState "
        "-p ActiveEnterTimestamp -p ExecMainStartTimestamp -p NRestarts -p ExecStart"
    )
    fields: dict[str, str] = {}
    for line in result.stdout.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            fields[key] = value

    flags: dict[str, object] = {}
    match = re.search(r"argv\[\]=(.*?)\s*;", fields.get("ExecStart", ""))
    if match:
        try:
            argv = shlex.split(match.group(1))
        except ValueError:
            argv = []
        index = 0
        while index < len(argv):
            arg = argv[index]
            if not arg.startswith("--"):
                index += 1
                continue
            name = arg[2:]
            if "=" in name:
                key, value = name.split("=", 1)
                index += 1
            elif index + 1 < len(argv) and not argv[index + 1].startswith("--"):
                key, value = name, argv[index + 1]
                index += 2
            else:
                key, value = name, True
                index += 1
            lower = key.lower()
            sensitive = (
                lower in {"key", "api-key", "apikey", "token", "secret", "password"}
                or "api-key" in lower or "apikey" in lower
                or "secret" in lower or "password" in lower
            )
            # 注意：prefix-cache-tokens / draft-tokens / vision-max-tokens 是数量，不是密钥。
            if sensitive:
                value = "***" if value else value
            flags[key] = value

    return {
        "name": SERVICE_NAME,
        "active": fields.get("ActiveState", "unknown"),
        "sub": fields.get("SubState", "unknown"),
        "since": fields.get("ActiveEnterTimestamp", ""),
        "restarts": fields.get("NRestarts", "0"),
        "flags": flags,
    }


def parse_logs(lines: list[str]) -> dict:
    requests: list[dict] = []
    series = {"prefill": [], "decode": [], "running": [], "waiting": []}
    live = {"state": "unknown", "running": 0, "waiting": 0, "prefilling": 0}
    throughput_seen = 0

    for line in reversed(lines):
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        kind = event.get("event")

        if kind == "request_done" and len(requests) < 40:
            req = event.get("request") or {}
            res = event.get("result") or {}
            timings = event.get("timings_seconds") or {}
            spec = event.get("speculative") or {}
            decode_s = float(timings.get("decode") or 0)
            output = int(res.get("completion_tokens") or 0)
            drafted = int(spec.get("drafted_tokens") or 0)
            requests.append(
                {
                    "id": req.get("request_id"),
                    "at": (event.get("timestamp_unix_ms") or 0) / 1000,
                    "protocol": req.get("protocol"),
                    "model": req.get("model"),
                    "prompt_tokens": int(res.get("prompt_tokens") or 0),
                    "reused_tokens": int(res.get("prefix_cache_hit_tokens") or 0),
                    "output_tokens": output,
                    "ttft_ms": round(float(timings.get("ttft") or 0) * 1000, 1),
                    "prefill_ms": round(float(timings.get("prefill") or 0) * 1000, 1),
                    "decode_ms": round(decode_s * 1000, 1),
                    "total_s": round(float(timings.get("total") or 0), 2),
                    "decode_tps": round(output / decode_s, 1) if decode_s > 0 else None,
                    "accept_rate": (
                        round(int(spec.get("accepted_tokens") or 0) / drafted, 3)
                        if drafted
                        else None
                    ),
                    "finish": res.get("finish_reason"),
                    "tools": int(req.get("tool_count") or 0),
                }
            )
        elif kind == "throughput" and throughput_seen < 60:
            throughput_seen += 1
            rates = event.get("throughput_tokens_per_second") or {}
            scheduler = event.get("scheduler") or {}
            series["prefill"].append(round(float(rates.get("prefill") or 0), 1))
            series["decode"].append(round(float(rates.get("decode") or 0), 1))
            series["running"].append(int(scheduler.get("running") or 0))
            series["waiting"].append(int(scheduler.get("waiting") or 0))
            if live["state"] == "unknown":
                running = int(scheduler.get("running") or 0)
                prefilling = int(scheduler.get("prefilling") or 0)
                waiting = int(scheduler.get("waiting") or 0)
                live = {
                    "state": "generating" if running or prefilling else "idle",
                    "running": running,
                    "waiting": waiting,
                    "prefilling": prefilling,
                }

    for key in series:
        series[key].reverse()
    return {"requests": requests, "series": series, "live": live}


class LogIndex:
    """Incrementally indexes the JSONL log for run totals and a 500-request window."""

    def __init__(self) -> None:
        self.size = 0
        self.loaded = False
        self.lock = threading.Lock()
        self.requests: deque[dict] = deque(maxlen=500)
        self.totals: dict[str, float] = {
            "requests": 0, "errors": 0, "prompt_tokens": 0, "reused": 0,
            "output_tokens": 0, "prefill_ms": 0.0, "prompt_ms": 0.0,
            "decode_ms": 0.0, "since": 0.0,
        }
        self.requests_reused = 0

    def _parse(self, line: str) -> None:
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            return
        kind = event.get("event")
        if kind in ("request_error", "request_rejected"):
            self.totals["errors"] += 1
            return
        if kind != "request_done":
            return

        req = event.get("request") or {}
        res = event.get("result") or {}
        timings = event.get("timings_seconds") or {}
        spec = event.get("speculative") or {}
        prompt = int(res.get("prompt_tokens") or 0)
        reused = int(res.get("prefix_cache_hit_tokens") or 0)
        output = int(res.get("completion_tokens") or 0)
        prefill_ms = float(timings.get("prefill") or 0) * 1000
        decode_ms = float(timings.get("decode") or 0) * 1000
        if not self.totals["since"]:
            self.totals["since"] = (event.get("timestamp_unix_ms") or 0) / 1000
        self.totals["requests"] += 1
        self.totals["prompt_tokens"] += prompt
        self.totals["reused"] += reused
        self.totals["output_tokens"] += output
        self.totals["prefill_ms"] += prefill_ms
        self.totals["prompt_ms"] += prefill_ms  # Strata 兼容字段名
        self.totals["decode_ms"] += decode_ms
        if reused > 0:
            self.requests_reused += 1

        drafted = int(spec.get("drafted_tokens") or 0)
        fresh = max(0, prompt - reused)
        self.requests.append(
            {
                "id": req.get("request_id"),
                "at": (event.get("timestamp_unix_ms") or 0) / 1000,
                "time": (event.get("timestamp_unix_ms") or 0) / 1000,  # Strata 兼容字段名
                "protocol": req.get("protocol"),
                "model": req.get("model"),
                "prompt_tokens": prompt,
                "reused_tokens": reused,
                "reused": reused,  # Strata 兼容字段名
                "output_tokens": output,
                "ttft_ms": round(float(timings.get("ttft") or 0) * 1000, 1),
                "prefill_ms": round(prefill_ms, 1),
                "prompt_ms": round(prefill_ms, 1),  # Strata 兼容字段名
                "decode_ms": round(decode_ms, 1),
                "total_s": round(float(timings.get("total") or 0), 2),
                "duration_s": round(float(timings.get("total") or 0), 2),
                "decode_tps": round(output / (decode_ms / 1000), 1) if decode_ms > 0 else None,
                "decode_tok_s": round(output / (decode_ms / 1000), 1) if decode_ms > 0 else None,
                "prefill_tps": round(fresh / (prefill_ms / 1000), 1) if prefill_ms > 0 and fresh else None,
                "prefix_reuse_path": res.get("prefix_reuse_path"),
                # ninfer 的日志目前不记录 MoE 专家缓存命中率，也不走 PCIe expert fetch。
                # 明确给 None，前端显示 “—”，不要伪造数字。
                "hit_rate": None,
                "pcie_share": None,
                "accept_rate": (
                    round(int(spec.get("accepted_tokens") or 0) / drafted, 3) if drafted else None
                ),
                "finish": res.get("finish_reason"),
                "tools": int(req.get("tool_count") or 0),
            }
        )

    def refresh(self) -> None:
        result = remote(f"stat -c %s {shlex.quote(LOG_PATH)}", timeout=5)
        try:
            size = int(result.stdout.strip())
        except ValueError:
            return
        with self.lock:
            if not self.loaded or size < self.size:
                self.requests.clear()
                self.totals.update({k: 0 for k in self.totals})
                self.requests_reused = 0
                offset = 0
            else:
                offset = self.size
            if size == offset and self.loaded:
                return
            # 54 MB 的 JSONL 直传 SSH 要 40+ 秒，gzip 后只要 ~4 秒。
            command = f"tail -c +{offset + 1} {shlex.quote(LOG_PATH)} | gzip -1"
            output = remote_raw(command, timeout=30)
            try:
                raw = gzip.decompress(output.stdout) if output.stdout else b""
            except OSError:
                raw = output.stdout
            lines = raw.decode("utf-8", "replace").splitlines()
            if offset and lines:
                lines = lines[1:]  # first line may be a partial leftover
            for line in lines:
                self._parse(line)
            self.size = size
            self.loaded = True

    def start(self, interval: float = 5.0) -> None:
        """Build the index once at boot, then follow appended log lines."""
        def loop() -> None:
            while True:
                try:
                    self.refresh()
                except Exception:
                    pass
                time.sleep(interval)

        threading.Thread(target=loop, name="request-log-index", daemon=True).start()

    def snapshot(self) -> dict:
        with self.lock:
            requests = list(self.requests)
            totals = dict(self.totals)
        rows = list(reversed(requests))
        window = requests[-500:]
        def bounds(key: str) -> dict:
            values = [r[key] for r in window if r.get(key) and r[key] > 0]
            return {"min": round(min(values), 1), "max": round(max(values), 1), "n": len(values)} if values else {}
        return {
            "requests": rows,
            "totals": totals,
            "requests_reused": sum(1 for r in window if (r.get("reused_tokens") or 0) > 0),
            "requests_kept": len(requests),
            "speed_window": {"decode": bounds("decode_tps"), "prefill": bounds("prefill_tps")},
        }


LOG_INDEX = LogIndex()


def _flag_number(flags: dict, name: str, cast) -> float | int | None:
    value = flags.get(name)
    if value is None or value is True or value == "":
        return None
    try:
        return cast(value)
    except (TypeError, ValueError):
        return None


def build_overview() -> dict:
    with ThreadPoolExecutor(max_workers=5) as pool:
        health_future = pool.submit(http_json, "/health")
        residency_future = pool.submit(http_json, "/ninfer/model")
        machine_future = pool.submit(machine_info)
        service_future = pool.submit(service_info)
        logs_future = pool.submit(tail_log, 500)

        health = health_future.result()
        residency = residency_future.result()
        machine = machine_future.result()
        service = service_future.result()
        parsed = parse_logs(logs_future.result())

    upstream_ok = "error" not in health
    residency_state = residency.get("state") if "error" not in residency else "unknown"
    flags = service.get("flags") or {}

    # 请求索引由后台线程增量维护，接口只取快照，避免首屏等待整个 JSONL。
    log = LOG_INDEX.snapshot()
    requests = log["requests"]
    totals = log["totals"]
    last = requests[0] if requests else None

    live = dict(parsed["live"])
    running = int(live.get("running") or 0)
    prefilling = int(live.get("prefilling") or 0)
    waiting = int(live.get("waiting") or 0)
    if not upstream_ok:
        state = "error"
    elif residency_state not in ("gpu", "gpu_resident", "loaded", "unknown"):
        state = "unloaded"
    elif running or prefilling:
        state = "reading" if prefilling and not running else "generating"
    else:
        state = "idle"
    active = state in ("reading", "generating")
    prompt = int((last or {}).get("prompt_tokens") or 0) if active else 0
    reused = int((last or {}).get("reused_tokens") or 0) if active else 0
    generated = int((last or {}).get("output_tokens") or 0) if state == "generating" else 0
    live.update(
        {
            "state": state,
            "running": running,
            "prefilling": prefilling,
            "queued": waiting,
            "waiting": waiting,
            "prompt_tokens": prompt,
            "prompt_total": prompt,
            "reused": reused,
            "prompt_read": max(0, prompt - reused),
            "generated": generated,
            "tok_s": (last or {}).get("decode_tps") if state == "generating" else None,
            "prefill_tok_s_mean": (last or {}).get("prefill_tps") if state == "reading" else None,
        }
    )

    history = HISTORY.snapshot()
    history["tok_s"] = list((parsed["series"].get("decode") or [])[-90:])
    history["prefill_tok_s_mean"] = list((parsed["series"].get("prefill") or [])[-90:])

    max_context = int(health.get("max_context") or 0)
    last_prompt = int((last or {}).get("prompt_tokens") or 0)
    last_reused = int((last or {}).get("reused_tokens") or 0)
    last_output = int((last or {}).get("output_tokens") or 0)
    if state in ("reading", "generating"):
        cached, reading, answering = last_reused, max(0, last_prompt - last_reused), generated
    else:
        cached, reading, answering = last_prompt + last_output, 0, 0
    context_used = cached + reading + answering
    context_pct = (
        round(min(100.0, context_used / max_context * 100), 1) if max_context else None
    )

    prefix_budget = _flag_number(flags, "prefix-cache-tokens", int)
    archive_mib = _flag_number(flags, "prefix-archive-max-mib", int)
    conversation_cache = {
        "enabled": bool(prefix_budget and prefix_budget > 0),
        "budget_tokens": prefix_budget,
        "archive": flags.get("prefix-archive"),
        "archive_max_mib": archive_mib,
        "requests": totals.get("requests", 0),
        "requests_reused": log["requests_reused"],
        "prompt_tokens": totals.get("prompt_tokens", 0),
        "reused_tokens": totals.get("reused", 0),
        "last_prompt": last_prompt if last else None,
        "last_reused": last_reused if last else None,
        "last_at": (last or {}).get("at"),
        "since": totals.get("since") or None,
        # ninfer 没有 Strata 的 --conversation-cache-mib 停车机制，对应能力是前缀缓存/归档。
        "note": (
            "ninfer 对应能力是前缀缓存"
            + (f"（{int(prefix_budget):,} tokens）" if prefix_budget else "")
            + (" 与磁盘归档" if flags.get("prefix-archive") else "")
            + "；这里展示复用统计，没有会话停车位概念。"
        ),
    }

    gpus = machine.get("gpus") or []
    engine = {
        "model": health.get("model") or (residency or {}).get("model"),
        "max_context": max_context,
        "images": bool(health.get("images")),
        "kv": flags.get("kv-dtype"),
        "spec": flags.get("spec"),
        "draft_tokens": _flag_number(flags, "draft-tokens", int),
        "tp": health.get("tp"),
        "version": None,
    }

    return {
        "ts": time.time(),
        "upstream_ok": upstream_ok,
        "base_url": BASE_URL,
        "health": health,
        "residency": residency,
        "machine": machine,
        "service": service,
        "engine": engine,
        "hardware_static": {
            "gpu_name": gpus[0].get("name") if gpus else None,
            "cores": machine.get("cores"),
            "threads": machine.get("threads"),
        },
        "live": live,
        "series": parsed["series"],
        "history": history,
        "requests": requests,
        "requests_kept": log["requests_kept"],
        "totals": totals,
        "speed_window": log["speed_window"],
        "conversation_cache": conversation_cache,
        "context": {
            "last_prompt_tokens": last_prompt,
            "last_reused_tokens": last_reused,
            "last_output_tokens": last_output,
            "max_tokens": max_context,
            "pct": context_pct,
            "cached": cached,
            "reading": reading,
            "answering": answering,
            "free": max(0, max_context - context_used) if max_context else 0,
        },
    }


def overview() -> dict:
    now = time.monotonic()
    with _CACHE_LOCK:
        cached_at = float(_CACHE["at"] or 0)
        cached = _CACHE["data"]
    if cached is not None and now - cached_at < OVERVIEW_TTL:
        return cached  # type: ignore[return-value]
    data = build_overview()
    with _CACHE_LOCK:
        _CACHE.update(at=now, data=data)
    return data


# --------------------------------------------------------------------------- #
# HTTP surface
# --------------------------------------------------------------------------- #
class Handler(BaseHTTPRequestHandler):
    server_version = "NinferAdmin/0.1"

    def log_message(self, fmt: str, *args: object) -> None:
        sys.stderr.write("[admin-ui] " + (fmt % args) + "\n")

    def send_json(self, payload: object, status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def send_static(self, filename: str) -> None:
        path = STATIC_DIR / filename
        if not path.is_file():
            self.send_error(404)
            return
        content = path.read_bytes()
        content_type = {
            ".html": "text/html; charset=utf-8",
            ".css": "text/css; charset=utf-8",
            ".js": "text/javascript; charset=utf-8",
            ".svg": "image/svg+xml",
        }.get(path.suffix, "application/octet-stream")
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(content)

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            return self.send_static("index.html")
        if path in ("/app.css", "/app.js"):
            return self.send_static(path.lstrip("/"))
        if path == "/api/overview":
            return self.send_json(overview())
        if path == "/api/requests":
            return self.send_json({"requests": overview()["requests"]})
        if path == "/api/log":
            lines = tail_log(160)
            return self.send_json({"lines": lines[-80:]})
        self.send_error(404)

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path == "/api/chat":
            return self.proxy_chat()
        if path in ("/api/model/load", "/api/model/offload"):
            if ADMIN_TOKEN and self.headers.get("X-Admin-Token") != ADMIN_TOKEN:
                return self.send_json({"error": "invalid admin token"}, 401)
            target = "/ninfer/model/load" if path.endswith("load") and "offload" not in path else "/ninfer/model/offload"
            return self.proxy_simple(target)
        self.send_error(404)

    def body_json(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b"{}"
        return json.loads(raw.decode("utf-8"))

    def proxy_simple(self, target: str) -> None:
        parsed = urlparse(BASE_URL)
        connection = http.client.HTTPConnection(parsed.hostname or "127.0.0.1", parsed.port or 80, timeout=30)
        headers = {"Content-Type": "application/json", "Connection": "close"}
        if ADMIN_TOKEN:
            headers["X-Admin-Token"] = ADMIN_TOKEN or ""
        try:
            connection.request("POST", target, body=b"{}", headers=headers)
            response = connection.getresponse()
            body = response.read().decode("utf-8", "replace")
            self.send_response(response.status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body.encode("utf-8"))))
            self.end_headers()
            self.wfile.write(body.encode("utf-8"))
        except Exception as exc:
            self.send_json({"error": str(exc)}, 502)
        finally:
            connection.close()

    def proxy_chat(self) -> None:
        try:
            payload = self.body_json()
        except (ValueError, UnicodeDecodeError) as exc:
            return self.send_json({"error": f"invalid request body: {exc}"}, 400)

        if not payload.get("model"):
            health = http_json("/health")
            if health.get("model"):
                payload["model"] = health["model"]

        parsed = urlparse(BASE_URL)
        api_key = self.headers.get("X-Ninfer-Key") or ""
        headers = {"Content-Type": "application/json", "Connection": "close"}
        if api_key:
            headers["Authorization"] = api_key if api_key.startswith("Bearer ") else "Bearer " + api_key

        connection = http.client.HTTPConnection(
            parsed.hostname or "127.0.0.1", parsed.port or 80, timeout=300
        )
        try:
            connection.request(
                "POST", "/v1/chat/completions", body=json.dumps(payload).encode("utf-8"), headers=headers
            )
            upstream = connection.getresponse()
            if upstream.status != 200:
                body = upstream.read()
                return self.send_json(
                    {"error": body.decode("utf-8", "replace")}, upstream.status
                )

            self.send_response(200)
            self.send_header("Content-Type", upstream.getheader("Content-Type", "text/event-stream"))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Accel-Buffering", "no")
            self.end_headers()
            try:
                while True:
                    chunk = upstream.read(2048)
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass  # the user navigated away; the upstream connection is closed below
        except Exception as exc:
            self.send_json({"error": str(exc)}, 502)
        finally:
            connection.close()


def main() -> int:
    global BASE_URL, TARGET_HOST, LOG_PATH
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default=os.environ.get("NINFER_ADMIN_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("NINFER_ADMIN_PORT", "8081")))
    parser.add_argument("--base-url", default=BASE_URL)
    parser.add_argument("--target-host", default=TARGET_HOST)
    parser.add_argument("--log-path", default=LOG_PATH)
    args = parser.parse_args()

    BASE_URL = args.base_url.rstrip("/")
    TARGET_HOST = args.target_host
    LOG_PATH = args.log_path

    HISTORY.start()
    LOG_INDEX.start()

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.daemon_threads = True
    print(f"[admin-ui] console on http://{args.host}:{args.port} -> {BASE_URL}", file=sys.stderr)
    print(f"[admin-ui] telemetry host: {TARGET_HOST}; log: {LOG_PATH}", file=sys.stderr)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
