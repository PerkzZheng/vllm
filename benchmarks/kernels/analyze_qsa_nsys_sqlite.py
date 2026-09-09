#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Summarize request-sized CUDA bursts in an Nsight Systems SQLite export."""

from __future__ import annotations

import argparse
import sqlite3
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import regex as re


@dataclass
class Kernel:
    device: int
    global_pid: int
    start: int
    end: int
    name: str


@dataclass
class NvtxRange:
    start: int
    end: int
    global_pid: int
    name: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sqlite", type=Path)
    parser.add_argument(
        "--gap-ms",
        type=float,
        default=20.0,
        help="Start a new burst after this much GPU idle time (default: 20).",
    )
    parser.add_argument(
        "--last",
        type=int,
        default=4,
        help="Show this many final bursts per device (default: 4).",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=12,
        help="Show this many kernels by aggregate duration per burst (default: 12).",
    )
    parser.add_argument(
        "--nvtx-pattern",
        help="Summarize matching NVTX ranges instead of idle-delimited bursts.",
    )
    parser.add_argument(
        "--nvtx-last",
        type=int,
        default=0,
        help="Keep only the final N matching NVTX ranges per process (default: all).",
    )
    parser.add_argument(
        "--aggregate-nvtx",
        action="store_true",
        help="Also aggregate all selected ranges for each process.",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Suppress individual ranges when --aggregate-nvtx is used.",
    )
    parser.add_argument(
        "--nvtx-name-summary",
        action="store_true",
        help="Group matching NVTX CPU spans by process and range name.",
    )
    return parser.parse_args()


def load_kernels(path: Path) -> dict[int, list[Kernel]]:
    query = """
        SELECT k.deviceId, k.globalPid, k.start, k.end, s.value
        FROM CUPTI_ACTIVITY_KIND_KERNEL AS k
        JOIN StringIds AS s ON s.id = k.demangledName
        ORDER BY k.deviceId, k.start
    """
    by_device: dict[int, list[Kernel]] = defaultdict(list)
    with sqlite3.connect(path) as connection:
        for device, global_pid, start, end, name in connection.execute(query):
            by_device[device].append(Kernel(device, global_pid, start, end, name))
    return by_device


def load_nvtx_ranges(path: Path, pattern: str) -> list[NvtxRange]:
    query = """
        SELECT n.start, n.end, n.globalTid, COALESCE(n.text, s.value)
        FROM NVTX_EVENTS AS n
        LEFT JOIN StringIds AS s ON s.id = n.textId
        WHERE n.end IS NOT NULL AND n.globalTid IS NOT NULL
        ORDER BY n.start
    """
    regex = re.compile(pattern)
    ranges: list[NvtxRange] = []
    with sqlite3.connect(path) as connection:
        for start, end, global_tid, name in connection.execute(query):
            if name is not None and regex.search(name):
                global_pid = global_tid & ~((1 << 24) - 1)
                ranges.append(NvtxRange(start, end, global_pid, name))
    return ranges


def split_bursts(kernels: list[Kernel], gap_ns: int) -> list[list[Kernel]]:
    bursts: list[list[Kernel]] = []
    for kernel in kernels:
        if not bursts or kernel.start - bursts[-1][-1].end > gap_ns:
            bursts.append([kernel])
        else:
            bursts[-1].append(kernel)
    return bursts


def summarize_burst(index: int, burst: list[Kernel], top: int) -> None:
    start = burst[0].start
    end = max(kernel.end for kernel in burst)
    duration_by_name: dict[str, int] = defaultdict(int)
    count_by_name: dict[str, int] = defaultdict(int)
    for kernel in burst:
        duration_by_name[kernel.name] += kernel.end - kernel.start
        count_by_name[kernel.name] += 1

    print(
        f"  burst {index}: start={start / 1e6:.3f} ms "
        f"span={(end - start) / 1e6:.3f} ms "
        f"kernel_sum={sum(duration_by_name.values()) / 1e6:.3f} ms "
        f"kernels={len(burst)}"
    )
    ranked = sorted(duration_by_name, key=duration_by_name.get, reverse=True)
    for name in ranked[:top]:
        print(
            f"    {duration_by_name[name] / 1e6:9.3f} ms "
            f"{count_by_name[name]:4d}  {name}"
        )


def summarize_nvtx_ranges(
    kernels_by_device: dict[int, list[Kernel]],
    ranges: list[NvtxRange],
    last: int,
    top: int,
    aggregate: bool,
    summary_only: bool,
) -> None:
    kernels_by_process: dict[int, list[Kernel]] = defaultdict(list)
    for kernels in kernels_by_device.values():
        for kernel in kernels:
            kernels_by_process[kernel.global_pid].append(kernel)

    ranges_by_process: dict[int, list[NvtxRange]] = defaultdict(list)
    for nvtx_range in ranges:
        ranges_by_process[nvtx_range.global_pid].append(nvtx_range)

    for global_pid, process_ranges in ranges_by_process.items():
        if last > 0:
            process_ranges = process_ranges[-last:]
        print(f"process {global_pid}: {len(process_ranges)} matching NVTX ranges")
        process_kernels = kernels_by_process.get(global_pid, [])
        selected_ranges: list[tuple[NvtxRange, list[Kernel]]] = []
        for index, nvtx_range in enumerate(process_ranges):
            selected = [
                kernel
                for kernel in process_kernels
                if kernel.start >= nvtx_range.start and kernel.end <= nvtx_range.end
            ]
            selected_ranges.append((nvtx_range, selected))
            if aggregate and summary_only:
                continue
            if selected:
                devices = sorted({kernel.device for kernel in selected})
                gpu_start = min(kernel.start for kernel in selected)
                gpu_end = max(kernel.end for kernel in selected)
                device_text = ",".join(str(device) for device in devices)
                print(
                    f"  range {index}: devices={device_text} "
                    f"cpu_span={(nvtx_range.end - nvtx_range.start) / 1e6:.3f} ms "
                    f"gpu_span={(gpu_end - gpu_start) / 1e6:.3f} ms "
                    f"{nvtx_range.name}"
                )
                summarize_burst(index, selected, top)
            else:
                print(
                    f"  range {index}: no kernels, "
                    f"cpu_span={(nvtx_range.end - nvtx_range.start) / 1e6:.3f} ms "
                    f"{nvtx_range.name}"
                )

        if aggregate:
            selected_kernels = [
                kernel for _, kernels in selected_ranges for kernel in kernels
            ]
            nonempty = [
                (nvtx_range, kernels)
                for nvtx_range, kernels in selected_ranges
                if kernels
            ]
            if nonempty:
                gpu_spans = [
                    max(kernel.end for kernel in kernels)
                    - min(kernel.start for kernel in kernels)
                    for _, kernels in nonempty
                ]
                first_gpu = min(
                    kernel.start for _, kernels in nonempty for kernel in kernels
                )
                last_gpu = max(
                    kernel.end for _, kernels in nonempty for kernel in kernels
                )
                cpu_spans = [
                    nvtx_range.end - nvtx_range.start for nvtx_range, _ in nonempty
                ]
                print(
                    f"  aggregate: ranges={len(nonempty)} "
                    f"cpu_sum={sum(cpu_spans) / 1e6:.3f} ms "
                    f"gpu_sum={sum(gpu_spans) / 1e6:.3f} ms "
                    f"gpu_wall={(last_gpu - first_gpu) / 1e6:.3f} ms "
                    f"gpu_avg={sum(gpu_spans) / len(gpu_spans) / 1e6:.3f} ms"
                )
                summarize_burst(-1, selected_kernels, top)


def summarize_nvtx_names(ranges: list[NvtxRange]) -> None:
    grouped: dict[tuple[int, str], list[int]] = defaultdict(list)
    for nvtx_range in ranges:
        grouped[(nvtx_range.global_pid, nvtx_range.name)].append(
            nvtx_range.end - nvtx_range.start
        )
    current_process: int | None = None
    for (global_pid, name), spans in sorted(grouped.items()):
        if global_pid != current_process:
            print(f"process {global_pid}: NVTX CPU spans by name")
            current_process = global_pid
        total = sum(spans)
        print(
            f"  {total / 1e6:9.3f} ms total "
            f"{total / len(spans) / 1e6:8.3f} ms avg "
            f"{len(spans):4d}  {name}"
        )


def main() -> None:
    args = parse_args()
    gap_ns = round(args.gap_ms * 1e6)
    kernels_by_device = load_kernels(args.sqlite)
    if args.nvtx_pattern:
        ranges = load_nvtx_ranges(args.sqlite, args.nvtx_pattern)
        if args.nvtx_name_summary:
            summarize_nvtx_names(ranges)
        summarize_nvtx_ranges(
            kernels_by_device,
            ranges,
            args.nvtx_last,
            args.top,
            args.aggregate_nvtx,
            args.summary_only,
        )
        return

    for device, kernels in kernels_by_device.items():
        bursts = split_bursts(kernels, gap_ns)
        print(f"device {device}: {len(kernels)} kernels, {len(bursts)} bursts")
        first = max(0, len(bursts) - args.last)
        for index in range(first, len(bursts)):
            summarize_burst(index, bursts[index], args.top)


if __name__ == "__main__":
    main()
