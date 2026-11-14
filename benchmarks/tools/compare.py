#!/usr/bin/env python3
"""
MobileLLM Performance Regression Comparator
Compares two benchmark JSON results and flags performance regressions.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple


def load_json(path: str) -> Dict[str, Any]:
    p = Path(path)
    if not p.is_file():
        print(f"Error: File not found: {path}", file=sys.stderr)
        sys.exit(1)
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def format_delta(delta_pct: float, higher_is_better: bool) -> Tuple[str, str]:
    is_improved = (delta_pct > 0) if higher_is_better else (delta_pct < 0)
    is_degraded = (delta_pct < 0) if higher_is_better else (delta_pct > 0)

    sign = "+" if delta_pct > 0 else ""
    str_val = f"{sign}{delta_pct:.2f}%"

    if abs(delta_pct) < 3.0:
        return str_val, "NEUTRAL"
    elif is_improved:
        return str_val, "IMPROVED"
    else:
        return str_val, "REGRESSION"


def main() -> None:
    parser = argparse.ArgumentParser(description="MobileLLM Benchmark Regression Comparator")
    parser.add_argument("baseline", help="Path to baseline benchmark JSON")
    parser.add_argument("candidate", help="Path to candidate benchmark JSON")
    parser.add_argument(
        "--threshold",
        type=float,
        default=5.0,
        help="Tolerance threshold percentage before flagging regression (default: 5.0%%)",
    )
    parser.add_argument(
        "--fail-on-regression",
        action="store_true",
        help="Exit with non-zero code if any regression exceeds threshold",
    )
    args = parser.parse_args()

    b_data = load_json(args.baseline)
    c_data = load_json(args.candidate)

    b_sum = b_data.get("summary", {})
    c_sum = c_data.get("summary", {})
    b_load = b_data.get("model_loading", {})
    c_load = c_data.get("model_loading", {})

    metrics_to_compare = [
        ("Time to First Token (TTFT)", b_sum.get("ttft_mean_ms", 0.0), c_sum.get("ttft_mean_ms", 0.0), "ms", False),
        ("Prefill Throughput", b_sum.get("prefill_tps_mean", 0.0), c_sum.get("prefill_tps_mean", 0.0), "tok/s", True),
        ("Decode Throughput", b_sum.get("decode_tps_mean", 0.0), c_sum.get("decode_tps_mean", 0.0), "tok/s", True),
        ("Median (P50) Latency", b_sum.get("p50_latency_ms", 0.0), c_sum.get("p50_latency_ms", 0.0), "ms", False),
        ("P95 Tail Latency", b_sum.get("p95_latency_ms", 0.0), c_sum.get("p95_latency_ms", 0.0), "ms", False),
        ("Peak Process RSS", b_sum.get("peak_rss_mb", 0.0), c_sum.get("peak_rss_mb", 0.0), "MB", False),
        ("Model Load Duration", b_load.get("duration_ms", 0.0), c_load.get("duration_ms", 0.0), "ms", False),
    ]

    print("=" * 88)
    print("                 MobileLLM Performance Regression Comparison")
    print("=" * 88)
    print(f"Baseline:  {args.baseline} (Arch: {b_data.get('arch')}, Threads: {b_data.get('configuration', {}).get('threads')})")
    print(f"Candidate: {args.candidate} (Arch: {c_data.get('arch')}, Threads: {c_data.get('configuration', {}).get('threads')})")
    print(f"Tolerance Threshold: ±{args.threshold:.1f}%")
    print("-" * 88)
    print(f"{'Metric':<28} | {'Baseline':<12} | {'Candidate':<12} | {'Change (%)':<12} | {'Status':<10}")
    print("-" * 88)

    has_regression = False

    for name, b_val, c_val, unit, higher_is_better in metrics_to_compare:
        if b_val <= 0.0:
            pct_change = 0.0
            delta_str, status = "N/A", "NEUTRAL"
        else:
            pct_change = ((c_val - b_val) / b_val) * 100.0
            delta_str, status = format_delta(pct_change, higher_is_better)

        is_severe_regression = False
        if higher_is_better and pct_change < -args.threshold:
            is_severe_regression = True
            status = "FAILED"
        elif not higher_is_better and pct_change > args.threshold:
            is_severe_regression = True
            status = "FAILED"

        if is_severe_regression:
            has_regression = True

        b_disp = f"{b_val:.2f} {unit}"
        c_disp = f"{c_val:.2f} {unit}"
        print(f"{name:<28} | {b_disp:<12} | {c_disp:<12} | {delta_str:<12} | {status:<10}")

    print("=" * 88)

    if has_regression:
        print(f"\n[ALERT] Performance regression detected exceeding ±{args.threshold:.1f}% threshold!")
        if args.fail_on_regression:
            sys.exit(1)
    else:
        print("\n[OK] All performance metrics within acceptable threshold.")


if __name__ == "__main__":
    main()
