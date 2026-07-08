#!/usr/bin/env python3
"""Aggregate classification results across multiple runs for a given partitioning type.

Reads all run directories under results/<type>/ and computes mean, std, min, max
for all metrics (per-class and aggregate), for both multiclass and binary reports.

Usage:
    python3 utils/aggregate_results.py --type default
    python3 utils/aggregate_results.py --type ensemble
    python3 utils/aggregate_results.py --type default --results-base ./results
"""

import argparse
import math
import os
import sys
import re
from ast import literal_eval
from collections import defaultdict


def parse_result_file(filepath):
    """Parse a results .txt file and extract the classification report dicts."""
    with open(filepath, 'r') as f:
        content = f.read()

    # Split multiclass and binary sections
    sections = content.split("--- Binary (normal vs attack) ---")

    multiclass_dict = None
    binary_dict = None
    collision_count = None

    # Parse collision count
    match = re.search(r'Collision count:\s*(\d+)', sections[0])
    if match:
        collision_count = int(match.group(1))

    # Parse multiclass dict (between collision count line and score info line)
    mc_match = re.search(r'(\{.*?\})\s*\nThe score information:', sections[0], re.DOTALL)
    if mc_match:
        try:
            multiclass_dict = literal_eval(mc_match.group(1))
        except (SyntaxError, ValueError):
            pass

    # Parse binary dict
    if len(sections) > 1:
        bin_match = re.search(r'(\{.*?\})\s*\nThe score information:', sections[1], re.DOTALL)
        if bin_match:
            try:
                binary_dict = literal_eval(bin_match.group(1))
            except (SyntaxError, ValueError):
                pass

    return multiclass_dict, binary_dict, collision_count


def collect_metrics(report_dict):
    """Extract all numeric metrics from a classification report dict into a flat structure."""
    metrics = {}
    for key, value in report_dict.items():
        if isinstance(value, dict):
            for metric_name, metric_value in value.items():
                if isinstance(metric_value, (int, float)):
                    metrics[f"{key}/{metric_name}"] = metric_value
        elif isinstance(value, (int, float)):
            metrics[key] = value
    return metrics


def _mean(values):
    return sum(values) / len(values)


def _std(values):
    if len(values) < 2:
        return 0.0
    m = _mean(values)
    return math.sqrt(sum((x - m) ** 2 for x in values) / (len(values) - 1))


def aggregate_metrics(all_metrics_list):
    """Compute mean, std, min, max across a list of metric dicts."""
    all_keys = set()
    for m in all_metrics_list:
        all_keys.update(m.keys())

    results = {}
    for key in sorted(all_keys):
        values = [m[key] for m in all_metrics_list if key in m]
        if len(values) > 0:
            results[key] = {
                'mean': _mean(values),
                'std': _std(values),
                'min': min(values),
                'max': max(values),
                'n': len(values),
            }
    return results


def format_table(aggregated, title):
    """Format aggregated metrics into a readable table."""
    lines = []
    lines.append(f"\n{'='*20} {title} {'='*20}")
    lines.append(f"{'Metric':<40} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8} {'N':>4}")
    lines.append("-" * 72)

    # Group by class/category
    grouped = defaultdict(list)
    for key, stats in aggregated.items():
        parts = key.split('/')
        if len(parts) == 2:
            grouped[parts[0]].append((parts[1], stats))
        else:
            grouped['_global'].append((key, stats))

    for group in sorted(grouped.keys()):
        if group != '_global':
            lines.append(f"\n  {group}:")
        for metric_name, stats in sorted(grouped[group]):
            label = f"    {metric_name}" if group != '_global' else f"  {metric_name}"
            lines.append(
                f"{label:<40} {stats['mean']:>8.4f} {stats['std']:>8.4f} "
                f"{stats['min']:>8.4f} {stats['max']:>8.4f} {stats['n']:>4d}"
            )

    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description="Aggregate results across runs")
    parser.add_argument("--type", required=True, choices=["default", "ensemble"],
                        help="Partitioning type (subdirectory under results/)")
    parser.add_argument("--results-base", default="./results",
                        help="Base results directory (default: ./results)")
    args = parser.parse_args()

    results_dir = os.path.join(args.results_base, args.type)
    if not os.path.isdir(results_dir):
        print(f"[ERROR] Directory not found: {results_dir}")
        sys.exit(1)

    # Find all run directories
    run_dirs = sorted([
        d for d in os.listdir(results_dir)
        if os.path.isdir(os.path.join(results_dir, d)) and d.startswith("run_")
    ])

    if not run_dirs:
        print(f"[ERROR] No run directories found in {results_dir}")
        sys.exit(1)

    print(f"Found {len(run_dirs)} runs in {results_dir}")

    multiclass_metrics_list = []
    binary_metrics_list = []
    collision_counts = []

    for run_dir in run_dirs:
        run_path = os.path.join(results_dir, run_dir)
        # Find the results .txt file (pattern: results_*.txt)
        txt_files = [f for f in os.listdir(run_path) if f.startswith("results_") and f.endswith(".txt")]
        if not txt_files:
            print(f"  [WARN] No results file in {run_dir}, skipping")
            continue

        result_file = os.path.join(run_path, txt_files[0])
        mc_dict, bin_dict, collisions = parse_result_file(result_file)

        if mc_dict:
            multiclass_metrics_list.append(collect_metrics(mc_dict))
        if bin_dict:
            binary_metrics_list.append(collect_metrics(bin_dict))
        if collisions is not None:
            collision_counts.append(collisions)

        print(f"  Parsed {run_dir}")

    # Aggregate
    output_lines = []
    output_lines.append(f"Aggregated Results for '{args.type}' partitioning")
    output_lines.append(f"Number of runs: {len(run_dirs)}")
    output_lines.append(f"Runs parsed successfully: {len(multiclass_metrics_list)} (multiclass), "
                        f"{len(binary_metrics_list)} (binary)")

    if collision_counts:
        output_lines.append(f"\nCollision counts: mean={_mean(collision_counts):.0f}, "
                            f"std={_std(collision_counts):.0f}, "
                            f"min={min(collision_counts)}, max={max(collision_counts)}")

    if multiclass_metrics_list:
        mc_agg = aggregate_metrics(multiclass_metrics_list)
        output_lines.append(format_table(mc_agg, "MULTICLASS"))

    if binary_metrics_list:
        bin_agg = aggregate_metrics(binary_metrics_list)
        output_lines.append(format_table(bin_agg, "BINARY (normal vs attack)"))

    output_text = '\n'.join(output_lines)
    print(output_text)

    # Save to file
    outfile = os.path.join(results_dir, "aggregated_stats.txt")
    with open(outfile, 'w') as f:
        f.write(output_text + '\n')
    print(f"\nSaved to {outfile}")


if __name__ == "__main__":
    main()
