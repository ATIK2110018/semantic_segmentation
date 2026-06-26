"""
Ablation Study Runner
=====================
Runs 3 incremental model configurations and compiles a comparison summary.

Configurations:
  1. Baseline              — 4-Bands (RGB+NIR), Dice+Focal Loss, Cosine LR
  2. Baseline + NDVI/NDWI  — 6-Bands (RGB+NIR+NDVI+NDWI), Dice+Focal Loss, Cosine LR
  3. Proposed (Full Model) — 6-Bands (RGB+NIR+NDVI+NDWI), Boundary-Weighted Loss, Cosine LR

Usage:
    python run_ablation.py [--data_path dataset] [--epochs 200] [--batch_size 16] ...

All arguments accepted by main.py are forwarded to each run (except the
flags that the ablation controls: --boundary_multiplier, --output_dir,
--model_save_path, --no_ndvi, --lr_schedule).
"""

import argparse
import csv
import os
import subprocess
import sys
import time
from pathlib import Path

# ── Ablation configurations ─────────────────────────────────────────────────
ABLATION_CONFIGS = [
    {
        "name": "1_Baseline",
        "label": "Version 1 (Baseline)",
        "flags": [
            "--boundary_multiplier", "0.0",
            "--no_ndvi",
        ],
    },
    {
        "name": "2_Baseline_NDVI",
        "label": "Version 2 (Baseline + NDVI/NDWI)",
        "flags": [
            "--boundary_multiplier", "0.0",
        ],
    },
    {
        "name": "3_Proposed_0.5",
        "label": "Version 3 (Proposed - Mult 0.5)",
        "flags": [
            "--boundary_multiplier", "0.5",
        ],
    },
    {
        "name": "4_Proposed_1.0",
        "label": "Version 4 (Proposed - Mult 1.0)",
        "flags": [
            "--boundary_multiplier", "1.0",
        ],
    },
    {
        "name": "5_Proposed_2.0",
        "label": "Version 5 (Proposed - Mult 2.0)",
        "flags": [
            "--boundary_multiplier", "2.0",
        ],
    },
    {
        "name": "6_Proposed_DualHead",
        "label": "Version 6 (Dual Head Network)",
        "flags": [
            "--boundary_multiplier", "0.0", # Let the second head handle boundaries entirely
            "--dual_head",
        ],
    },
]


def parse_evaluation_csv(csv_path):
    """Parse evaluation_results.csv and return per-class and summary metrics."""
    per_class = {}
    summary = {}
    section = "per_class"

    with open(csv_path, "r") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        for row in reader:
            if not row or all(c.strip() == "" for c in row):
                section = "summary"
                continue
            if section == "per_class" and len(row) >= 8:
                name = row[0]
                if name == "Summary Metric":
                    section = "summary"
                    continue
                per_class[name] = {
                    "iou": float(row[2]),
                    "precision": float(row[3]),
                    "recall": float(row[4]),
                    "f1": float(row[5]),
                    "accuracy": float(row[6]),
                    "support": int(row[7]),
                }
            elif section == "summary" and len(row) >= 2:
                try:
                    summary[row[0]] = float(row[1])
                except ValueError:
                    pass

    return per_class, summary


def parse_boundary_csv(csv_path):
    """Parse boundary_results_global.csv and return metrics dict."""
    metrics = {}
    if not os.path.exists(csv_path):
        return metrics
    with open(csv_path, "r") as f:
        reader = csv.reader(f)
        next(reader, None)  # skip header
        for row in reader:
            if len(row) >= 2:
                try:
                    metrics[row[0]] = float(row[1])
                except ValueError:
                    pass
    return metrics


def format_duration(seconds):
    """Format seconds into human-readable string."""
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h}h {m}m {s}s"
    return f"{m}m {s}s"


def run_ablation(common_args, base_output_dir="ablation_results"):
    """Run all ablation configurations sequentially."""
    os.makedirs(base_output_dir, exist_ok=True)

    results = []
    total_start = time.time()

    print("\n" + "=" * 80)
    print("  ABLATION STUDY")
    print("=" * 80)
    print(f"  Configurations : {len(ABLATION_CONFIGS)}")
    print(f"  Output directory: {base_output_dir}/")
    print(f"  Common args     : {' '.join(common_args)}")
    print("=" * 80 + "\n")

    for i, config in enumerate(ABLATION_CONFIGS):
        run_name = config["name"]
        run_label = config["label"]
        run_output = os.path.join(base_output_dir, run_name)

        print("\n" + "━" * 80)
        print(f"  RUN {i + 1}/{len(ABLATION_CONFIGS)}: {run_label}")
        print(f"  Output: {run_output}/")
        print("━" * 80 + "\n")

        cmd = [
            sys.executable, "main.py",
            *common_args,
            "--output_dir", run_output,
            "--no_save_model",
            *config["flags"],
        ]

        print(f"  Command: {' '.join(cmd)}\n")
        run_start = time.time()

        try:
            subprocess.run(cmd, check=True)
            run_duration = time.time() - run_start

            # Parse results
            eval_csv = os.path.join(run_output, "evaluation_results.csv")
            boundary_csv = os.path.join(run_output, "boundary_results_global.csv")

            per_class, summary = {}, {}
            boundary = {}

            if os.path.exists(eval_csv):
                per_class, summary = parse_evaluation_csv(eval_csv)
            if os.path.exists(boundary_csv):
                boundary = parse_boundary_csv(boundary_csv)

            results.append({
                "name": run_name,
                "label": run_label,
                "status": "SUCCESS",
                "duration": run_duration,
                "summary": summary,
                "boundary": boundary,
                "per_class": per_class,
            })

            print(f"\n  ✓ Run completed in {format_duration(run_duration)}")

        except subprocess.CalledProcessError as e:
            run_duration = time.time() - run_start
            results.append({
                "name": run_name,
                "label": run_label,
                "status": f"FAILED (exit code {e.returncode})",
                "duration": run_duration,
                "summary": {},
                "boundary": {},
                "per_class": {},
            })
            print(f"\n  ✗ Run FAILED after {format_duration(run_duration)}")

    total_duration = time.time() - total_start

    # ── Print comparison table ───────────────────────────────────────────
    print("\n\n" + "=" * 110)
    print("  ABLATION STUDY — COMPARISON TABLE")
    print("=" * 110)

    header = (
        f"{'Configuration':<42} {'OA':>7} {'mIoU':>7} {'wIoU':>7} "
        f"{'mF1':>7} {'BF':>7} {'B-IoU':>7} {'Time':>10} {'Status':>8}"
    )
    print(header)
    print("-" * 110)

    for r in results:
        s = r["summary"]
        b = r["boundary"]
        oa = f"{s['overall_accuracy']:.4f}" if "overall_accuracy" in s else "  —"
        miou = f"{s['mean_iou_all']:.4f}" if "mean_iou_all" in s else "  —"
        wiou = f"{s['weighted_iou']:.4f}" if "weighted_iou" in s else "  —"
        mf1 = f"{s['mean_f1']:.4f}" if "mean_f1" in s else "  —"
        bf = f"{b['bf_score']:.4f}" if "bf_score" in b else "  —"
        biou = f"{b['boundary_iou']:.4f}" if "boundary_iou" in b else "  —"
        dur = format_duration(r["duration"])
        status = "✓" if r["status"] == "SUCCESS" else "✗"

        print(
            f"  {r['label']:<40} {oa:>7} {miou:>7} {wiou:>7} "
            f"{mf1:>7} {bf:>7} {biou:>7} {dur:>10} {status:>8}"
        )

    print("-" * 110)
    print(f"  Total time: {format_duration(total_duration)}")
    print("=" * 110)

    # ── Per-class IoU comparison table ───────────────────────────────────
    # Collect all class names from the first successful run
    all_classes = []
    for r in results:
        if r["per_class"]:
            all_classes = list(r["per_class"].keys())
            break

    if all_classes:
        print("\n\n" + "=" * 100)
        print("  PER-CLASS IoU COMPARISON")
        print("=" * 100)

        class_header = f"{'Class':<22}"
        for r in results:
            short = r["name"].split("_", 1)[1].replace("_", " ")[:16]
            class_header += f" {short:>16}"
        print(class_header)
        print("-" * 100)

        for cls in all_classes:
            row = f"  {cls:<20}"
            for r in results:
                if cls in r.get("per_class", {}):
                    row += f" {r['per_class'][cls]['iou']:>16.4f}"
                else:
                    row += f" {'—':>16}"
            print(row)
        print("=" * 100)

    # ── Save comparison CSV ──────────────────────────────────────────────
    comparison_csv = os.path.join(base_output_dir, "ablation_comparison.csv")
    with open(comparison_csv, "w", newline="") as f:
        writer = csv.writer(f)

        # Summary metrics table
        writer.writerow([
            "Configuration", "Overall_Accuracy", "Mean_IoU", "Weighted_IoU",
            "Mean_F1", "BF_Score", "Boundary_IoU", "Duration_s", "Status",
        ])
        for r in results:
            s = r["summary"]
            b = r["boundary"]
            writer.writerow([
                r["label"],
                f"{s.get('overall_accuracy', 0):.6f}",
                f"{s.get('mean_iou_all', 0):.6f}",
                f"{s.get('weighted_iou', 0):.6f}",
                f"{s.get('mean_f1', 0):.6f}",
                f"{b.get('bf_score', 0):.6f}",
                f"{b.get('boundary_iou', 0):.6f}",
                f"{r['duration']:.1f}",
                r["status"],
            ])

        writer.writerow([])
        writer.writerow([])

        # Per-class IoU table
        if all_classes:
            writer.writerow(["Per-Class IoU"] + [r["label"] for r in results])
            for cls in all_classes:
                row = [cls]
                for r in results:
                    if cls in r.get("per_class", {}):
                        row.append(f"{r['per_class'][cls]['iou']:.6f}")
                    else:
                        row.append("")
                writer.writerow(row)

    print(f"\n  Comparison saved to: {comparison_csv}")
    print(f"  Individual results in: {base_output_dir}/*/")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ablation Study Runner — runs 3 incremental configurations",
        epilog=(
            "All other arguments are forwarded to main.py for each run.\n\n"
            "Example:\n"
            "  python run_ablation.py --data_path dataset --epochs 200 --batch_size 16"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--ablation_output",
        type=str,
        default="ablation_results",
        help="Base directory for all ablation outputs (default: ablation_results)",
    )

    # Parse only the known arg; everything else goes to main.py
    known, remaining = parser.parse_known_args()

    # Remove flags that the ablation controls (in case user passes them)
    controlled = {
        "--disable_attention", "--disable_residual",
        "--boundary_multiplier", "--output_dir", "--model_save_path",
        "--no_ndvi", "--lr_schedule",
    }
    cleaned = []
    skip_next = False
    for arg in remaining:
        if skip_next:
            skip_next = False
            continue
        if arg in controlled:
            # If it's a flag that takes a value, skip the next arg too
            if arg in ("--boundary_multiplier", "--output_dir", "--model_save_path", "--lr_schedule"):
                skip_next = True
            continue
        cleaned.append(arg)

    run_ablation(cleaned, base_output_dir=known.ablation_output)
