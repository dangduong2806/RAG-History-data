"""
Compute inter-annotator agreement (IAA) for free-form QA answers.
"""
import argparse
import csv
import json
import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

try:
    from evaluation.IAA_metrics import (
        aggregate_mean,
        canonicalize_answer,
        compute_exact,
        compute_f1,
    )
except ImportError:
    from evaluation.IAA_metrics import aggregate_mean, canonicalize_answer, compute_exact, compute_f1


DEFAULT_SUBSET = "data/iaa/subset.csv"
DEFAULT_ANNOTATOR_A = "data/iaa/annotator_a.csv"
DEFAULT_ANNOTATOR_B = "data/iaa/annotator_b.csv"
DEFAULT_OUTPUT_JSON = "data/iaa/iaa_report.json"


def read_csv_rows(path):
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = []
        for row in reader:
            cleaned = {}
            for key, value in row.items():
                clean_key = key.strip() if isinstance(key, str) else key
                clean_value = value.strip() if isinstance(value, str) else value
                cleaned[clean_key] = clean_value
            rows.append(cleaned)
        return rows


def resolve_path(path):
    if not path:
        return path
    if os.path.isabs(path):
        return path
    return os.path.join(PROJECT_ROOT, path)


def validate_alignment(rows_a, rows_b, subset_rows=None):
    if len(rows_a) != len(rows_b):
        raise ValueError(
            f"Annotator row counts do not match: A={len(rows_a)} vs B={len(rows_b)}."
        )

    if subset_rows is not None and len(rows_a) != len(subset_rows):
        raise ValueError(
            f"Subset row count ({len(subset_rows)}) does not match annotation row count ({len(rows_a)})."
        )

    for idx, (row_a, row_b) in enumerate(zip(rows_a, rows_b), start=1):
        if row_a.get("item_id") != row_b.get("item_id"):
            raise ValueError(f"Row {idx}: item_id mismatch between annotator files.")
        if row_a.get("question") != row_b.get("question"):
            raise ValueError(f"Row {idx}: question mismatch between annotator files.")

        if subset_rows is not None:
            subset_row = subset_rows[idx - 1]
            if row_a.get("item_id") != subset_row.get("item_id"):
                raise ValueError(f"Row {idx}: item_id mismatch against subset.csv.")
            if row_a.get("question") != subset_row.get("question"):
                raise ValueError(f"Row {idx}: question mismatch against subset.csv.")


def score_vs_reference(reference, answer):
    return {
        "exact": compute_exact(reference, answer),
        "canonical_exact": compute_exact(reference, answer, canonical=True),
        "f1": compute_f1(reference, answer),
        "canonical_f1": compute_f1(reference, answer, canonical=True),
    }


def main():
    parser = argparse.ArgumentParser(description="Compute IAA for free-form QA annotations.")
    parser.add_argument("--subset", default=DEFAULT_SUBSET, help="Subset CSV with questions and reference answers.")
    parser.add_argument("--annotator-a", default=DEFAULT_ANNOTATOR_A, help="Annotator A CSV.")
    parser.add_argument("--annotator-b", default=DEFAULT_ANNOTATOR_B, help="Annotator B CSV.")
    parser.add_argument("--output-json", default=DEFAULT_OUTPUT_JSON, help="Optional JSON report output path.")
    parser.add_argument("--hide-disagreements", action="store_true", help="Skip printing item-level disagreements.")
    args = parser.parse_args()

    subset_path = resolve_path(args.subset)
    annotator_a_path = resolve_path(args.annotator_a)
    annotator_b_path = resolve_path(args.annotator_b)
    output_json_path = resolve_path(args.output_json) if args.output_json else None

    subset_rows = read_csv_rows(subset_path) if os.path.exists(subset_path) else None
    rows_a = read_csv_rows(annotator_a_path)
    rows_b = read_csv_rows(annotator_b_path)
    validate_alignment(rows_a, rows_b, subset_rows)

    exact_scores = []
    canonical_exact_scores = []
    f1_scores = []
    canonical_f1_scores = []
    disagreements = []
    annotator_a_vs_ref = []
    annotator_b_vs_ref = []

    for idx, (row_a, row_b) in enumerate(zip(rows_a, rows_b), start=1):
        answer_a = (row_a.get("answer") or "").strip()
        answer_b = (row_b.get("answer") or "").strip()

        exact = compute_exact(answer_a, answer_b)
        canonical_exact = compute_exact(answer_a, answer_b, canonical=True)
        f1 = compute_f1(answer_a, answer_b)
        canonical_f1 = compute_f1(answer_a, answer_b, canonical=True)

        exact_scores.append(exact)
        canonical_exact_scores.append(canonical_exact)
        f1_scores.append(f1)
        canonical_f1_scores.append(canonical_f1)

        item_result = {
            "item_id": row_a.get("item_id", str(idx)),
            "question": row_a.get("question", ""),
            "annotator_a": answer_a,
            "annotator_b": answer_b,
            "annotator_a_canonical": canonicalize_answer(answer_a),
            "annotator_b_canonical": canonicalize_answer(answer_b),
            "exact": exact,
            "canonical_exact": canonical_exact,
            "f1": round(f1, 4),
            "canonical_f1": round(canonical_f1, 4),
        }

        if subset_rows is not None:
            reference = (subset_rows[idx - 1].get("reference_answer") or "").strip()
            item_result["reference_answer"] = reference
            item_result["annotator_a_vs_reference"] = score_vs_reference(reference, answer_a)
            item_result["annotator_b_vs_reference"] = score_vs_reference(reference, answer_b)
            annotator_a_vs_ref.append(item_result["annotator_a_vs_reference"])
            annotator_b_vs_ref.append(item_result["annotator_b_vs_reference"])

        if canonical_exact == 0:
            disagreements.append(item_result)

    summary = {
        "num_items": len(rows_a),
        "agreement": {
            "exact": round(aggregate_mean(exact_scores), 4),
            "canonical_exact": round(aggregate_mean(canonical_exact_scores), 4),
            "token_f1": round(aggregate_mean(f1_scores), 4),
            "canonical_token_f1": round(aggregate_mean(canonical_f1_scores), 4),
            "num_disagreements": len(disagreements),
        },
    }

    if annotator_a_vs_ref and annotator_b_vs_ref:
        summary["annotator_a_vs_reference"] = {
            "exact": round(aggregate_mean([x["exact"] for x in annotator_a_vs_ref]), 4),
            "canonical_exact": round(
                aggregate_mean([x["canonical_exact"] for x in annotator_a_vs_ref]), 4
            ),
            "f1": round(aggregate_mean([x["f1"] for x in annotator_a_vs_ref]), 4),
            "canonical_f1": round(
                aggregate_mean([x["canonical_f1"] for x in annotator_a_vs_ref]), 4
            ),
        }
        summary["annotator_b_vs_reference"] = {
            "exact": round(aggregate_mean([x["exact"] for x in annotator_b_vs_ref]), 4),
            "canonical_exact": round(
                aggregate_mean([x["canonical_exact"] for x in annotator_b_vs_ref]), 4
            ),
            "f1": round(aggregate_mean([x["f1"] for x in annotator_b_vs_ref]), 4),
            "canonical_f1": round(
                aggregate_mean([x["canonical_f1"] for x in annotator_b_vs_ref]), 4
            ),
        }

    print("=" * 64)
    print("IAA SUMMARY")
    print("=" * 64)
    print(f"Items:                 {summary['num_items']}")
    print(f"Exact agreement:       {summary['agreement']['exact']:.4f}")
    print(f"Canonical exact:       {summary['agreement']['canonical_exact']:.4f}")
    print(f"Token F1:              {summary['agreement']['token_f1']:.4f}")
    print(f"Canonical token F1:    {summary['agreement']['canonical_token_f1']:.4f}")
    print(f"Disagreements:         {summary['agreement']['num_disagreements']}")

    if "annotator_a_vs_reference" in summary:
        print("-" * 64)
        print("Annotator A vs reference:")
        print(
            "  exact={exact:.4f} canonical_exact={canonical_exact:.4f} "
            "f1={f1:.4f} canonical_f1={canonical_f1:.4f}".format(
                **summary["annotator_a_vs_reference"]
            )
        )
        print("Annotator B vs reference:")
        print(
            "  exact={exact:.4f} canonical_exact={canonical_exact:.4f} "
            "f1={f1:.4f} canonical_f1={canonical_f1:.4f}".format(
                **summary["annotator_b_vs_reference"]
            )
        )

    if disagreements and not args.hide_disagreements:
        print("-" * 64)
        print("DISAGREEMENTS")
        print("-" * 64)
        for item in disagreements:
            print(f"[{item['item_id']}] {item['question']}")
            print(f"  A: {item['annotator_a']}")
            print(f"  B: {item['annotator_b']}")
            if "reference_answer" in item:
                print(f"  Ref: {item['reference_answer']}")
            print(
                "  Metrics:"
                f" exact={item['exact']}"
                f" canonical_exact={item['canonical_exact']}"
                f" f1={item['f1']:.4f}"
                f" canonical_f1={item['canonical_f1']:.4f}"
            )

    report = {
        "summary": summary,
        "disagreements": disagreements,
    }

    if output_json_path:
        output_dir = os.path.dirname(output_json_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print("-" * 64)
        print(f"Saved JSON report to {output_json_path}")


if __name__ == "__main__":
    main()
