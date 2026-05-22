"""
Sample a random subset from the QA test set for inter-annotator agreement (IAA).
"""
import argparse
import csv
import os
import random


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
DEFAULT_QUESTIONS = "data/test/questions.txt"
DEFAULT_REFERENCES = "data/test/reference_answers.txt"
DEFAULT_OUTPUT_DIR = "data/iaa"


def read_nonempty_lines(path):
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def write_csv(path, fieldnames, rows):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def resolve_path(path):
    if os.path.isabs(path):
        return path
    return os.path.join(PROJECT_ROOT, path)


def main():
    parser = argparse.ArgumentParser(description="Sample an IAA subset from the test set.")
    parser.add_argument("--questions", default=DEFAULT_QUESTIONS, help="Path to test questions.")
    parser.add_argument("--references", default=DEFAULT_REFERENCES, help="Path to reference answers.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Directory for IAA files.")
    parser.add_argument("--size", type=int, default=8, help="Subset size to sample.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    questions_path = resolve_path(args.questions)
    references_path = resolve_path(args.references)
    output_dir = resolve_path(args.output_dir)

    questions = read_nonempty_lines(questions_path)
    references = read_nonempty_lines(references_path)

    if len(questions) != len(references):
        raise ValueError(
            f"Question count ({len(questions)}) does not match reference count ({len(references)})."
        )

    if not questions:
        raise ValueError("The test set is empty.")

    subset_size = min(args.size, len(questions))
    rng = random.Random(args.seed)
    sampled_indices = sorted(rng.sample(range(len(questions)), subset_size))

    os.makedirs(output_dir, exist_ok=True)

    subset_rows = []
    annotator_a_rows = []
    annotator_b_rows = []

    for item_id, idx in enumerate(sampled_indices, start=1):
        row = {
            "item_id": item_id,
            "source_index": idx + 1,
            "question": questions[idx],
            "reference_answer": references[idx],
        }
        subset_rows.append(row)

        blank_annotation = {
            "item_id": item_id,
            "source_index": idx + 1,
            "question": questions[idx],
            "answer": "",
        }
        annotator_a_rows.append(blank_annotation.copy())
        annotator_b_rows.append(blank_annotation.copy())

    subset_path = os.path.join(output_dir, "subset.csv")
    annotator_a_path = os.path.join(output_dir, "annotator_a.csv")
    annotator_b_path = os.path.join(output_dir, "annotator_b.csv")

    write_csv(
        subset_path,
        ["item_id", "source_index", "question", "reference_answer"],
        subset_rows,
    )
    write_csv(
        annotator_a_path,
        ["item_id", "source_index", "question", "answer"],
        annotator_a_rows,
    )
    write_csv(
        annotator_b_path,
        ["item_id", "source_index", "question", "answer"],
        annotator_b_rows,
    )

    print(f"Sampled {subset_size} / {len(questions)} test items with seed={args.seed}")
    print(f"Subset file:      {subset_path}")
    print(f"Annotator A file: {annotator_a_path}")
    print(f"Annotator B file: {annotator_b_path}")
    print("Note: Keep subset.csv for adjudication only; annotators should fill only their own CSV.")


if __name__ == "__main__":
    main()
