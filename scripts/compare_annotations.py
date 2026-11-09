"""Compare two independently annotated CSVs by case ID."""

import argparse
import csv
from pathlib import Path


LABELS = {"reject", "ignore", "accommodate", "adopt", "unclear"}
REQUIRED_COLUMNS = {"case_id", "stance_label"}


def read_annotations(path):
    annotations = {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = REQUIRED_COLUMNS - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path}: missing columns {', '.join(sorted(missing))}")
        for row in reader:
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"{path}: malformed CSV row near line {reader.line_num}")
            case_id = row["case_id"].strip()
            label = row["stance_label"].strip().lower()
            if not case_id or case_id in annotations:
                raise ValueError(f"{path}: blank or duplicate case ID near line {reader.line_num}")
            if label and label not in LABELS:
                raise ValueError(f"{path}: invalid label {label!r} for {case_id}")
            annotations[case_id] = (label.capitalize(), row.get("evidence_quote", ""))
    if not annotations:
        raise ValueError(f"{path}: no cases found")
    return annotations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("first", type=Path, help="First annotator's CSV")
    parser.add_argument("second", type=Path, help="Second annotator's CSV")
    parser.add_argument("--output", type=Path, default=Path("annotation_disagreements.csv"))
    args = parser.parse_args()

    try:
        if args.output.resolve() in {args.first.resolve(), args.second.resolve()}:
            raise ValueError("The report must have a different path from both inputs.")
        first = read_annotations(args.first)
        second = read_annotations(args.second)
        shared = sorted(first.keys() & second.keys())
        compared = 0
        disagreements = []
        for case_id in shared:
            first_label, first_quote = first[case_id]
            second_label, second_quote = second[case_id]
            if not first_label or not second_label:
                continue
            compared += 1
            if first_label != second_label:
                disagreements.append((case_id, first_label, second_label, first_quote, second_quote))

        with args.output.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(("case_id", "first_label", "second_label", "first_evidence", "second_evidence"))
            writer.writerows(disagreements)
    except (OSError, UnicodeError, ValueError, csv.Error) as error:
        parser.exit(1, f"Comparison failed: {error}\n")

    for name, cases in (
        ("IDs only in first CSV", sorted(first.keys() - second.keys())),
        ("IDs only in second CSV", sorted(second.keys() - first.keys())),
        ("Shared IDs with blank first label", [key for key in shared if not first[key][0]]),
        ("Shared IDs with blank second label", [key for key in shared if not second[key][0]]),
    ):
        print(f"{name}: {len(cases)}" + (f" ({', '.join(cases)})" if cases else ""))

    agreed = compared - len(disagreements)
    if compared:
        print(f"Exact agreement: {agreed}/{compared} ({agreed / compared:.1%})")
    else:
        print("Exact agreement: unavailable (no jointly labeled cases)")
    print(f"Wrote {len(disagreements)} disagreements to {args.output}")


if __name__ == "__main__":
    main()
