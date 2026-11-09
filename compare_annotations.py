"""Compare independently annotated CSVs by case ID.

Examples
  python3 compare_annotations.py a.csv b.csv
  python3 compare_annotations.py a.csv b.csv c.csv d.csv --output disagreements.csv

Each UTF-8 CSV needs case_id and stance_label. Use the same case IDs
across files. The evidence_quote column is optional. Valid labels are
Reject, Ignore, Accommodate, Adopt, and Unclear. Leave pending labels blank.

Unanimous agreement uses cases labeled by everyone. Each pairwise rate
uses cases labeled by both people in that pair. Rates are descriptive.
The output contains disagreements among fully labeled cases and defaults
to annotation_disagreements.csv.
"""

import argparse
import csv
from itertools import combinations
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


def compare_group(annotations, output):
    case_ids = sorted(set().union(*annotations))
    indexed = [
        (case_id, [sheet.get(case_id, ("", "")) for sheet in annotations])
        for case_id in case_ids
    ]
    complete = [(case_id, votes) for case_id, votes in indexed if all(label for label, _ in votes)]
    disagreements = [
        (case_id, votes) for case_id, votes in complete
        if len({label for label, _ in votes}) > 1
    ]

    headers = ["case_id"]
    for number in range(1, len(annotations) + 1):
        headers.extend((f"annotator_{number}_label", f"annotator_{number}_evidence"))
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(headers)
        for case_id, votes in disagreements:
            writer.writerow([case_id, *(value for vote in votes for value in vote)])

    print(f"Cases labeled by everyone: {len(complete)}/{len(case_ids)}")
    for number, sheet in enumerate(annotations, start=1):
        absent = sum(case_id not in sheet for case_id in case_ids)
        blank = sum(not label for label, _ in sheet.values())
        print(f"Annotator {number}: {absent} missing IDs, {blank} blank labels")
    print_agreement("Unanimous agreement", len(complete) - len(disagreements), len(complete))
    for i, j in combinations(range(len(annotations)), 2):
        pairs = [(votes[i][0], votes[j][0]) for _, votes in indexed if votes[i][0] and votes[j][0]]
        print_agreement(f"Annotators {i + 1} vs {j + 1}", sum(a == b for a, b in pairs), len(pairs))
    print(f"Wrote {len(disagreements)} disagreements to {output}")


def print_agreement(name, agreed, compared):
    if compared:
        print(f"{name}: {agreed}/{compared} ({agreed / compared:.1%})")
    else:
        print(f"{name}: unavailable (no jointly labeled cases)")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("first", type=Path, help="First annotator's CSV")
    parser.add_argument("second", type=Path, help="Second annotator's CSV")
    parser.add_argument("additional", nargs="*", type=Path, help="Additional annotators’ CSVs")
    parser.add_argument("--output", type=Path, default=Path("annotation_disagreements.csv"))
    args = parser.parse_args()

    try:
        inputs = [args.first, args.second, *args.additional]
        paths = {path.resolve() for path in inputs}
        if len(paths) != len(inputs):
            raise ValueError("Each annotator must have a separate input CSV.")
        if args.output.resolve() in paths:
            raise ValueError("The report must have a different path from both inputs.")
        annotations = [read_annotations(path) for path in inputs]
        if args.additional:
            for number, path in enumerate(inputs, start=1):
                print(f"Annotator {number}: {path}")
            compare_group(annotations, args.output)
            return
        first, second = annotations
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
    print_agreement("Exact agreement", agreed, compared)
    print(f"Wrote {len(disagreements)} disagreements to {args.output}")


if __name__ == "__main__":
    main()
