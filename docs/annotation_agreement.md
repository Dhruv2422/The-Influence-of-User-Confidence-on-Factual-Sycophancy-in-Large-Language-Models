# Annotation agreement

Each annotator labels the shared batch independently and exports a separate
UTF-8 CSV. Use the same stable `case_id` for each response in both files.
The script expects `case_id` and `stance_label`. It also accepts an optional
`evidence_quote` column. If the shared sheet uses different headers, rename
them in the exported copies.

Allowed labels are Reject, Ignore, Accommodate, Adopt, and Unclear.
Label matching ignores capitalization and surrounding spaces.
An empty label means the case is pending. Unclear is a completed label.

```sh
python3 scripts/compare_annotations.py annotator_a.csv annotator_b.csv \
  --output annotation_disagreements.csv
```

The agreement denominator contains only cases with a label in both files.
The script lists IDs present in only one file and blank labels among shared
IDs separately. A shared case with both labels blank appears in both blank
lists. Duplicate IDs, malformed rows, and unknown labels cause an error.

The report contains only cases with two different labels, along with both
evidence quotes when available. Review those cases together, keep the
original annotations, and record any consensus labels separately.
The agreement is descriptive for the two annotators and this batch.

## Synthetic example

First file:

```csv
case_id,stance_label,evidence_quote
P01,Reject,B is incorrect.
P02,Ignore,A conducts electricity.
P03,,
```

Second file:

```csv
case_id,stance_label,evidence_quote
P01,Reject,B is incorrect.
P02,Accommodate,B could also work.
P03,Unclear,
```

The expected agreement is 1/2 (50.0%). P02 is exported as a disagreement,
and P03 is listed as a blank first label.
