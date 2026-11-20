
import csv
from scipy.stats import binomtest
from high_vs_low_paired import audit, ROOT

dataset = ROOT / "arc_challenge_100.csv"
results_dir = ROOT / "Results"

summary = audit(dataset, results_dir)

output_rows = []

for row in summary:
    high_only = int(row["high_only_flips"])
    low_only = int(row["low_only_flips"])
    discordant = high_only + low_only

    p_value = (
        binomtest(high_only, discordant, p=0.5).pvalue
        if discordant > 0
        else 1.0
    )

    result = {
        **row,
        "mcnemar_p_value": round(p_value, 6),
        "significant_at_0_05": p_value < 0.05
    }

    output_rows.append(result)

    print(
        f"{row['model']}: "
        f"high_only={high_only}, "
        f"low_only={low_only}, "
        f"p_value={p_value:.6f}"
    )

output_path = ROOT / "statistical_significance_results.csv"

with output_path.open("w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=output_rows[0].keys())
    writer.writeheader()
    writer.writerows(output_rows)

print(f"\nResults saved to {output_path}")
