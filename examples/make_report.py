from __future__ import annotations

from databaseMLUtils.reporting import class_distribution_report


if __name__ == "__main__":
    df = class_distribution_report("results/crops", out_csv="results/reports/class_dist.csv")
    print(df)

