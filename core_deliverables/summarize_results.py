import os
import pandas as pd


def main():
    base_dir = os.path.dirname(__file__)
    csv_path = os.path.join(base_dir, "model_results.csv")

    if not os.path.exists(csv_path):
        print("No model_results.csv found. Run the core model scripts first.")
        return

    df = pd.read_csv(csv_path)

    # Pretty, compact table for the report
    print("\n=== Summary of Core Deliverables ===")
    with pd.option_context("display.max_columns", None, "display.width", 120):
        print(df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main()

