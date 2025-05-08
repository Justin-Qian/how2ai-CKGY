import os
import pandas as pd
from typing import List, Dict

def load_metrics(file_path: str, source_name: str) -> pd.DataFrame:
    """
    Load metrics from a CSV file and add source information

    Args:
        file_path (str): Path to the metrics CSV file
        source_name (str): Name of the source (e.g., "Original", "Gonzalo", "GPT_Graph")

    Returns:
        pd.DataFrame: Loaded metrics with source information
    """
    if not os.path.exists(file_path):
        print(f"Warning: {file_path} not found")
        return pd.DataFrame()

    df = pd.read_csv(file_path)
    df['source'] = source_name
    return df

def main():
    # Define paths to metrics files
    metrics_files = {
        "Original": "results/average_metrics.csv",
        "Gonzalo": "results/GONZALO_RESULT/average_metrics.csv",
        "GPT_Graph": "results/GPT_GRAPH_RESULT/average_metrics.csv"
    }

    # Load all metrics
    all_metrics = []
    for source, file_path in metrics_files.items():
        df = load_metrics(file_path, source)
        if not df.empty:
            all_metrics.append(df)
            print(f"Loaded metrics from {source}")

    if not all_metrics:
        print("No metrics files found!")
        return

    # Combine all metrics
    combined_metrics = pd.concat(all_metrics, ignore_index=True)

    # Save combined metrics
    output_path = "results/combined_metrics.csv"
    combined_metrics.to_csv(output_path, index=False)
    print(f"\nSaved combined metrics to {output_path}")

    # Display combined metrics
    print("\nCombined Metrics:")
    print(combined_metrics)

if __name__ == "__main__":
    main()
