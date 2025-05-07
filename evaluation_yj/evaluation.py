import os
import pandas as pd
from metrics import evaluate_text
from typing import Dict, List
import glob

def load_summaries(filename: str) -> Dict[str, str]:
    """
    Load original and generated summaries for a given filename

    Args:
        filename (str): Base filename without extension

    Returns:
        Dict[str, str]: Dictionary containing original and generated summaries
    """
    # Load original summary
    original_path = os.path.join("original_summary", f"{filename}.txt")
    with open(original_path, "r", encoding="utf-8") as f:
        original_summary = f.read().strip()

    # Load generated summaries
    generated_path = os.path.join("baseline01_summary", f"{filename}.csv")
    generated_df = pd.read_csv(generated_path)

    summaries = {
        "original": original_summary,
        "annotations": generated_df.loc[generated_df["mode"] == "A-ONLY", "generated_summary"].iloc[0]
    }

    # Add generated summaries by mode
    for _, row in generated_df.iterrows():
        summaries[row["mode"]] = row["generated_summary"]

    return summaries

def evaluate_file(filename: str) -> pd.DataFrame:
    """
    Evaluate summaries for a single file

    Args:
        filename (str): Base filename without extension

    Returns:
        pd.DataFrame: Evaluation results
    """
    # Load summaries
    summaries = load_summaries(filename)

    # Evaluate each mode
    results = []
    modes = ["B-TEXT", "A-TAG", "A-ONLY", "A-ADD"]

    for mode in modes:
        metrics = evaluate_text(
            reference=summaries["original"],
            candidate=summaries[mode],
            highlight=summaries["annotations"]
        )

        metrics.update({
            "filename": filename,
            "mode": mode
        })
        results.append(metrics)

    return pd.DataFrame(results)

def main():
    # Get all original summary files
    original_files = glob.glob(os.path.join("original_summary", "*.txt"))

    # Extract base filenames without extension
    filenames = [os.path.splitext(os.path.basename(f))[0] for f in original_files]

    # Create results directory if it doesn't exist
    os.makedirs("results", exist_ok=True)

    # Process each file
    all_results = []
    for filename in filenames:
        try:
            results = evaluate_file(filename)
            all_results.append(results)

            # Save individual file results
            results_path = os.path.join("results", f"{filename}_metrics.csv")
            results.to_csv(results_path, index=False)
            print(f"Saved metrics for {filename} to {results_path}")

        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")

    if all_results:
        # Combine all results
        combined_results = pd.concat(all_results, ignore_index=True)

        # Calculate and save average metrics across all files
        avg_metrics = combined_results.groupby("mode")[
            ["bleu", "rouge-1", "rouge-L", "highlight-p", "highlight-r", "highlight-f1"]
        ].mean()

        # Save average metrics
        avg_path = os.path.join("results", "average_metrics.csv")
        avg_metrics.to_csv(avg_path)
        print(f"\nSaved average metrics to {avg_path}")

        # Display average metrics
        print("\nAverage Metrics across all files:")
        print(avg_metrics)

        # Save detailed results
        detailed_path = os.path.join("results", "detailed_metrics.csv")
        combined_results.to_csv(detailed_path, index=False)
        print(f"Saved detailed metrics to {detailed_path}")

if __name__ == "__main__":
    main()
