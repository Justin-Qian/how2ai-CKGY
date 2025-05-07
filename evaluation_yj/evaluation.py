import os
import pandas as pd
from metrics import evaluate_text
from typing import Dict, List
import glob

def load_summaries() -> Dict[str, pd.DataFrame]:
    """
    Load all summary files from generated_summary directory

    Returns:
        Dict[str, pd.DataFrame]: Dictionary containing DataFrames for each mode
    """
    summary_dir = "generated_summary"
    mode_files = {
        "ORIGINAL": "ORIGINAL.csv",
        "B-TEXT": "B_TEXT.csv",
        "A-TAG": "A_TAG.csv",
        "A-ONLY": "A_ONLY.csv",
        "A-ADD": "A_ADD.csv"
    }

    summaries = {}
    for mode, filename in mode_files.items():
        file_path = os.path.join(summary_dir, filename)
        if os.path.exists(file_path):
            summaries[mode] = pd.read_csv(file_path)
        else:
            print(f"Warning: {filename} not found in {summary_dir}")

    return summaries

def evaluate_file(doc_id: str, summaries: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Evaluate summaries for a single document

    Args:
        doc_id (str): Document ID
        summaries (Dict[str, pd.DataFrame]): Dictionary containing all summary DataFrames

    Returns:
        pd.DataFrame: Evaluation results
    """
    # Get original summary
    original_summary = summaries["ORIGINAL"].loc[
        summaries["ORIGINAL"]["doc_id"] == doc_id, "generated_summary"
    ].iloc[0]

    # Get annotations (from A-ONLY mode)
    annotations = summaries["A-ONLY"].loc[
        summaries["A-ONLY"]["doc_id"] == doc_id, "generated_summary"
    ].iloc[0]

    # Evaluate each mode
    results = []
    modes = ["ORIGINAL", "B-TEXT", "A-TAG", "A-ONLY", "A-ADD"]

    for mode in modes:
        if mode in summaries:
            candidate_summary = summaries[mode].loc[
                summaries[mode]["doc_id"] == doc_id, "generated_summary"
            ].iloc[0]

            metrics = evaluate_text(
                reference=original_summary,
                candidate=candidate_summary,
                highlight=annotations
            )

            metrics.update({
                "doc_id": doc_id,
                "mode": mode
            })
            results.append(metrics)

    return pd.DataFrame(results)

def main():
    # Create results directory if it doesn't exist
    os.makedirs("results", exist_ok=True)

    # Load all summaries
    print("Loading summaries...")
    summaries = load_summaries()

    if not summaries:
        print("No summary files found!")
        return

    # Get unique document IDs from ORIGINAL summaries
    doc_ids = summaries["ORIGINAL"]["doc_id"].unique()

    # Process each document
    all_results = []
    for doc_id in doc_ids:
        try:
            results = evaluate_file(doc_id, summaries)
            all_results.append(results)

            # Save individual file results
            results_path = os.path.join("results", f"{doc_id}_metrics.csv")
            results.to_csv(results_path, index=False)
            print(f"Saved metrics for {doc_id} to {results_path}")

        except Exception as e:
            print(f"Error processing {doc_id}: {str(e)}")

    if all_results:
        # Combine all results
        combined_results = pd.concat(all_results, ignore_index=True)

        # Calculate and save average metrics across all files
        avg_metrics = combined_results.groupby("mode")[
            ["rouge-1", "rouge-L", "highlight-p", "highlight-r", "highlight-f1"]
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
