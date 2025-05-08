import os
import pandas as pd
from metrics import evaluate_text
from typing import Dict, List
import glob

def load_summaries() -> Dict[str, pd.DataFrame]:
    """
    Load all summary files from PRO_GONZALO directory and highlight file

    Returns:
        Dict[str, pd.DataFrame]: Dictionary containing DataFrames for each mode
    """
    summary_dir = "generated_summary"
    pro_gonzalo_dir = os.path.join(summary_dir, "PRO_GONZALO_filtered")

    # Load highlight annotations
    highlight_path = os.path.join(summary_dir, "highlight.csv")
    if not os.path.exists(highlight_path):
        raise FileNotFoundError(f"Highlight file not found at {highlight_path}")

    summaries = {
        "HIGHLIGHT": pd.read_csv(highlight_path)
    }

    # Load all CSV files from PRO_GONZALO directory
    csv_files = glob.glob(os.path.join(pro_gonzalo_dir, "*.csv"))
    for csv_file in csv_files:
        mode = os.path.splitext(os.path.basename(csv_file))[0]
        summaries[mode] = pd.read_csv(csv_file)

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
    # Get highlight annotations
    highlight = summaries["HIGHLIGHT"].loc[
        summaries["HIGHLIGHT"]["doc_id"] == doc_id, "highlight"
    ].iloc[0]

    # Evaluate each mode
    results = []
    modes = [mode for mode in summaries.keys() if mode != "HIGHLIGHT"]

    for mode in modes:
        try:
            candidate_summary = summaries[mode].loc[
                summaries[mode]["doc_id"] == doc_id, "generated_summary"
            ].iloc[0]

            metrics = evaluate_text(
                reference=highlight,  # Use highlight as reference
                candidate=candidate_summary,
                highlight=highlight
            )

            metrics.update({
                "doc_id": doc_id,
                "mode": mode
            })
            results.append(metrics)
        except Exception as e:
            print(f"Error evaluating {mode} for {doc_id}: {str(e)}")

    return pd.DataFrame(results)

def main():
    # Create results directory if it doesn't exist
    os.makedirs("results", exist_ok=True)
    os.makedirs("results/GONZALO_FILTERED_RESULT", exist_ok=True)

    # Load all summaries
    print("Loading summaries...")
    try:
        summaries = load_summaries()
    except Exception as e:
        print(f"Error loading summaries: {str(e)}")
        return

    if not summaries:
        print("No summary files found!")
        return

    # Get unique document IDs from highlight file
    doc_ids = summaries["HIGHLIGHT"]["doc_id"].unique()

    # Process each document
    all_results = []
    for doc_id in doc_ids:
        try:
            results = evaluate_file(doc_id, summaries)
            all_results.append(results)

            # Save individual file results
            results_path = os.path.join("results/GONZALO_FILTERED_RESULT", f"{doc_id}_metrics.csv")
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
        avg_path = os.path.join("results/GONZALO_FILTERED_RESULT", "average_metrics.csv")
        avg_metrics.to_csv(avg_path)
        print(f"\nSaved average metrics to {avg_path}")

        # Display average metrics
        print("\nAverage Metrics across all files:")
        print(avg_metrics)

        # Save detailed results
        detailed_path = os.path.join("results/GONZALO_FILTERED_RESULT", "detailed_metrics.csv")
        combined_results.to_csv(detailed_path, index=False)
        print(f"Saved detailed metrics to {detailed_path}")

if __name__ == "__main__":
    main()
