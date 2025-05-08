import os
import pandas as pd
from metrics import evaluate_text
from typing import Dict, List
import glob

def load_summaries() -> Dict[str, pd.DataFrame]:
    """
    Load GPT graph summary and highlight file

    Returns:
        Dict[str, pd.DataFrame]: Dictionary containing DataFrames for summaries
    """
    summary_dir = "generated_summary"

    # Load highlight annotations
    highlight_path = os.path.join(summary_dir, "highlight.csv")
    if not os.path.exists(highlight_path):
        raise FileNotFoundError(f"Highlight file not found at {highlight_path}")

    # Load GPT graph summary
    gpt_path = os.path.join(summary_dir, "gpt_graph.csv")
    if not os.path.exists(gpt_path):
        raise FileNotFoundError(f"GPT graph summary not found at {gpt_path}")

    summaries = {
        "HIGHLIGHT": pd.read_csv(highlight_path),
        "GPT_GRAPH": pd.read_csv(gpt_path)
    }

    return summaries

def evaluate_file(doc_id: str, summaries: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Evaluate GPT graph summary for a single document

    Args:
        doc_id (str): Document ID
        summaries (Dict[str, pd.DataFrame]): Dictionary containing summary DataFrames

    Returns:
        pd.DataFrame: Evaluation results
    """
    try:
        # Get GPT graph summary
        gpt_summary = summaries["GPT_GRAPH"].loc[
            summaries["GPT_GRAPH"]["doc_id"] == doc_id, "generated_summary"
        ].iloc[0]

        # Get highlight annotations if available
        highlight_mask = summaries["HIGHLIGHT"]["doc_id"] == doc_id
        if not highlight_mask.any():
            print(f"Warning: No highlight found for {doc_id}")
            return pd.DataFrame()

        highlight = summaries["HIGHLIGHT"].loc[highlight_mask, "highlight"].iloc[0]

        # Evaluate
        metrics = evaluate_text(
            reference=highlight,  # Use highlight as reference
            candidate=gpt_summary,
            highlight=highlight
        )

        metrics.update({
            "doc_id": doc_id,
            "mode": "GPT_GRAPH"
        })

        return pd.DataFrame([metrics])
    except Exception as e:
        print(f"Error evaluating {doc_id}: {str(e)}")
        return pd.DataFrame()

def main():
    # Create results directory if it doesn't exist
    os.makedirs("results", exist_ok=True)
    os.makedirs("results/GPT_GRAPH_RESULT", exist_ok=True)

    # Load summaries
    print("Loading summaries...")
    try:
        summaries = load_summaries()
    except Exception as e:
        print(f"Error loading summaries: {str(e)}")
        return

    if not summaries:
        print("No summary files found!")
        return

    # Get unique document IDs from GPT graph summary
    doc_ids = summaries["GPT_GRAPH"]["doc_id"].unique()
    print(f"Found {len(doc_ids)} documents in GPT graph summary")

    # Process each document
    all_results = []
    for doc_id in doc_ids:
        try:
            results = evaluate_file(doc_id, summaries)
            if not results.empty:
                all_results.append(results)

                # Save individual file results
                results_path = os.path.join("results/GPT_GRAPH_RESULT", f"{doc_id}_metrics.csv")
                results.to_csv(results_path, index=False)
                print(f"Saved metrics for {doc_id} to {results_path}")

        except Exception as e:
            print(f"Error processing {doc_id}: {str(e)}")

    if all_results:
        # Combine all results
        combined_results = pd.concat(all_results, ignore_index=True)

        # Calculate average metrics
        avg_metrics = pd.DataFrame([{
            'mode': 'GPT_GRAPH',
            'rouge-1': combined_results['rouge-1'].mean(),
            'rouge-L': combined_results['rouge-L'].mean(),
            'highlight-p': combined_results['highlight-p'].mean(),
            'highlight-r': combined_results['highlight-r'].mean(),
            'highlight-f1': combined_results['highlight-f1'].mean()
        }])

        # Save average metrics
        avg_path = os.path.join("results/GPT_GRAPH_RESULT", "average_metrics.csv")
        avg_metrics.to_csv(avg_path, index=False)
        print(f"\nSaved average metrics to {avg_path}")

        # Display average metrics
        print("\nAverage Metrics across all files:")
        print(avg_metrics)

        # Save detailed results
        detailed_path = os.path.join("results/GPT_GRAPH_RESULT", "detailed_metrics.csv")
        combined_results.to_csv(detailed_path, index=False)
        print(f"Saved detailed metrics to {detailed_path}")
    else:
        print("\nNo valid results to save!")

if __name__ == "__main__":
    main()
