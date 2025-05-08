import os
import json
import pandas as pd
from typing import List, Dict, Tuple

def load_data(json_path: str) -> Tuple[str, str]:
    """
    Load and extract highlight information from a single document JSON file

    Args:
        json_path (str): Path to JSON file containing a single document

    Returns:
        Tuple[str, str]: (doc_id, highlight_text)
    """
    try:
        # Load original JSON
        with open(json_path, "r", encoding="utf-8") as f:
            doc = json.load(f)

        # Extract document ID
        doc_id = doc["document"]["id"]

        # Extract annotations
        annotations = doc["annotations"]

        # Filter and sort highlights
        highlights = [
            a["annotated_text"]
            for a in sorted(annotations, key=lambda x: x["referenced_char_start"])
            if a["type"] == "highlight"
        ]

        # Join highlights with space
        highlight_text = " ".join(highlights)

        return doc_id, highlight_text
    except KeyError as e:
        raise ValueError(f"Missing required field in JSON: {str(e)}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format: {str(e)}")
    except Exception as e:
        raise ValueError(f"Error processing file: {str(e)}")

def main():
    # Set paths
    data_dir = "data"
    output_dir = "generated_summary"

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Get all JSON files in the data directory
    json_files = [f for f in os.listdir(data_dir) if f.endswith('.json')]

    if not json_files:
        print(f"No JSON files found in {data_dir}")
        return

    print(f"Found {len(json_files)} JSON files to process")

    # Initialize DataFrame for results
    results_df = pd.DataFrame(columns=["doc_id", "highlight"])

    # Process each JSON file
    for json_file in json_files:
        json_path = os.path.join(data_dir, json_file)
        json_filename = os.path.splitext(json_file)[0]

        print(f"\nProcessing {json_file}...")

        try:
            # Load data and extract highlights
            doc_id, highlight_text = load_data(json_path)

            # Add to DataFrame
            new_row = pd.DataFrame([{
                "doc_id": doc_id,
                "highlight": highlight_text
            }])
            results_df = pd.concat([results_df, new_row], ignore_index=True)

            print(f"✅ Extracted highlights for {doc_id}")

        except Exception as e:
            print(f"❌ Error processing {json_file}: {str(e)}")
            continue

    # Save results
    if not results_df.empty:
        output_file = os.path.join(output_dir, "highlight.csv")
        results_df.to_csv(output_file, index=False)
        print(f"\n✅ Saved highlights to {output_file}")
        print(f"Processed {len(results_df)} documents")
    else:
        print("\n❌ No results to save!")

if __name__ == "__main__":
    main()
