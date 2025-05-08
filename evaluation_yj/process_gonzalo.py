import os
import pandas as pd
import glob
from typing import Dict, List
import re
import numpy as np

def extract_base_doc_id(chunk_id: str) -> str:
    """
    Extract base document ID from chunk ID
    Example: doc_82805800_chunk0 -> doc_82805800
    """
    return re.sub(r'_chunk\d+$', '', chunk_id)

def safe_extract_chunk_number(doc_id: str) -> int:
    """
    Safely extract chunk number from doc_id
    """
    try:
        match = re.search(r'chunk(\d+)', doc_id)
        if match:
            return int(match.group(1))
        return 0
    except:
        return 0

def process_file(input_file: str, output_dir: str) -> None:
    """
    Process a single CSV file, combining chunks for each document

    Args:
        input_file (str): Path to input CSV file
        output_dir (str): Directory to save processed file
    """
    try:
        # Read the input file
        df = pd.read_csv(input_file)

        # Ensure required columns exist
        required_columns = ['doc_id', 'mode', 'generated_summary']
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"Missing required columns. Found: {df.columns.tolist()}")

        # Clean the data
        df = df.dropna(subset=['doc_id', 'generated_summary'])
        df['doc_id'] = df['doc_id'].astype(str)
        df['mode'] = df['mode'].astype(str)
        df['generated_summary'] = df['generated_summary'].astype(str)

        # Create a dictionary to store combined summaries
        combined_data = {}

        # Group by base document ID and mode
        for (base_doc_id, mode), group in df.groupby(
            [df['doc_id'].apply(extract_base_doc_id), 'mode']
        ):
            # Sort by chunk number to ensure correct order
            sorted_group = group.sort_values(
                by='doc_id',
                key=lambda x: x.apply(safe_extract_chunk_number)
            )

            # Combine summaries
            combined_summary = ' '.join(sorted_group['generated_summary'].tolist())

            # Store in dictionary
            if base_doc_id not in combined_data:
                combined_data[base_doc_id] = {}
            combined_data[base_doc_id][mode] = combined_summary

        # Convert to DataFrame
        rows = []
        for doc_id, modes in combined_data.items():
            for mode, summary in modes.items():
                rows.append({
                    'doc_id': doc_id,
                    'mode': mode,
                    'generated_summary': summary
                })

        result_df = pd.DataFrame(rows)

        # Create output filename
        input_filename = os.path.basename(input_file)
        output_file = os.path.join(output_dir, input_filename)

        # Save to CSV
        result_df.to_csv(output_file, index=False)
        print(f"Successfully processed {input_file} -> {output_file}")
        print(f"Combined {len(df)} chunks into {len(result_df)} documents")

    except Exception as e:
        print(f"Error processing {input_file}: {str(e)}")
        print(f"DataFrame info:")
        print(f"Shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        print(f"Sample data:\n{df.head()}")

def main():
    # Define input and output directories
    input_dir = "generated_summary/GONZALO_filtered"
    output_dir = "generated_summary/PRO_GONZALO_filtered"

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Get all CSV files in input directory
    csv_files = glob.glob(os.path.join(input_dir, "*.csv"))

    if not csv_files:
        print(f"No CSV files found in {input_dir}")
        return

    print(f"Found {len(csv_files)} CSV files to process")

    # Process each file
    for csv_file in csv_files:
        print(f"\nProcessing {csv_file}...")
        process_file(csv_file, output_dir)

    print("\nProcessing completed!")

if __name__ == "__main__":
    main()
