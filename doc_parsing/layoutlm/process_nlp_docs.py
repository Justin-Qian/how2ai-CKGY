#!/usr/bin/env python
"""
Script to process all PDF files in a directory and output statistics.
"""
import os
import sys
import json
import subprocess
import argparse
from pathlib import Path
from collections import Counter
import shutil

def count_annotations_by_type(json_path):
    """Count the number of annotations by type in a processed JSON file."""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        annotations = data.get('annotations', [])
        counts = Counter(anno.get('type', 'unknown') for anno in annotations)
        return counts, len(annotations)
    except Exception as e:
        print(f"Error processing {json_path}: {e}")
        return Counter(), 0

def main():
    """Process all PDF files and generate statistics."""
    parser = argparse.ArgumentParser(description="Process all PDF files in a directory and output statistics.")
    parser.add_argument("--input_dir", default="PDF_data/NLP", help="Input directory with PDF files")
    parser.add_argument("--output_dir", default="doc_parsing/layoutlm/output_503", help="Output directory for processed files")
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get list of PDF files
    input_dir = Path(args.input_dir)
    pdf_files = list(input_dir.glob("*.pdf"))
    
    if not pdf_files:
        print(f"No PDF files found in {args.input_dir}")
        return
    
    print(f"Found {len(pdf_files)} PDF files to process")
    
    # Process each PDF file
    results = []
    
    for pdf_file in pdf_files:
        filename = pdf_file.name
        output_json = os.path.join(args.output_dir, f"{pdf_file.stem}_processed.json")
        output_vis = os.path.join(args.output_dir, f"{pdf_file.stem}_visualized.pdf")
        
        print(f"\nProcessing {filename}...")
        
        # Run document processor
        try:
            cmd = [
                sys.executable, 
                "-m", "doc_parsing.layoutlm.document_processor", 
                "--pdf_path", str(pdf_file)
            ]
            print(f"  Running command: {' '.join(cmd)}")
            # Remove capture_output=True to allow output to appear in real-time
            process = subprocess.run(cmd, text=True, check=False)
            print(f"  Document processor completed with return code: {process.returncode}")
            
            # Move output file to the target directory
            source_json = f"doc_parsing/layoutlm/output/{pdf_file.stem}_processed.json"
            if os.path.exists(source_json):
                shutil.copy2(source_json, output_json)
                print(f"  Copied JSON to {output_json}")
            else:
                print(f"  Warning: Output JSON not found at {source_json}")
            
            # Run visualization
            if os.path.exists(output_json):
                vis_cmd = [
                    sys.executable,
                    "-m", "doc_parsing.layoutlm.visualize_new_format",
                    "--pdf_path", str(pdf_file),
                    "--json_path", output_json,
                    "--output_path", output_vis
                ]
                print(f"  Running visualization: {' '.join(vis_cmd)}")
                # Remove capture_output=True to allow output to appear in real-time
                subprocess.run(vis_cmd, text=True, check=False)
                print(f"  Created visualization at {output_vis}")
            
            # Count annotations
            if os.path.exists(output_json):
                annotation_counts, total_count = count_annotations_by_type(output_json)
                comment_count = annotation_counts.get('comment', 0)
                results.append({
                    'filename': filename,
                    'total_annotations': total_count,
                    'comment_count': comment_count,
                    'annotation_counts': dict(annotation_counts)
                })
                print(f"  Found {comment_count} comments out of {total_count} total annotations")
            
        except subprocess.CalledProcessError as e:
            print(f"  Error processing {filename}: {e}")
            print(f"  Stdout: {e.stdout}")
            print(f"  Stderr: {e.stderr}")
        except Exception as e:
            print(f"  Error: {e}")
    
    # Rank files by comment count
    results.sort(key=lambda x: x['comment_count'], reverse=True)
    
    # Output summary
    print("\n=== Documents Ranked by Comment Count ===")
    for i, result in enumerate(results, 1):
        print(f"{i}. {result['filename']}: {result['comment_count']} comments (total annotations: {result['total_annotations']})")
    
    # Save results to JSON
    summary_path = os.path.join(args.output_dir, "processing_summary.json")
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump({
            'ranked_by_comments': results
        }, f, indent=2)
    
    print(f"\nSummary saved to {summary_path}")

if __name__ == "__main__":
    main() 