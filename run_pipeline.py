#!/usr/bin/env python3

import os
import argparse
from knowledge_pipeline import KnowledgeModelingPipeline

def main():
    parser = argparse.ArgumentParser(description="Run the knowledge modeling pipeline with a sample document")
    parser.add_argument("--input", required=True, help="Path to input document (PDF or JSON)")
    parser.add_argument("--output", default="pipeline_output.json", help="Path to save pipeline output")
    parser.add_argument("--clusters", type=int, default=10, help="Number of concept clusters")
    args = parser.parse_args()
    
    print(f"Running pipeline with input: {args.input}")
    
    # Initialize pipeline
    pipeline = KnowledgeModelingPipeline(
        num_clusters=args.clusters
    )
    
    # Run pipeline
    try:
        _, text_items, concept_clusters, gnn_input = pipeline.run_pipeline(
            args.input, 
            args.output
        )
        
        print("\n--- Pipeline Complete ---")
        print(f"Processed {len(text_items)} text items")
        print(f"Created {len(concept_clusters)} concept clusters")
        print(f"GNN input prepared with {gnn_input.num_nodes} nodes and {gnn_input.num_edges} edges")
        print(f"Output saved to {args.output}")
        
    except Exception as e:
        print(f"Error running pipeline: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 