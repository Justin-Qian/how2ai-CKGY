#!/usr/bin/env python3

import os
import json
import argparse
import torch
from torch_geometric.data import Data

# Import our pipeline and GNN components
from knowledge_pipeline import KnowledgeModelingPipeline
from gnn.model import StaticGCN
import gnn.config as gnn_config

def run_complete_pipeline(args):
    """
    Run the complete pipeline from document processing to GNN training and evaluation.
    
    Args:
        args: Command-line arguments
    """
    print("\n=== STAGE 1: KNOWLEDGE MODELING PIPELINE ===")
    
    # Initialize the knowledge modeling pipeline
    pipeline = KnowledgeModelingPipeline(
        embedding_model_name=args.embeddings,
        similarity_threshold=args.similarity,
        num_clusters=args.clusters
    )
    
    # Run the pipeline to get processed document and GNN input data
    _, text_items, concept_clusters, gnn_data = pipeline.run_pipeline(
        args.input, 
        args.pipeline_output
    )
    
    print(f"\nCreated {len(concept_clusters)} concept clusters")
    print(f"Not understood clusters: {sum(1 for c in concept_clusters if c['not_understood'])}")
    print(f"GNN input prepared with {gnn_data.num_nodes} nodes and {gnn_data.num_edges} edges")
    
    if not args.skip_training:
        print("\n=== STAGE 2: GNN MODEL TRAINING ===")
        
        # Train the GNN model
        train_and_evaluate_gnn(gnn_data, args)
    
    return gnn_data, concept_clusters

def train_and_evaluate_gnn(gnn_data, args):
    """
    Train and evaluate the GNN model on the prepared data.
    
    Args:
        gnn_data: PyTorch Geometric Data object
        args: Command-line arguments
    """
    device = torch.device(gnn_config.DEVICE)
    
    # Move data to the correct device
    gnn_data = gnn_data.to(device)
    
    # Create the GNN model
    in_feats = gnn_data.num_node_features
    hidden_dim = gnn_config.GNN_HIDDEN_DIM
    model = StaticGCN(in_feats=in_feats, hidden_dim=hidden_dim)
    model = model.to(device)
    
    print(f"Training GNN model with {in_feats} input features and {hidden_dim} hidden dimensions")
    
    # Set up optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=gnn_config.LEARNING_RATE,
        weight_decay=gnn_config.WEIGHT_DECAY
    )
    
    # Set up loss function for binary classification
    loss_fn = torch.nn.BCEWithLogitsLoss()
    
    # Training loop
    model.train()
    for epoch in range(args.epochs):
        optimizer.zero_grad()
        
        # Forward pass
        logits = model(gnn_data)
        
        # Compute loss (only on nodes that have labels)
        loss = loss_fn(logits.squeeze(), gnn_data.y)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Print progress
        if (epoch + 1) % 5 == 0 or epoch == 0:
            with torch.no_grad():
                # Compute predictions
                pred = torch.sigmoid(logits.squeeze()) > 0.5
                correct = pred.eq(gnn_data.y.bool()).sum().item()
                acc = correct / gnn_data.y.size(0)
                
                print(f"Epoch {epoch+1}/{args.epochs}: Loss = {loss.item():.4f}, Acc = {acc:.4f}")
    
    # Save the model
    torch.save(model.state_dict(), args.model_output)
    print(f"Model saved to {args.model_output}")
    
    # Evaluation
    print("\n=== STAGE 3: GNN MODEL EVALUATION ===")
    evaluate_gnn(model, gnn_data)
    
    return model

def evaluate_gnn(model, gnn_data):
    """
    Evaluate the GNN model on the test data.
    
    Args:
        model: Trained GNN model
        gnn_data: PyTorch Geometric Data object
    """
    model.eval()
    with torch.no_grad():
        # Forward pass
        logits = model(gnn_data)
        probs = torch.sigmoid(logits.squeeze())
        preds = probs > 0.5
        
        # Compute accuracy
        correct = preds.eq(gnn_data.y.bool()).sum().item()
        acc = correct / gnn_data.y.size(0)
        
        print(f"Evaluation accuracy: {acc:.4f}")
        
        # Compute more detailed metrics if needed
        true_pos = ((preds == 1) & (gnn_data.y.bool() == 1)).sum().item()
        false_pos = ((preds == 1) & (gnn_data.y.bool() == 0)).sum().item()
        true_neg = ((preds == 0) & (gnn_data.y.bool() == 0)).sum().item()
        false_neg = ((preds == 0) & (gnn_data.y.bool() == 1)).sum().item()
        
        precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0
        recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        
        # Print confusion matrix
        print("\nConfusion Matrix:")
        print(f"True Positive: {true_pos}")
        print(f"False Positive: {false_pos}")
        print(f"True Negative: {true_neg}")
        print(f"False Negative: {false_neg}")
    
    return preds

def save_predictions_with_clusters(gnn_data, concept_clusters, output_path):
    """
    Save model predictions along with cluster information for interpretation.
    
    Args:
        gnn_data: PyTorch Geometric Data object with predictions
        concept_clusters: List of concept cluster dictionaries
        output_path: Path to save the output
    """
    # Get predictions
    model_path = gnn_config.MODEL_SAVE_PATH
    if os.path.exists(model_path):
        # Load the model
        device = torch.device(gnn_config.DEVICE)
        in_feats = gnn_data.num_node_features
        hidden_dim = gnn_config.GNN_HIDDEN_DIM
        model = StaticGCN(in_feats=in_feats, hidden_dim=hidden_dim)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model = model.to(device)
        
        # Get predictions
        model.eval()
        with torch.no_grad():
            logits = model(gnn_data)
            probs = torch.sigmoid(logits.squeeze())
            preds = probs > 0.5
            
            # Create output dict with predictions and clusters
            output_dict = {
                "clusters": []
            }
            
            for i, cluster in enumerate(concept_clusters):
                cluster_dict = {
                    "cluster_id": cluster["cluster_id"],
                    "representative_text": cluster["representative_text"],
                    "ground_truth": bool(cluster["not_understood"]),
                    "prediction": bool(preds[i].item()),
                    "probability": float(probs[i].item()),
                    "num_items": len(cluster["items"]),
                    "highlighted_items": sum(1 for item in cluster["items"] if item["is_highlighted"])
                }
                output_dict["clusters"].append(cluster_dict)
            
            # Save to JSON
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(output_dict, f, indent=2)
            
            print(f"Predictions saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the complete knowledge modeling pipeline with GNN")
    
    # Pipeline parameters
    parser.add_argument("--input", required=True, help="Path to input document (PDF or JSON)")
    parser.add_argument("--pipeline-output", default="pipeline_output.json", help="Path to save pipeline output")
    parser.add_argument("--clusters", type=int, default=10, help="Number of concept clusters to create")
    parser.add_argument("--similarity", type=float, default=0.7, help="Similarity threshold for edges")
    parser.add_argument("--embeddings", default="all-MiniLM-L6-v2", help="Embedding model to use")
    
    # GNN parameters
    parser.add_argument("--epochs", type=int, default=gnn_config.EPOCHS, help="Number of training epochs")
    parser.add_argument("--model-output", default=gnn_config.MODEL_SAVE_PATH, help="Path to save the trained model")
    parser.add_argument("--skip-training", action="store_true", help="Skip GNN training and only run pipeline")
    parser.add_argument("--predictions-output", default="predictions.json", help="Path to save the predictions")
    
    args = parser.parse_args()
    
    # Run the pipeline and GNN
    gnn_data, concept_clusters = run_complete_pipeline(args)
    
    # Save predictions with cluster information
    save_predictions_with_clusters(gnn_data, concept_clusters, args.predictions_output) 