import torch
from sklearn.metrics import accuracy_score, roc_auc_score, precision_recall_fscore_support, confusion_matrix
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import config # GNN configurations
from data_loader import load_graph_data # Function to load graph data
from model import StaticGCN # The GCN model definition

def evaluate_model():
    """
    Loads the trained model and data, performs evaluation, and prints metrics.
    """
    print("--- Starting GNN Evaluation ---")
    device = torch.device(config.DEVICE)
    print(f"Using device: {device}")

    # 1. Load Data
    try:
        data = load_graph_data()
        print("Graph data loaded successfully.")
        # data = data.to(device) # Ensure data is on the correct device
    except Exception as e:
        print(f"Error loading graph data: {e}")
        return

    # 2. Instantiate Model and Load Weights
    input_dim = data.num_node_features
    hidden_dim = config.GNN_HIDDEN_DIM
    model = StaticGCN(in_feats=input_dim, hidden_dim=hidden_dim, out_feats=1).to(device)

    try:
        model.load_state_dict(torch.load(config.MODEL_SAVE_PATH, map_location=device))
        print(f"Model weights loaded from: {config.MODEL_SAVE_PATH}")
    except FileNotFoundError:
        print(f"Error: Model file not found at {config.MODEL_SAVE_PATH}. Please train the model first.")
        return
    except Exception as e:
        print(f"Error loading model weights: {e}")
        return

    model.eval() # Set model to evaluation mode

    # 3. Perform Inference
    with torch.no_grad():
        out_logits = model(data) # Get model predictions (logits)
        # Apply sigmoid to get probabilities
        out_probs = torch.sigmoid(out_logits).squeeze() # Shape: [num_nodes]
        # Get binary predictions (e.g., using 0.5 threshold)
        out_preds = (out_probs >= 0.5).float() # Shape: [num_nodes]

    # 4. Prepare Labels and Predictions for Evaluation (Exclude Global Node)
    valid_node_mask = (data.y != -1).squeeze().cpu().numpy() # Boolean mask on CPU
    true_labels = data.y.squeeze().cpu().numpy()[valid_node_mask]
    pred_labels = out_preds.cpu().numpy()[valid_node_mask]
    pred_probs = out_probs.cpu().numpy()[valid_node_mask] # Probabilities for ROC AUC

    if len(true_labels) == 0:
        print("Error: No valid nodes found for evaluation.")
        return

    print(f"Evaluating on {len(true_labels)} valid nodes.")

    # 5. Calculate Metrics
    accuracy = accuracy_score(true_labels, pred_labels)
    # Handle cases where only one class is present in true labels for ROC AUC
    if len(np.unique(true_labels)) > 1:
        roc_auc = roc_auc_score(true_labels, pred_probs)
    else:
        roc_auc = float('nan') # Not defined if only one class
        print("Warning: Only one class present in true labels. ROC AUC cannot be calculated.")

    # Precision, Recall, F1-score (binary classification, pos_label=1 for "Not Understood")
    precision, recall, f1, support = precision_recall_fscore_support(
        true_labels, pred_labels, average='binary', pos_label=1, zero_division=0
    )
    # Also get metrics per class
    precision_all, recall_all, f1_all, support_all = precision_recall_fscore_support(
        true_labels, pred_labels, average=None, labels=[0, 1], zero_division=0
    )

    print("\n--- Evaluation Results ---")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")
    print("\nMetrics for 'Not Understood' class (Positive Class, Label=1):")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    print(f"  Support:   {support_all[1] if support_all is not None else 'N/A'}") # Support for class 1

    print("\nMetrics for 'Understood' class (Negative Class, Label=0):")
    print(f"  Precision: {precision_all[0]:.4f}")
    print(f"  Recall:    {recall_all[0]:.4f}")
    print(f"  F1-Score:  {f1_all[0]:.4f}")
    print(f"  Support:   {support_all[0] if support_all is not None else 'N/A'}") # Support for class 0

    # 6. Confusion Matrix
    try:
        cm = confusion_matrix(true_labels, pred_labels, labels=[0, 1])
        print("\nConfusion Matrix:")
        print("             Predicted 0  Predicted 1")
        print(f"Actual 0    {cm[0, 0]:^10d}  {cm[0, 1]:^10d}")
        print(f"Actual 1    {cm[1, 0]:^10d}  {cm[1, 1]:^10d}")

        # Optional: Plot confusion matrix
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Understood (0)', 'Not Understood (1)'], yticklabels=['Understood (0)', 'Not Understood (1)'])
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        # Save or show the plot
        plt.savefig("confusion_matrix.png")
        print("\nConfusion matrix saved to confusion_matrix.png")
        # plt.show() # Uncomment to display plot interactively

    except Exception as e:
        print(f"\nCould not generate confusion matrix: {e}")


if __name__ == "__main__":
    # Ensure necessary libraries are installed
    # Might need: pip install torch torch_geometric sentence-transformers scikit-learn pandas seaborn matplotlib
    evaluate_model()
