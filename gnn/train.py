print("Starting train.py execution...") # Add this line
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.data import Data

print("Importing modules in train.py...") # Add this line
import config # GNN configurations
from data_loader import load_graph_data # Function to load graph data
from model import StaticGCN # The GCN model definition
print("Imports in train.py successful.") # Add this line

def train_model():
    """
    Loads data, trains the StaticGCN model, and saves the trained weights.
    """
    print("--- Starting GNN Training ---")
    device = torch.device(config.DEVICE)
    print(f"Using device: {device}")

    # 1. Load Data
    try:
        data = load_graph_data()
        print("Graph data loaded successfully.")
        # Ensure data is on the correct device (already handled in load_graph_data)
        # data = data.to(device)
    except Exception as e:
        print(f"Error loading graph data: {e}")
        return

    # 2. Instantiate Model
    input_dim = data.num_node_features
    hidden_dim = config.GNN_HIDDEN_DIM
    model = StaticGCN(in_feats=input_dim, hidden_dim=hidden_dim, out_feats=1).to(device)
    print("Model instantiated:")
    print(model)

    # 3. Define Loss and Optimizer
    # Use BCEWithLogitsLoss for binary classification (outputs logits)
    # We need to ignore the global node (label == -1) during loss calculation
    criterion = torch.nn.BCEWithLogitsLoss(reduction='none') # Calculate loss per node first
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    print(f"Optimizer: Adam (LR={config.LEARNING_RATE}, Weight Decay={config.WEIGHT_DECAY})")
    print(f"Loss Function: BCEWithLogitsLoss")

    # Create a mask for valid nodes (exclude global node with label -1)
    # Ensure labels are on the CPU for boolean indexing if needed, or use torch functions
    valid_node_mask = (data.y != -1).squeeze() # Squeeze to make it 1D boolean tensor
    if not valid_node_mask.any():
        print("Error: No valid nodes found for training (all labels are -1).")
        return
    print(f"Training on {valid_node_mask.sum().item()} valid nodes (excluding global node).")


    # 4. Training Loop
    print(f"Starting training for {config.EPOCHS} epochs...")
    model.train() # Set model to training mode
    for epoch in range(config.EPOCHS):
        optimizer.zero_grad() # Clear gradients

        # Forward pass
        out_logits = model(data) # Shape: [num_nodes, 1]

        # Calculate loss only on valid nodes
        raw_loss = criterion(out_logits, data.y.unsqueeze(1)) # Ensure target has same dims

        # Apply mask - select losses for valid nodes and compute mean
        masked_loss = raw_loss[valid_node_mask]
        if masked_loss.numel() > 0: # Check if there are any valid losses
             loss = masked_loss.mean()
        else:
             print(f"Warning: No valid nodes contributed to loss in epoch {epoch+1}. Skipping backward pass.")
             loss = torch.tensor(0.0, device=device, requires_grad=True) # Dummy loss if no valid nodes


        # Backward pass and optimization
        if masked_loss.numel() > 0:
            loss.backward()
            optimizer.step()

        # Print loss periodically
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f'Epoch {epoch+1:03d}/{config.EPOCHS} | Loss: {loss.item():.4f}')

    print("--- Training Finished ---")

    # 5. Save Model
    try:
        torch.save(model.state_dict(), config.MODEL_SAVE_PATH)
        print(f"Trained model saved to: {config.MODEL_SAVE_PATH}")
    except Exception as e:
        print(f"Error saving model: {e}")

if __name__ == "__main__":
    # Ensure necessary libraries (like sentence-transformers) are installed
    # Might need: pip install torch torch_geometric sentence-transformers scikit-learn
    train_model()
