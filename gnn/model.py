import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data

import config # Import GNN configurations

class StaticGCN(torch.nn.Module):
    """
    A simple static Graph Convolutional Network (GCN) model for node classification.
    Outputs logits for binary classification.
    """
    def __init__(self, in_feats: int, hidden_dim: int, out_feats: int = 1):
        """
        Initializes the StaticGCN model.

        Args:
            in_feats (int): Dimensionality of input node features.
            hidden_dim (int): Dimensionality of the hidden layer.
            out_feats (int): Dimensionality of the output (default: 1 for binary logits).
        """
        super().__init__()
        self.conv1 = GCNConv(in_feats, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, out_feats)
        # Optional: Add dropout for regularization
        # self.dropout_prob = 0.5 # Example dropout probability

    def forward(self, data: Data) -> torch.Tensor:
        """
        Performs the forward pass of the GCN model.

        Args:
            data (Data): PyTorch Geometric Data object containing node features (x)
                         and edge index (edge_index).

        Returns:
            torch.Tensor: Output logits for each node. Shape: [num_nodes, out_feats]
        """
        x, edge_index = data.x, data.edge_index

        # First GCN layer + ReLU activation
        x = self.conv1(x, edge_index)
        x = F.relu(x)

        # Optional: Apply dropout
        # x = F.dropout(x, p=self.dropout_prob, training=self.training)

        # Second GCN layer (output logits)
        x = self.conv2(x, edge_index)

        return x

if __name__ == '__main__':
    # Example usage:
    # Assume we have loaded data using data_loader
    # from data_loader import load_graph_data
    # graph_data = load_graph_data()

    # Create dummy data for demonstration if data_loader is not run
    num_nodes = 10
    num_features = 384 # Example embedding dim from 'all-MiniLM-L6-v2'
    dummy_x = torch.randn(num_nodes, num_features)
    dummy_edge_index = torch.randint(0, num_nodes, (2, 20), dtype=torch.long)
    dummy_data = Data(x=dummy_x, edge_index=dummy_edge_index)

    # Get input feature dimension from data
    input_dim = dummy_data.num_node_features
    hidden_dimension = config.GNN_HIDDEN_DIM

    # Instantiate the model
    model = StaticGCN(in_feats=input_dim, hidden_dim=hidden_dimension)
    print("Model Instantiated:")
    print(model)

    # Perform a forward pass
    model.eval() # Set to evaluation mode
    with torch.no_grad():
        output_logits = model(dummy_data)

    print(f"\nInput node features shape: {dummy_data.x.shape}")
    print(f"Output logits shape: {output_logits.shape}")
    print(f"Output logits (first 5 nodes):\n{output_logits[:5]}")

    # Check output for the global node (if included in dummy data)
    # print(f"Output logit for global node (last node): {output_logits[-1]}")
