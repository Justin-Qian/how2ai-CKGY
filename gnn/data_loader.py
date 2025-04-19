import json
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from torch_geometric.data import Data
import sys
import os

# Adjust path to import from the parent directory (doc_parsing)
# This assumes the script is run from the 'gnn' directory or the project root
# A more robust solution might involve setting PYTHONPATH or packaging
script_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(script_dir)
sys.path.append(parent_dir)

try:
    from doc_parsing.layoutlm.data_structures import ProcessedDocument, BoundingBox, TextBlock, Annotation
    import config # Use gnn.config
except ImportError as e:
    print(f"Error importing modules. Make sure paths are correct: {e}")
    print(f"Current sys.path: {sys.path}")
    # Fallback for running from project root?
    # from how2ai-CKGY.doc_parsing.layoutlm.data_structures import ProcessedDocument, BoundingBox, TextBlock, Annotation
    # import gnn.config as config
    # This part needs careful handling depending on execution context.
    # For now, assume running from within gnn/ or project root with adjusted sys.path.
    raise

def calculate_overlap_area(box1: BoundingBox, box2: BoundingBox) -> float:
    """Calculates the intersection area of two bounding boxes."""
    x_left = max(box1.x0, box2.x0)
    y_top = max(box1.y0, box2.y0)
    x_right = min(box1.x1, box2.x1)
    y_bottom = min(box1.y1, box2.y1)

    if x_right < x_left or y_bottom < y_top:
        return 0.0  # No overlap

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    return intersection_area

def get_box_area(box: BoundingBox) -> float:
    """Calculates the area of a bounding box."""
    return (box.x1 - box.x0) * (box.y1 - box.y0)

def load_graph_data() -> Data:
    """
    Loads processed document data, builds a graph, and returns a PyG Data object.
    """
    print(f"Loading processed data from: {config.PROCESSED_DATA_PATH}")
    try:
        with open(config.PROCESSED_DATA_PATH, 'r', encoding='utf-8') as f:
            data_dict = json.load(f)
        processed_doc = ProcessedDocument(**data_dict)
    except FileNotFoundError:
        print(f"Error: Processed document file not found at {config.PROCESSED_DATA_PATH}")
        raise
    except Exception as e:
        print(f"Error loading or parsing JSON: {e}")
        raise

    node_texts = []
    node_bboxes = []
    node_page_indices = [] # To map nodes back to pages for highlight association
    node_original_indices = {} # Map (page_num, block_idx_on_page) -> global_node_idx
    current_node_idx = 0

    print("Extracting text blocks as nodes...")
    for page in processed_doc.pages:
        for block_idx, block in enumerate(page.text_blocks):
            if block.text.strip(): # Only include blocks with actual text
                node_texts.append(block.text.strip())
                node_bboxes.append(block.bbox)
                node_page_indices.append(page.page_number)
                node_original_indices[(page.page_number, block_idx)] = current_node_idx
                current_node_idx += 1

    num_text_nodes = len(node_texts)
    if num_text_nodes == 0:
        raise ValueError("No text blocks found in the document. Cannot build graph.")
    print(f"Found {num_text_nodes} text nodes.")

    # Initialize labels (0: Understood, 1: Not Understood) based on text block type
    node_labels = torch.zeros(num_text_nodes, dtype=torch.float) # Use float for BCEWithLogitsLoss
    print("Step 1 & 2: Initializing node labels based on text block type...")

    not_understood_count = 0
    for page in processed_doc.pages:
        for block_idx, block in enumerate(page.text_blocks):
            original_key = (page.page_number, block_idx)
            if original_key in node_original_indices:
                global_node_idx = node_original_indices[original_key]

                # --- New Labeling Logic ---
                # Assume block has a 'type' attribute. Adjust key/values as needed.
                block_type = getattr(block, 'type', 'unknown').lower() # Safely get type, default to 'unknown'
                is_highlight_type = block_type in ["highlight", "annotation"] # Add other relevant types if necessary

                if is_highlight_type:
                    node_labels[global_node_idx] = 1.0 # Mark as Not Understood
                    not_understood_count += 1
                else:
                    node_labels[global_node_idx] = 0.0 # Mark as Understood (already default, but explicit)
                # --- End New Labeling Logic ---

    print(f"Step 1 & 2 Complete: Assigned labels based on block type. Marked {not_understood_count} nodes as 'Not Understood'.")

    # Compute Embeddings for all nodes
    print(f"Step 3: Loading sentence transformer model: {config.SENTENCE_TRANSFORMER_MODEL}...")
    model = SentenceTransformer(config.SENTENCE_TRANSFORMER_MODEL, device=config.DEVICE)
    print(f"Step 3 Complete: Sentence transformer model loaded on device {config.DEVICE}.")
    print("Step 4: Computing node embeddings (this may take time)...")
    with torch.no_grad():
        embeddings = model.encode(node_texts, convert_to_tensor=True, show_progress_bar=True, device=config.DEVICE)
    print(f"Step 4 Complete: Embeddings computed. Shape: {embeddings.shape}")

    # Build Graph Edges based on Cosine Similarity
    print("Step 5: Calculating cosine similarities (this may take time)...")
    similarities = cosine_similarity(embeddings.cpu().numpy()) # Cosine similarity on CPU
    print("Step 5 Complete: Cosine similarities calculated.")
    print("Step 6: Building graph edges...")
    edge_list = []
    edge_attr_list = []
    num_text_nodes = num_total_nodes # Renaming for clarity in loop below
    for i in range(num_text_nodes):
        for j in range(i + 1, num_text_nodes): # Avoid self-loops and duplicates
            sim = similarities[i, j]
            if sim >= config.SIMILARITY_THRESHOLD:
                edge_list.append([i, j])
                edge_list.append([j, i]) # Add edges in both directions for undirected graph
                edge_attr_list.append(sim)
                edge_attr_list.append(sim)

    if not edge_list:
        print("Warning: No edges created based on the similarity threshold. Graph will be disconnected.")
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 1), dtype=torch.float)
    else:
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr_list, dtype=torch.float).unsqueeze(1) # Add feature dimension
    print(f"Step 6 Complete: Graph edges created: {edge_index.shape[1]} edges.")

    # Add Global Node
    print("Step 7: Adding global node...")
    global_node_feature = embeddings.mean(dim=0, keepdim=True) # Mean of all text node features
    embeddings = torch.cat([embeddings, global_node_feature], dim=0)
    global_node_idx = num_text_nodes # Index of the new global node

    # Add edges connecting global node to all text nodes
    global_edges = []
    global_edge_attrs = []
    for i in range(num_text_nodes):
        global_edges.append([i, global_node_idx])
        global_edges.append([global_node_idx, i])
        # Use a default similarity (e.g., 1.0 or average similarity) for global node connections
        # Or calculate similarity between global node feature and each text node feature
        # For simplicity, let's use 1.0 for now
        global_edge_attrs.extend([1.0, 1.0])

    if global_edges:
        global_edge_index = torch.tensor(global_edges, dtype=torch.long).t().contiguous()
        global_edge_attr = torch.tensor(global_edge_attrs, dtype=torch.float).unsqueeze(1)

        edge_index = torch.cat([edge_index, global_edge_index], dim=1)
        edge_attr = torch.cat([edge_attr, global_edge_attr], dim=0)

    # Add a placeholder label for the global node (e.g., -1, will be ignored during training/loss calculation)
    global_node_label = torch.tensor([-1.0], dtype=torch.float)
    node_labels = torch.cat([node_labels, global_node_label], dim=0)

    num_total_nodes = embeddings.shape[0]
    print(f"Step 7 Complete: Global node added. Total nodes: {num_total_nodes}. Total edges: {edge_index.shape[1]}.")

    # Create PyG Data object
    print("Step 8: Creating PyG Data object...")
    data = Data(x=embeddings.to(config.DEVICE), # Ensure features are on the correct device
                edge_index=edge_index.to(config.DEVICE),
                edge_attr=edge_attr.to(config.DEVICE),
                y=node_labels.to(config.DEVICE)) # Ensure labels are on the correct device

    print("Step 8 Complete: PyG Data object created successfully.")
    return data

if __name__ == '__main__':
    # Example usage:
    try:
        graph_data = load_graph_data()
        print("\n--- Graph Data Summary ---")
        print(f"Number of nodes: {graph_data.num_nodes}")
        print(f"Number of edges: {graph_data.num_edges}")
        print(f"Number of node features: {graph_data.num_node_features}")
        print(f"Number of edge features: {graph_data.num_edge_features}")
        print(f"Node features shape (X): {graph_data.x.shape}")
        print(f"Edge index shape: {graph_data.edge_index.shape}")
        print(f"Edge attributes shape: {graph_data.edge_attr.shape}")
        print(f"Labels shape (Y): {graph_data.y.shape}")
        print(f"Contains isolated nodes: {graph_data.has_isolated_nodes()}")
        print(f"Contains self-loops: {graph_data.has_self_loops()}")
        print(f"Is undirected: {graph_data.is_undirected()}")
        print(f"Labels (first 10): {graph_data.y[:10]}")
        print(f"Label for global node: {graph_data.y[-1]}")
    except Exception as e:
        print(f"An error occurred during data loading: {e}")
