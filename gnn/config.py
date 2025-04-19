import torch

# --- Data Loading & Preprocessing ---
PROCESSED_DATA_PATH = "doc_parsing/layoutlm/output/processed_document.json" # Relative path from project root
HIGHLIGHT_OVERLAP_THRESHOLD = 0.5 # Minimum overlap area ratio for a text block to be considered highlighted
SIMILARITY_THRESHOLD = 0.7 # Cosine similarity threshold for creating graph edges
SENTENCE_TRANSFORMER_MODEL = 'all-MiniLM-L6-v2' # Model for generating text embeddings

# --- Model Configuration ---
GNN_HIDDEN_DIM = 128 # Hidden dimension size for GCN layers

# --- Training Configuration ---
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4 # L2 regularization factor
EPOCHS = 50 # Number of training epochs
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Evaluation ---
# (Add any evaluation-specific configs later if needed)

# --- Output ---
MODEL_SAVE_PATH = "gnn_model.pth" # Path to save the trained model
