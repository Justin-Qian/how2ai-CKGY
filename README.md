# how2ai-CKGY

## Knowledge Modeling Pipeline Overview

Integrates document processing with concept clustering and graph neural networks (GNN) to model knowledge states based on highlighted content in documents.

The pipeline processes documents to extract text and highlights, clusters related concepts, and uses a GNN to predict whether a concept cluster is understood or not understood by a learner.

### Core Components

1. **Document Processing** (`doc_parsing/layoutlm/`)
   - Extracts text, structure, and annotations from documents
   - Processes highlights to determine "understood" vs "not understood" content

2. **Knowledge Modeling Pipeline** (`knowledge_pipeline.py`)
   - Extracts text and highlight information
   - Computes semantic embeddings using Sentence Transformers
   - Clusters related concepts using hierarchical clustering
   - Prepares input for the GNN

3. **Graph Neural Network** (`gnn/`)
   - Models relationships between concept clusters
   - Predicts knowledge state (understood/not understood) for concepts

## Getting Started

### Prerequisites

Install the required dependencies:

```bash
pip install -r requirements.txt
```

### Running the Pipeline

To run the complete pipeline:

```bash
python pipeline_runner.py --input /path/to/document.pdf
```

#### Command Line Arguments

- `--input`: Path to input document (PDF or JSON)
- `--pipeline-output`: Path to save pipeline output (default: pipeline_output.json)
- `--clusters`: Number of concept clusters (default: 10)
- `--similarity`: Similarity threshold for edges (default: 0.7)
- `--embeddings`: Embedding model to use (default: all-MiniLM-L6-v2)
- `--epochs`: Number of training epochs (default: 50)
- `--model-output`: Path to save the trained model (default: gnn_model.pth)
- `--skip-training`: Skip GNN training and only run pipeline
- `--predictions-output`: Path to save predictions (default: predictions.json)

### Pipeline Steps

1. **Document Processing**
   - Process the PDF document to extract text blocks and highlights
   - Highlight information determines what content is "understood" vs "not understood"

2. **Concept Clustering**
   - Create semantic embeddings for each text block
   - Cluster text blocks based on semantic similarity
   - Determine concept cluster status based on contained highlights

3. **GNN Preparation**
   - Transform concept clusters into graph nodes
   - Create edges between related concepts
   - Define node features and labels

4. **GNN Training and Evaluation**
   - Train the GNN to predict understanding state of concept clusters
   - Evaluate model performance
   - Generate predictions for interpretation

## Output Files

- `pipeline_output.json`: Contains text items, concept clusters, and GNN data structure
- `gnn_model.pth`: Trained GNN model
- `predictions.json`: Model predictions with cluster information

## Understanding the Results

The pipeline provides interpretable concept clusters with:
- Representative text for each cluster
- Understanding status based on highlights
- GNN predictions for each concept cluster
- Related text items within each cluster

---

## Example Usage

Process a document with highlights:

```bash
python pipeline_runner.py --input doc_parsing/layoutlm/Input_files/sample_doc.pdf --clusters 15
```

Process a previously parsed document:

```bash
python pipeline_runner.py --input doc_parsing/layoutlm/output/processed_document.json
```