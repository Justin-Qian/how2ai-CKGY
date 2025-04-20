import os
import json
import torch
import numpy as np
from typing import List, Dict, Any, Tuple
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
from torch_geometric.data import Data

# Import from doc_parsing module
from doc_parsing.layoutlm.document_processor import process_document, save_processed_document
from doc_parsing.layoutlm.data_structures import (
    ProcessedDocument, Annotation, TextBlock, BoundingBox
)

# Import from gnn module
import gnn.config as gnn_config


class KnowledgeModelingPipeline:
    """
    Main pipeline that connects document processing with GNN input preparation.
    This implements the core pipeline: Document -> Know/Not Know Highlights -> Clustering -> GNN Input
    """
    
    def __init__(self, 
                 embedding_model_name: str = 'all-MiniLM-L6-v2',
                 similarity_threshold: float = 0.7,
                 num_clusters: int = 10,
                 device: str = None):
        """
        Initialize the pipeline with configuration parameters.
        
        Args:
            embedding_model_name: Name of the SentenceTransformer model to use
            similarity_threshold: Threshold for creating edges between nodes
            num_clusters: Number of concept clusters to create (approximate)
            device: Device to use for computations ('cpu' or 'cuda')
        """
        self.embedding_model_name = embedding_model_name
        self.similarity_threshold = similarity_threshold
        self.num_clusters = num_clusters
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"Initializing pipeline with embedding model: {embedding_model_name} on {self.device}")
        self.embedding_model = SentenceTransformer(embedding_model_name, device=self.device)
    
    def process_document(self, pdf_path: str, output_dir: str = "doc_parsing/layoutlm/output") -> ProcessedDocument:
        """
        Process the input document using LayoutLM document processor.
        
        Args:
            pdf_path: Path to the PDF document to process
            output_dir: Directory to save the processed document JSON
            
        Returns:
            ProcessedDocument object with the processed document data
        """
        print(f"Processing document: {pdf_path}")
        processed_doc = process_document(pdf_path)
        
        # Save the processed document
        filename = os.path.basename(pdf_path).replace('.pdf', '_processed.json')
        save_processed_document(processed_doc, output_dir, filename)
        
        return processed_doc
    
    def load_processed_document(self, json_path: str) -> ProcessedDocument:
        """
        Load a previously processed document from JSON.
        
        Args:
            json_path: Path to the processed document JSON file
            
        Returns:
            ProcessedDocument object with the loaded document data
        """
        print(f"Loading processed document from: {json_path}")
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data_dict = json.load(f)
            return ProcessedDocument(**data_dict)
        except Exception as e:
            print(f"Error loading processed document: {e}")
            raise
    
    def _extract_text_and_highlights(self, processed_doc: ProcessedDocument) -> Tuple[List[Dict], List[str]]:
        """
        Extract text blocks and determine their highlight status.
        
        Args:
            processed_doc: ProcessedDocument object
            
        Returns:
            Tuple of (text_items, text_for_embedding)
            - text_items: List of dicts with text info including highlight status
            - text_for_embedding: List of text strings for embedding
        """
        print("Extracting text blocks and highlights...")
        text_items = []
        text_for_embedding = []
        
        for page in processed_doc.pages:
            page_num = page.page_number
            
            # Process all text blocks
            for idx, block in enumerate(page.text_blocks):
                if not block.text.strip():
                    continue
                
                # Initialize as "understood" (not highlighted)
                is_highlighted = False
                highlight_info = None
                
                # Check for overlapping highlights/annotations
                for annot in page.annotations:
                    if annot.type.lower() in ["highlight", "underline"]:
                        # Calculate overlap between text block and highlight
                        overlap = self._calculate_overlap(block.bbox, annot.bbox)
                        if overlap > 0.5:  # If significant overlap
                            is_highlighted = True
                            highlight_info = {
                                "type": annot.type,
                                "text": annot.text_content,
                                "comment": annot.comment_info
                            }
                            break
                
                # Create the text item
                text_item = {
                    "page_num": page_num,
                    "block_idx": idx,
                    "text": block.text.strip(),
                    "bbox": {
                        "x0": block.bbox.x0,
                        "y0": block.bbox.y0,
                        "x1": block.bbox.x1,
                        "y1": block.bbox.y1
                    },
                    "is_highlighted": is_highlighted,  # True for "not understood"
                    "highlight_info": highlight_info
                }
                
                text_items.append(text_item)
                text_for_embedding.append(block.text.strip())
        
        print(f"Extracted {len(text_items)} text blocks, {sum(1 for item in text_items if item['is_highlighted'])} highlighted")
        return text_items, text_for_embedding
    
    def _calculate_overlap(self, box1: BoundingBox, box2: BoundingBox) -> float:
        """Calculate overlap ratio between two bounding boxes."""
        x_left = max(box1.x0, box2.x0)
        y_top = max(box1.y0, box2.y0)
        x_right = min(box1.x1, box2.x1)
        y_bottom = min(box1.y1, box2.y1)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0  # No overlap
        
        intersection = (x_right - x_left) * (y_bottom - y_top)
        box1_area = (box1.x1 - box1.x0) * (box1.y1 - box1.y0)
        
        return intersection / box1_area
    
    def compute_embeddings(self, texts: List[str]) -> torch.Tensor:
        """
        Compute embeddings for a list of text strings.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            Tensor of embeddings
        """
        print(f"Computing embeddings for {len(texts)} text items...")
        with torch.no_grad():
            embeddings = self.embedding_model.encode(
                texts,
                convert_to_tensor=True,
                show_progress_bar=True,
                device=self.device
            )
        return embeddings
    
    def cluster_concepts(self, 
                         embeddings: torch.Tensor, 
                         text_items: List[Dict]) -> Tuple[List[Dict], np.ndarray]:
        """
        Cluster text items into concept clusters based on their embeddings.
        
        Args:
            embeddings: Tensor of text embeddings
            text_items: List of dicts with text info including highlight status
            
        Returns:
            Tuple of (concept_clusters, labels)
            - concept_clusters: List of dicts with cluster info
            - labels: Numpy array of cluster labels
        """
        print(f"Clustering {len(text_items)} text items into approximately {self.num_clusters} clusters...")
        
        # Convert embeddings to numpy for clustering
        embeddings_np = embeddings.cpu().numpy()
        
        # Use Agglomerative Clustering with cosine distance
        clustering = AgglomerativeClustering(
            n_clusters=min(self.num_clusters, len(text_items)),
            metric='precomputed',
            linkage='average'
        )
        
        # Compute distance matrix (1 - cosine similarity)
        similarity_matrix = cosine_similarity(embeddings_np)
        distance_matrix = 1 - similarity_matrix
        
        # Fit clustering
        labels = clustering.fit_predict(distance_matrix)
        unique_clusters = np.unique(labels)
        print(f"Created {len(unique_clusters)} concept clusters")
        
        # Organize text items by cluster
        concept_clusters = []
        for cluster_id in unique_clusters:
            cluster_indices = np.where(labels == cluster_id)[0]
            cluster_items = [text_items[i] for i in cluster_indices]
            
            # Determine if cluster is "not understood" based on highlighted items
            not_understood = any(item["is_highlighted"] for item in cluster_items)
            
            # Find most representative item as cluster label (closest to centroid)
            cluster_embeddings = embeddings_np[cluster_indices]
            centroid = np.mean(cluster_embeddings, axis=0)
            distances_to_centroid = np.array([
                np.linalg.norm(emb - centroid) for emb in cluster_embeddings
            ])
            representative_idx = cluster_indices[np.argmin(distances_to_centroid)]
            representative_text = text_items[representative_idx]["text"]
            
            # Create cluster dict
            cluster = {
                "cluster_id": int(cluster_id),
                "not_understood": not_understood,
                "representative_text": representative_text,
                "items": cluster_items,
                "item_indices": cluster_indices.tolist(),
                "embedding": centroid.tolist()  # Store centroid as cluster embedding
            }
            concept_clusters.append(cluster)
        
        return concept_clusters, labels
    
    def prepare_gnn_input(self, 
                          concept_clusters: List[Dict], 
                          embeddings: torch.Tensor, 
                          similarity_threshold: float = None) -> Data:
        """
        Prepare a PyTorch Geometric Data object for GNN input.
        
        Args:
            concept_clusters: List of dicts with cluster info
            embeddings: Tensor of text embeddings
            similarity_threshold: Similarity threshold for edge creation (optional)
            
        Returns:
            PyTorch Geometric Data object
        """
        print("Preparing GNN input...")
        if similarity_threshold is None:
            similarity_threshold = self.similarity_threshold
        
        # Create node features from cluster centroids
        num_clusters = len(concept_clusters)
        node_embeddings = torch.tensor(
            [cluster["embedding"] for cluster in concept_clusters],
            dtype=torch.float32,
            device=self.device
        )
        
        # Create node labels (0: understood, 1: not understood)
        node_labels = torch.tensor(
            [1.0 if cluster["not_understood"] else 0.0 for cluster in concept_clusters],
            dtype=torch.float32,
            device=self.device
        )
        
        # Calculate similarities between clusters
        embeddings_np = node_embeddings.cpu().numpy()
        similarities = cosine_similarity(embeddings_np)
        
        # Create edges based on similarity threshold
        edge_list = []
        edge_attr_list = []
        for i in range(num_clusters):
            for j in range(i + 1, num_clusters):
                sim = similarities[i, j]
                if sim >= similarity_threshold:
                    edge_list.append([i, j])
                    edge_list.append([j, i])  # Add edges in both directions
                    edge_attr_list.append(sim)
                    edge_attr_list.append(sim)
        
        # Convert to PyG format
        if not edge_list:
            print("Warning: No edges created based on similarity threshold")
            edge_index = torch.empty((2, 0), dtype=torch.long, device=self.device)
            edge_attr = torch.empty((0, 1), dtype=torch.float, device=self.device)
        else:
            edge_index = torch.tensor(edge_list, dtype=torch.long, device=self.device).t().contiguous()
            edge_attr = torch.tensor(edge_attr_list, dtype=torch.float, device=self.device).unsqueeze(1)
        
        print(f"Created graph with {num_clusters} nodes and {edge_index.shape[1]} edges")
        
        # Create PyG Data object
        data = Data(
            x=node_embeddings,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=node_labels
        )
        
        return data
    
    def save_pipeline_output(self, 
                            text_items: List[Dict], 
                            concept_clusters: List[Dict], 
                            gnn_input: Data, 
                            output_path: str):
        """
        Save pipeline output to JSON for future use.
        
        Args:
            text_items: List of dicts with text info
            concept_clusters: List of dicts with cluster info
            gnn_input: PyTorch Geometric Data object
            output_path: Path to save the output JSON
        """
        print(f"Saving pipeline output to {output_path}")
        
        # Convert PyG Data to serializable format
        gnn_data_dict = {
            "num_nodes": gnn_input.num_nodes,
            "num_edges": gnn_input.num_edges,
            "node_features_shape": list(gnn_input.x.shape),
            "edge_index_shape": list(gnn_input.edge_index.shape),
            "edge_attr_shape": list(gnn_input.edge_attr.shape) if gnn_input.edge_attr is not None else None,
            "labels_shape": list(gnn_input.y.shape),
            "has_isolated_nodes": gnn_input.has_isolated_nodes(),
            "has_self_loops": gnn_input.has_self_loops(),
            "is_undirected": gnn_input.is_undirected()
        }
        
        # Create output dict
        output_dict = {
            "text_items": text_items,
            "concept_clusters": concept_clusters,
            "gnn_data": gnn_data_dict
        }
        
        # Save to JSON
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_dict, f, indent=2)
    
    def run_pipeline(self, input_path: str, output_path: str = "pipeline_output.json"):
        """
        Run the complete pipeline: Document -> Highlights -> Clustering -> GNN Input.
        
        Args:
            input_path: Path to the input document (PDF or JSON)
            output_path: Path to save the pipeline output
            
        Returns:
            Tuple of (processed_doc, text_items, concept_clusters, gnn_input)
        """
        # Step 1: Process document or load processed document
        if input_path.lower().endswith('.pdf'):
            processed_doc = self.process_document(input_path)
        else:
            processed_doc = self.load_processed_document(input_path)
        
        # Step 2: Extract text and highlights
        text_items, texts_for_embedding = self._extract_text_and_highlights(processed_doc)
        
        # Step 3: Compute embeddings
        embeddings = self.compute_embeddings(texts_for_embedding)
        
        # Step 4: Cluster concepts
        concept_clusters, cluster_labels = self.cluster_concepts(embeddings, text_items)
        
        # Step 5: Prepare GNN input
        gnn_input = self.prepare_gnn_input(concept_clusters, embeddings)
        
        # Step 6: Save output
        self.save_pipeline_output(text_items, concept_clusters, gnn_input, output_path)
        
        return processed_doc, text_items, concept_clusters, gnn_input


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run knowledge modeling pipeline")
    parser.add_argument("--input", required=True, help="Path to input document (PDF or JSON)")
    parser.add_argument("--output", default="pipeline_output.json", help="Path to save pipeline output")
    parser.add_argument("--clusters", type=int, default=10, help="Number of concept clusters to create")
    parser.add_argument("--similarity", type=float, default=0.7, help="Similarity threshold for edges")
    parser.add_argument("--embeddings", default="all-MiniLM-L6-v2", help="Embedding model to use")
    
    args = parser.parse_args()
    
    # Initialize and run pipeline
    pipeline = KnowledgeModelingPipeline(
        embedding_model_name=args.embeddings,
        similarity_threshold=args.similarity,
        num_clusters=args.clusters
    )
    
    _, _, _, gnn_input = pipeline.run_pipeline(args.input, args.output)
    
    print("\n--- Pipeline Complete ---")
    print(f"Created GNN input with {gnn_input.num_nodes} nodes and {gnn_input.num_edges} edges")
    print(f"Output saved to {args.output}") 