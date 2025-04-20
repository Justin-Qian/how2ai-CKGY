from pydantic import BaseModel, Field
from typing import List, Optional, Tuple, Dict, Any

class BoundingBox(BaseModel):
    """Represents a bounding box with coordinates."""
    x0: float
    y0: float
    x1: float
    y1: float

class TextBlock(BaseModel):
    """Represents a block of text extracted from the document."""
    text: str
    bbox: BoundingBox
    # Optional: Placeholder for LayoutLM embeddings. Storing large embeddings directly
    # in JSON might be inefficient. Consider storing paths or omitting if not needed downstream.
    layoutlm_embedding: Optional[List[float]] = None # Or Optional[str] for path

class Annotation(BaseModel):
    """Represents an annotation (highlight, comment, etc.) in the document."""
    type: str # e.g., "highlight", "underline", "comment", "ink"
    bbox: BoundingBox
    text_content: Optional[str] = None # Text covered by the annotation
    comment_info: Optional[Dict[str, Any]] = None # e.g., {"author": "user", "comment": "text"}
    color: Optional[Dict[str, Tuple[float, float, float]]] = None # e.g., {'stroke': (r,g,b), 'fill': (r,g,b)} - colors normalized to 0-1 range
    vertices: Optional[List[List[Tuple[float, float]]]] = None # For 'ink' annotations (list of paths, each path is a list of (x,y) points)
    semantic_tag: Optional[str] = None # Interpreted meaning (e.g., 'important', 'question', 'definition')
    detected_color_name: Optional[str] = None # Name of color detected by CV (e.g., 'yellow')

class VisualElement(BaseModel):
    """Represents a visual element like a figure or table identified in the document."""
    type: str # e.g., "figure", "table", "chart", "handwriting"
    bbox: BoundingBox
    # Description or structured data extracted by VLM
    vlm_description: Optional[str] = None
    vlm_structured_data: Optional[Dict[str, Any]] = None
    # Optional: Reference to nearby text blocks
    associated_text_indices: Optional[List[int]] = None

class EquationElement(BaseModel):
    """Represents an equation identified in the document, potentially with VLM transcription."""
    type: str = "equation" # Fixed type
    bbox: BoundingBox
    vlm_transcription: Optional[str] = None # LaTeX or text transcription from VLM
    detection_source: Optional[str] = None # How it was detected (e.g., 'layout_heuristic', 'symbol_check')

class PageData(BaseModel):
    """Represents all extracted data for a single page."""
    page_number: int
    dimensions: Tuple[float, float] # width, height
    text_blocks: List[TextBlock] = Field(default_factory=list)
    annotations: List[Annotation] = Field(default_factory=list)
    visual_elements: List[VisualElement] = Field(default_factory=list)
    equations: List[EquationElement] = Field(default_factory=list)
    # Optional: Path to the preprocessed image for this page
    image_path: Optional[str] = None

class DocumentMetadata(BaseModel):
    """Metadata about the processed document."""
    filename: str
    total_pages: int
    processing_timestamp: Optional[str] = None # ISO format timestamp

class ProcessedDocument(BaseModel):
    """Root model for the final structured JSON output."""
    metadata: DocumentMetadata
    pages: List[PageData] = Field(default_factory=list)
