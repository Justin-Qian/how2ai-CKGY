from pydantic import BaseModel, Field
from typing import List, Optional, Tuple, Dict, Any, Union

class BoundingBox(BaseModel):
    """Represents a bounding box with coordinates."""
    x0: float
    y0: float
    x1: float
    y1: float

class Document(BaseModel):
    """Represents the top-level document information."""
    id: str
    title: str
    sections: List[str] = Field(default_factory=list)
    character_start: int = 0
    character_end: int
    summary: Optional[str] = None

class Section(BaseModel):
    """Represents a logical section of the document."""
    id: str
    heading: str
    character_start: int
    character_end: int
    paragraphs: List[str] = Field(default_factory=list)

class Paragraph(BaseModel):
    """Represents a paragraph in the document."""
    id: str
    annotations: List[str] = Field(default_factory=list)
    text: str
    character_start: int
    character_end: int
    # Additional fields for internal processing
    bbox: Optional[BoundingBox] = None
    page_number: Optional[int] = None  # Store the page number explicitly

class Annotation(BaseModel):
    """Represents an annotation (highlight, comment, symbol, ink, visual_element, equation) in the document."""
    id: str
    type: str  # highlight, comment, symbol, ink, visual_element, equation
    color: str  # color name or hex code
    referenced_text: str  # text span the annotation refers to
    referenced_char_start: int
    referenced_char_end: int
    previous_text: str = ""  # context before reference
    posterior_text: str = ""  # context after reference
    paragraph_id: str
    annotated_text: str  # content of the annotation
    
    # Additional fields for internal processing
    bbox: Optional[BoundingBox] = None
    comment_info: Optional[Dict[str, Any]] = None
    vertices: Optional[List[List[Tuple[float, float]]]] = None
    semantic_tag: Optional[str] = None
    detected_color_name: Optional[str] = None
    color_info: Optional[Dict[str, Tuple[float, float, float]]] = None

class ProcessedDocument(BaseModel):
    """Root model for the structured JSON output."""
    document: Document
    sections: List[Section] = Field(default_factory=list)
    paragraphs: List[Paragraph] = Field(default_factory=list)
    annotations: List[Annotation] = Field(default_factory=list)
    
    # Additional fields for internal processing
    metadata: Optional[Dict[str, Any]] = None
    dimensions: Optional[Tuple[float, float]] = None  # width, height
    visual_elements: Optional[List[Dict[str, Any]]] = None
    equations: Optional[List[Dict[str, Any]]] = None

# Legacy models to maintain compatibility during transition
class TextBlock(BaseModel):
    """Represents a block of text extracted from the document."""
    text: str
    bbox: BoundingBox
    # Optional: Placeholder for LayoutLM embeddings. Storing large embeddings directly
    # in JSON might be inefficient. Consider storing paths or omitting if not needed downstream.
    layoutlm_embedding: Optional[List[float]] = None # Or Optional[str] for path

class LegacyAnnotation(BaseModel):
    """Legacy annotation model for compatibility."""
    type: str # e.g., "highlight", "underline", "comment", "ink"
    bbox: BoundingBox
    text_content: Optional[str] = None # Text covered by the annotation
    comment_info: Optional[Dict[str, Any]] = None # e.g., {"author": "user", "comment": "text"}
    color: Optional[Dict[str, Tuple[float, float, float]]] = None # e.g., {'stroke': (r,g,b), 'fill': (r,g,b)}
    vertices: Optional[List[List[Tuple[float, float]]]] = None # For 'ink' annotations
    semantic_tag: Optional[str] = None
    detected_color_name: Optional[str] = None

class VisualElement(BaseModel):
    """Represents a visual element like a figure or table identified in the document."""
    type: str # e.g., "figure", "table", "chart", "handwriting", "drawing"
    bbox: BoundingBox
    vlm_description: Optional[str] = None
    vlm_structured_data: Optional[Dict[str, Any]] = None
    associated_text_indices: Optional[List[int]] = None
    # Add visualization properties 
    visualization_color: Optional[str] = None

class EquationElement(BaseModel):
    """Represents an equation identified in the document, potentially with VLM transcription."""
    type: str = "equation" # Fixed type
    bbox: BoundingBox
    vlm_transcription: Optional[str] = None # LaTeX or text transcription from VLM
    detection_source: Optional[str] = None # How it was detected

class PageData(BaseModel):
    """Represents all extracted data for a single page."""
    page_number: int
    dimensions: Tuple[float, float] # width, height
    text_blocks: List[TextBlock] = Field(default_factory=list)
    annotations: List[LegacyAnnotation] = Field(default_factory=list)
    visual_elements: List[VisualElement] = Field(default_factory=list)
    equations: List[EquationElement] = Field(default_factory=list)
    image_path: Optional[str] = None

class DocumentMetadata(BaseModel):
    """Metadata about the processed document."""
    filename: str
    total_pages: int
    processing_timestamp: Optional[str] = None # ISO format timestamp

class LegacyProcessedDocument(BaseModel):
    """Legacy root model for compatibility."""
    metadata: DocumentMetadata
    pages: List[PageData] = Field(default_factory=list)
