import os
import sys
import json
import argparse
import asyncio
import numpy as np
import re
from datetime import datetime
import uuid
from PIL import Image
import fitz  # PyMuPDF
from typing import List, Dict, Tuple, Optional, Union

try:
    import config
except ImportError:
    # Try relative import as fallback
    try:
        from . import config
    except ImportError:
        print("Warning: Could not import config module. Using default settings.")
        # Create a minimal config
        class DefaultConfig:
            def __init__(self):
                self.OUTPUT_DIR = "output"
                self.CV_HIGHLIGHT_HSV_RANGES = {}
                self.JSON_INDENT = 2
                self.OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
                
        config = DefaultConfig()

from .cv_utils import detect_color_highlight_regions, detect_ink_regions
from .vlm_utils import analyze_image_region_with_vlm
from .layoutlm_utils import extract_layoutlm_features
from .data_structures import (
    BoundingBox, TextBlock, LegacyAnnotation, VisualElement, EquationElement, PageData,
    DocumentMetadata, LegacyProcessedDocument, Document, Section, Paragraph, 
    Annotation, ProcessedDocument
)

# --- Document Section Detection Functions ---

def is_citation_section(text: str) -> bool:
    """
    Detect if a section appears to be citations/references.
    
    Args:
        text: The section heading or text block to analyze
        
    Returns:
        True if the text indicates a citation/references section
    """
    citation_indicators = [
        "reference", "references", "bibliography", "works cited", 
        "literature cited", "cited literature"
    ]
    return any(indicator in text.lower() for indicator in citation_indicators)

def is_appendix_section(text: str) -> bool:
    """
    Detect if a section is an appendix.
    
    Args:
        text: The section heading or text block to analyze
        
    Returns:
        True if the text indicates an appendix section
    """
    appendix_indicators = [
        "appendix", "appendices", "supplementary", "supplemental", 
        "additional material"
    ]
    return any(indicator in text.lower() for indicator in appendix_indicators)

def is_summary_section(text: str, color_info=None) -> bool:
    """
    Detect if a section is a summary section, including color indicators.
    
    Args:
        text: The section heading or text block to analyze
        color_info: Optional color information if available
        
    Returns:
        True if the text indicates a summary section
    """
    summary_indicators = ["summary"]
    
    # Check text indicators
    text_match = any(indicator in text.lower() for indicator in summary_indicators)
    
    # Check if text appears to be in red (if color info available)
    is_red_text = False
    if color_info and 'stroke' in color_info:
        r, g, b = color_info['stroke']
        # Simple red detection (high R, low G and B)
        if r > 0.7 and g < 0.4 and b < 0.4:
            is_red_text = True
    
    return text_match or is_red_text

# --- Utility Functions ---

def pdf_coords_to_bbox(rect: fitz.Rect) -> BoundingBox:
    """Converts PyMuPDF Rect coordinates to our BoundingBox model."""
    return BoundingBox(x0=rect.x0, y0=rect.y0, x1=rect.x1, y1=rect.y1)

def is_extreme_aspect_ratio(rect: fitz.Rect, max_ratio: float = 10.0) -> bool:
    """Check if a rectangle is likely too thin or wide (potentially decorative)."""
    width = rect.width
    height = rect.height
    if width <= 0 or height <= 0:
        return True # Invalid rect
    ratio = max(width / height, height / width)
    return ratio > max_ratio

def is_rect_overlap(bbox1: BoundingBox, bbox2: BoundingBox) -> bool:
    """Check if two bounding boxes overlap."""
    # Check if one rectangle is to the left of the other
    if bbox1.x1 < bbox2.x0 or bbox2.x1 < bbox1.x0:
        return False
    
    # Check if one rectangle is above the other
    if bbox1.y1 < bbox2.y0 or bbox2.y1 < bbox1.y0:
        return False
    
    # If we get here, the rectangles overlap
    return True

def interpret_annotation_semantics(annotation_data: LegacyAnnotation) -> Optional[str]:
    """Interprets the semantic meaning of an annotation based on rules in config."""
    semantic_map = getattr(config, 'ANNOTATION_SEMANTIC_MAP', {})
    if not semantic_map:
        return None

    tag = None

    # 1. Check by color (priority: stroke, then fill)
    if annotation_data.color:
        color_tuple = None
        if 'stroke' in annotation_data.color and annotation_data.color['stroke']:
            color_tuple = annotation_data.color['stroke']
        elif 'fill' in annotation_data.color and annotation_data.color['fill']:
             color_tuple = annotation_data.color['fill']
        
        if color_tuple:
            # Normalize/round color tuple for matching keys in the map (adjust precision as needed)
            rounded_color = tuple(round(c, 2) for c in color_tuple)
            tag = semantic_map.get(rounded_color)
            if tag: return tag # Return if color match found

    # 2. Check by type (if no color match)
    tag = semantic_map.get(annotation_data.type)
    if tag: return tag

    # 3. Check by content (e.g., for comments or specific symbols in text)
    # Example: Check comment content for keywords
    if annotation_data.type == 'comment' and annotation_data.comment_info and 'content' in annotation_data.comment_info:
        comment_text = annotation_data.comment_info['content'].lower()
        if 'question:' in comment_text or '?' in comment_text:
            return semantic_map.get("comment_question", "question") # Use specific tag or fallback
        # Add more keyword checks based on config or hardcoded rules
    
    # Example: Check associated text for symbols (like **)
    if annotation_data.text_content:
        if annotation_data.text_content.strip().startswith("**") and annotation_data.text_content.strip().endswith("**"):
             return semantic_map.get("double_asterisk", "very_important")
        if annotation_data.text_content.strip().startswith("*") and annotation_data.text_content.strip().endswith("*"):
             return semantic_map.get("single_asterisk", "important")

    # Add more complex rules as needed

    return tag # Return None if no rules matched

def identify_complex_regions(page: fitz.Page, text_blocks: list) -> List[VisualElement]:
    """
    Identifies regions potentially needing VLM analysis (images, drawings).
    Improved with drawing detection and basic overlap checks.
    """
    identified_elements = []
    page_rect = page.rect
    page_width, page_height = page_rect.width, page_rect.height

    # Use a set to keep track of areas already covered to avoid duplicates
    covered_rects = []

    # 1. Detect significant vector drawings
    drawings = page.get_drawings()
    # TODO: More sophisticated grouping of drawing paths might be needed.
    # For now, treat each significant drawing object's bbox as a potential region.
    for drawing in drawings:
        # 'items' contains drawing commands, 'rect' is the bounding box
        drawing_rect = drawing['rect']
        drawing_area = drawing_rect.width * drawing_rect.height
        
        # Check area, aspect ratio, and if it likely contains non-trivial paths
        if (drawing_area > getattr(config, 'MIN_DRAWING_AREA_FOR_VLM', 5000) and 
            not is_extreme_aspect_ratio(drawing_rect, getattr(config, 'MAX_DRAWING_ASPECT_RATIO', 15)) and
            len(drawing.get('items', [])) > 1): # Ensure it's not just a simple line
            
            element = VisualElement(
                type="drawing", 
                bbox=pdf_coords_to_bbox(drawing_rect)
            )
            identified_elements.append(element)
            covered_rects.append(drawing_rect)

    # 2. Detect significant images, avoiding overlap with drawings
    image_list = page.get_image_info(xrefs=True)
    for img_info in image_list:
        img_bbox_rect = fitz.Rect(img_info['bbox'])
        img_area = img_bbox_rect.width * img_bbox_rect.height

        # Check area and aspect ratio
        if (img_area > getattr(config, 'MIN_IMAGE_AREA_FOR_VLM', 10000) and 
            not is_extreme_aspect_ratio(img_bbox_rect, getattr(config, 'MAX_IMAGE_ASPECT_RATIO', 15))):
            
            # Check for significant overlap with already identified drawings
            is_overlapping_drawing = False
            for dr in covered_rects:
                intersect_area = dr.intersect(img_bbox_rect).get_area()
                if intersect_area > 0.5 * img_area or intersect_area > 0.5 * dr.get_area():
                    is_overlapping_drawing = True
                    break
            
            if not is_overlapping_drawing:
                # Optional: Check for overlap with text blocks (simple version)
                # is_overlapping_text = False
                # for tb in text_blocks:
                #     tb_rect = fitz.Rect(tb.bbox.x0, tb.bbox.y0, tb.bbox.x1, tb.bbox.y1)
                #     if tb_rect.intersects(img_bbox_rect):
                #         intersect_area = tb_rect.intersect(img_bbox_rect).get_area()
                #         # Allow minor overlap, threshold might need tuning
                #         if intersect_area > 0.1 * img_area:
                #             is_overlapping_text = True
                #             break
                # if not is_overlapping_text:
                    
                element = VisualElement(
                     type="figure", # Assume figure for images
                     bbox=pdf_coords_to_bbox(img_bbox_rect)
                 )
                identified_elements.append(element)
                covered_rects.append(img_bbox_rect) # Add image rect to covered areas

    # 3. Optional: Further refinement - merge overlapping/nearby regions if needed
    # ... implementation could go here ...

    # 4. Optional: Use LayoutLM output if it classifies figures/tables
    # ... integration logic needed ...

    print(f"  Identified {len(identified_elements)} visual regions (drawings/images).")
    return identified_elements

def identify_equation_regions(text_blocks: List[TextBlock], page_dims: Tuple[float, float]) -> List[EquationElement]:
    """Identifies potential equation regions based on layout and content heuristics (v3 - Stricter)."""
    equations = []
    page_width, page_height = page_dims
    center_margin = page_width * 0.20
    min_math_char_ratio = 0.25 # Increased significantly
    required_symbols = ['=', '<', '>', '\u2264', '\u2265'] # <=, >= unicode
    exclude_keywords = ['table', 'figure', 'fig.', ' et al', ' appendix', ' acknowledgments', ' references']
    
    # Expanded math pattern
    math_chars_pattern = re.compile(r'[=+\-*/<>\(\)\[\]\{\}\|^\d\.,_θΣσμπ∈∀∃∞∫∂∇\u2264\u2265]|[α-ωΑ-Ω]|\b(sin|cos|tan|log|exp|sqrt|sum|lim)\b') 
    equation_num_pattern = re.compile(r'\(\s*([A-Z]?\d+[a-z]?|[A-Z])\s*\)$|\[\s*(\d+)\s*\]$')

    if not text_blocks: return []

    potential_equations = []
    last_block_y1 = 0

    for i, block in enumerate(text_blocks):
        rect = fitz.Rect(block.bbox.x0, block.bbox.y0, block.bbox.x1, block.bbox.y1)
        block_text = block.text.strip()
        block_text_lower = block_text.lower()
        if not block_text: continue
        
        # Exclude based on keywords
        if any(keyword in block_text_lower for keyword in exclude_keywords):
            last_block_y1 = rect.y1
            continue

        # Content Check
        math_chars_count = len(math_chars_pattern.findall(block_text))
        text_len = len(block_text.replace(" ", ""))
        math_ratio = math_chars_count / text_len if text_len > 0 else 0
        has_required_symbol = any(sym in block_text for sym in required_symbols)
        
        # Check for equation number 
        has_equation_num = equation_num_pattern.search(block_text) is not None
        
        # Layout Check: Centering & Vertical spacing (logic unchanged)
        is_centered = abs((rect.x0 + rect.x1) / 2 - page_width / 2) < center_margin
        space_above = rect.y0 - last_block_y1 if i > 0 else rect.y0
        space_below = text_blocks[i+1].bbox.y0 - rect.y1 if i < len(text_blocks) - 1 else page_height - rect.y1
        min_space = rect.height * 0.4 
        has_vert_space = space_above > min_space and space_below > min_space

        # Combine Heuristics (v3 - Stricter)
        is_potential = False
        # Rule 1: Must have equation number OR a required symbol (like '=')
        if has_equation_num or has_required_symbol:
            # AND requires decent math ratio OR good layout cues
            if math_ratio > min_math_char_ratio or (is_centered and has_vert_space):
                 is_potential = True
        # Rule 2: Maybe allow very high math ratio even without =/num, if layout is good?
        elif math_ratio > 0.4 and (is_centered or has_vert_space):
             is_potential = True 
            
        if is_potential:
            potential_equations.append({"index": i, "block": block, "rect": rect})
        
        last_block_y1 = rect.y1

    # Merge consecutive potential equation blocks
    if not potential_equations: return []

    merged_equations = []
    current_merge = None

    for i, pot_eq in enumerate(potential_equations):
        block_index = pot_eq["index"]
        block_rect = pot_eq["rect"]

        if current_merge is None:
            current_merge = {"indices": [block_index], "bbox": block_rect}
        else:
            last_merged_index = current_merge["indices"][-1]
            prev_block_y1 = text_blocks[last_merged_index].bbox.y1
            vertical_gap = block_rect.y0 - prev_block_y1
            max_merge_gap = block_rect.height * 1.5
            
            if block_index == last_merged_index + 1 and vertical_gap < max_merge_gap:
                current_merge["indices"].append(block_index)
                current_merge["bbox"].include_rect(block_rect)
            else:
                merged_equations.append(current_merge)
                current_merge = {"indices": [block_index], "bbox": block_rect}
    
    if current_merge:
        merged_equations.append(current_merge)

    # Create EquationElement objects
    for merged in merged_equations:
         equations.append(EquationElement(
             bbox=pdf_coords_to_bbox(merged["bbox"]),
             detection_source="layout_symbol_heuristic_v3"
         ))

    print(f"  Identified {len(equations)} potential equation regions (v3 heuristic).")
    return equations


# --- Main Processing Functions ---

async def process_page_async(page: fitz.Page, page_num: int, total_pages: int) -> PageData:
    """Processes a single page asynchronously (including async VLM calls)."""
    print(f"--- Processing Page {page_num}/{total_pages} ---")

    page_width, page_height = page.rect.width, page.rect.height
    page_dims = (page_width, page_height)

    page_data = PageData(
        page_number=page_num,
        dimensions=page_dims
    )
    
    # 1. Extract Text Blocks (Synchronous)
    words = []
    boxes_original = []
    text_content = page.get_text("words") 
    current_block_text = ""
    current_block_bbox = None
    for x0, y0, x1, y1, word, block_no, line_no, word_no in text_content:
        words.append(word)
        boxes_original.append((x0, y0, x1, y1))
        word_rect = fitz.Rect(x0, y0, x1, y1)
        if current_block_bbox is None:
            current_block_bbox = word_rect
            current_block_text = word
        elif word_rect.intersects(current_block_bbox) or abs(word_rect.y0 - current_block_bbox.y0) < 5:
            current_block_bbox.include_rect(word_rect)
            current_block_text += " " + word
        else:
            page_data.text_blocks.append(TextBlock(text=current_block_text.strip(), bbox=pdf_coords_to_bbox(current_block_bbox)))
            current_block_bbox = word_rect
            current_block_text = word
    if current_block_bbox:
         page_data.text_blocks.append(TextBlock(text=current_block_text.strip(), bbox=pdf_coords_to_bbox(current_block_bbox)))

    # 2. Get Page Image for Models (Synchronous)
    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2)) # Use consistent high-res image
    page_image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    img_render_width, img_render_height = page_image.size

    # 3. Extract LayoutLM Features (Synchronous)
    if words and boxes_original:
        try:
            input_ids, embeddings = extract_layoutlm_features(page_image, words, boxes_original, page_dims)
            print(f"  Extracted LayoutLM features (Embeddings shape: {embeddings.shape if hasattr(embeddings, 'shape') else 'N/A'})")
        except Exception as layout_err:
            print(f"  Error extracting LayoutLM features: {layout_err}")
    else:
        print("  Skipping LayoutLM feature extraction (no words/boxes).")

    # 4. Extract Annotations (Native PDF + CV Highlights + CV Ink)
    extracted_annotations = [] 
    
    # 4a. Native Annotations (via page.annots)
    page_annots = page.annots()
    if page_annots:
        print(f"  Page {page_num}: Found {len(list(page_annots))} raw native annotations.")
        for i, annot in enumerate(page_annots):
            try:
                print(f"    Raw Annot {i}: Type={annot.type}, Rect={annot.rect}, Colors={annot.colors}, Info={annot.info}")
                annot_type_code = annot.type[0]
                annot_type_str = annot.type[1]
                annot_rect = annot.rect
                text_in_annot = page.get_text("text", clip=annot_rect, sort=True).strip()
                comment_info, color_info, vertices_info, annot_info = None, None, None, None
                annot_info = annot.info
                color_info = annot.colors
                if annot_type_code == fitz.PDF_ANNOT_TEXT: annot_type_str, comment_info = "comment", annot_info
                elif annot_type_code == fitz.PDF_ANNOT_HIGHLIGHT: annot_type_str = "highlight"
                elif annot_type_code == fitz.PDF_ANNOT_UNDERLINE: annot_type_str = "underline"
                elif annot_type_code == fitz.PDF_ANNOT_STRIKEOUT: annot_type_str = "strikeout"
                elif annot_type_code == fitz.PDF_ANNOT_SQUARE: annot_type_str = "square"
                elif annot_type_code == fitz.PDF_ANNOT_CIRCLE: annot_type_str = "circle"
                elif annot_type_code == fitz.PDF_ANNOT_INK: annot_type_str, vertices_info = "ink", annot.vertices
                else: annot_type_str = annot_type_str.lower()
                annotation_obj = LegacyAnnotation(
                    type=annot_type_str.lower(), bbox=pdf_coords_to_bbox(annot_rect),
                    text_content=text_in_annot or None, comment_info=comment_info,
                    color=color_info or None, vertices=vertices_info or None
                )
                annotation_obj.semantic_tag = interpret_annotation_semantics(annotation_obj)
                extracted_annotations.append(annotation_obj)
            except Exception as annot_err:
                print(f"  Error processing native annotation {i}: {annot_err}.")
    else:
        print(f"  Page {page_num}: No native annotations found by page.annots().")

    # 4b. CV-based Highlight Detection
    try:
        print(f"  Page {page_num}: Running CV highlight detection for multiple colors...")
        color_ranges = getattr(config, 'CV_HIGHLIGHT_HSV_RANGES', {})
        print(f"    Loaded CV_HIGHLIGHT_HSV_RANGES from config: {color_ranges.keys() if color_ranges else 'None or Empty'}")
        if not color_ranges:
             print("    Warning: CV_HIGHLIGHT_HSV_RANGES not defined in config.py. Skipping CV highlight detection.")
        else:
            for color_name, hsv_bounds in color_ranges.items():
                if color_name.lower() == 'red': continue
                hsv_lower = hsv_bounds.get('lower')
                hsv_upper = hsv_bounds.get('upper')
                if hsv_lower is None or hsv_upper is None: continue
                print(f"    Detecting '{color_name}' highlights...")
                cv_highlight_bboxes = detect_color_highlight_regions(page_image, hsv_lower=hsv_lower, hsv_upper=hsv_upper)
                print(f"      CV detected {len(cv_highlight_bboxes)} potential '{color_name}' regions.")
                for cv_bbox in cv_highlight_bboxes:
                    x0_pdf = cv_bbox['x0'] * (page_width / img_render_width)
                    y0_pdf = cv_bbox['y0'] * (page_height / img_render_height)
                    x1_pdf = cv_bbox['x1'] * (page_width / img_render_width)
                    y1_pdf = cv_bbox['y1'] * (page_height / img_render_height)
                    pdf_rect = fitz.Rect(x0_pdf, y0_pdf, x1_pdf, y1_pdf)
                    text_in_cv_annot = page.get_text("text", clip=pdf_rect, sort=True).strip()
                    annotation_obj = LegacyAnnotation(
                        type="cv_highlight", bbox=pdf_coords_to_bbox(pdf_rect),
                        text_content=text_in_cv_annot or None, color=None,
                        detected_color_name=color_name
                    )
                    extracted_annotations.append(annotation_obj)
    except ImportError: print("  Skipping CV highlight detection: opencv-python or numpy not installed.")
    except Exception as cv_err: print(f"  Error during CV highlight detection: {cv_err}")

    # 4c. CV-based Ink/Drawing Detection (e.g., for Red)
    try:
        print(f"  Page {page_num}: Running CV ink detection (e.g., for red)... ")
        color_ranges = getattr(config, 'CV_HIGHLIGHT_HSV_RANGES', {})
        red_bounds = color_ranges.get('red')
        if not red_bounds: 
            print("    Warning: 'red' not defined in CV_HIGHLIGHT_HSV_RANGES. Skipping red ink detection.")
        else:
            hsv_lower1 = red_bounds.get('lower1')
            hsv_upper1 = red_bounds.get('upper1')
            hsv_lower2 = red_bounds.get('lower2')
            hsv_upper2 = red_bounds.get('upper2')
            if hsv_lower1 is None or hsv_upper1 is None:
                print("    Warning: Primary red HSV bounds (lower1/upper1) missing. Skipping red ink detection.")
            else:
                cv_ink_bboxes = detect_ink_regions(
                    page_image,
                    hsv_lower1=hsv_lower1, hsv_upper1=hsv_upper1,
                    hsv_lower2=hsv_lower2, hsv_upper2=hsv_upper2
                )
                print(f"    CV detected {len(cv_ink_bboxes)} potential red ink/drawing regions.")
                for cv_bbox in cv_ink_bboxes:
                    x0_pdf = cv_bbox['x0'] * (page_width / img_render_width)
                    y0_pdf = cv_bbox['y0'] * (page_height / img_render_height)
                    x1_pdf = cv_bbox['x1'] * (page_width / img_render_width)
                    y1_pdf = cv_bbox['y1'] * (page_height / img_render_height)
                    pdf_rect = fitz.Rect(x0_pdf, y0_pdf, x1_pdf, y1_pdf)
                    
                    # Extract text from the detected red regions
                    text_in_cv_annot = page.get_text("text", clip=pdf_rect, sort=True).strip()
                    
                    # Determine if this is likely a comment or drawing based on position and text
                    annot_type = "cv_drawing"
                    if text_in_cv_annot or (x0_pdf < page_width * 0.2 or x0_pdf > page_width * 0.8):
                        # If it has text or is in the margin, it's likely a comment
                        annot_type = "comment"
                    
                    annotation_obj = LegacyAnnotation(
                        type=annot_type, bbox=pdf_coords_to_bbox(pdf_rect),
                        text_content=text_in_cv_annot, color=None, 
                        detected_color_name="red"
                    )
                    extracted_annotations.append(annotation_obj)
    except ImportError: print("  Skipping CV ink detection: opencv-python or numpy not installed.")
    except Exception as cv_err: print(f"  Error during CV ink detection: {cv_err}")

    # Store combined annotations
    page_data.annotations = extracted_annotations
    print(f"  Stored {len(page_data.annotations)} total annotations (native + CV highlight + CV ink) for Page {page_num}.")

    # 5. Identify Visual Elements 
    visual_elements = identify_complex_regions(page, page_data.text_blocks)
    page_data.visual_elements = visual_elements

    # 5b. Identify Equation Regions
    equation_elements = identify_equation_regions(page_data.text_blocks, page_dims)
    page_data.equations = equation_elements

    # 6. Analyze Visual & Equation Elements with VLM Concurrently
    vlm_tasks = []
    elements_to_process = [] 
    for i, element in enumerate(visual_elements):
        if element.type == "figure": prompt = getattr(config, 'VLM_PROMPT_FIGURE', "Describe figure.")
        elif element.type == "drawing": prompt = getattr(config, 'VLM_PROMPT_DRAWING', "Describe drawing.")
        elif element.type == "table": prompt = getattr(config, 'VLM_PROMPT_TABLE', "Summarize table.")
        else: prompt = getattr(config, 'VLM_PROMPT_DESCRIPTION', "Describe region.")
        elements_to_process.append((element, i, prompt, "visual"))
    equation_prompt = getattr(config, 'VLM_PROMPT_EQUATION', "Transcribe equation.")
    for i, element in enumerate(equation_elements):
         elements_to_process.append((element, i, equation_prompt, "equation"))
    if elements_to_process:
        print(f"  Creating {len(elements_to_process)} VLM analysis tasks (Visuals & Equations)...")
        for element, index, prompt, elem_type in elements_to_process:
            try:
                img_bbox = (element.bbox.x0, element.bbox.y0, element.bbox.x1, element.bbox.y1)
                img_bbox_pil = (
                    max(0, img_bbox[0] * (pix.width / page_width)),
                    max(0, img_bbox[1] * (pix.height / page_height)),
                    min(pix.width, img_bbox[2] * (pix.width / page_width)),
                    min(pix.height, img_bbox[3] * (pix.height / page_height))
                )
                if img_bbox_pil[0] < img_bbox_pil[2] and img_bbox_pil[1] < img_bbox_pil[3]: 
                    region_image = page_image.crop(img_bbox_pil)
                    task = asyncio.create_task(analyze_image_region_with_vlm(region_image, prompt), name=f"VLM_Task_Page{page_num}_{elem_type}{index}")
                    vlm_tasks.append((elem_type, index, task))
                else:
                    print(f"    Skipping invalid crop area for {elem_type} {index} region {element.bbox}")
            except Exception as e:
                print(f"    Error preparing VLM task for {elem_type} {index} region {element.bbox}: {e}")
        if vlm_tasks:
            print(f"  Running {len(vlm_tasks)} VLM tasks concurrently for page {page_num}...")
            results = await asyncio.gather(*(task for _, _, task in vlm_tasks), return_exceptions=True)
            print(f"  Finished VLM tasks for page {page_num}.")
            refusal_phrases = [
                "i'm sorry", "i cannot", "i can't", "unable to assist", 
                "no equation visible", "not an equation", "no equation is present",
                "no equation to transcribe", "image does not contain an equation"
            ]
            for idx, (elem_type, element_index, _) in enumerate(vlm_tasks):
                result = results[idx]
                element_list = page_data.visual_elements if elem_type == "visual" else page_data.equations
                if element_index < len(element_list):
                    element_to_update = element_list[element_index]
                    processed_content = None
                    log_message = f"    VLM task for {elem_type} {element_index}"
                    if isinstance(result, Exception):
                        log_message += f" failed: {result}"
                    elif result and result.get('choices') and len(result['choices']) > 0:
                        message = result['choices'][0].get('message')
                        if message and message.get('content'):
                            content = message['content'].strip()
                            content_lower = content.lower()
                            is_refusal = False
                            for phrase in refusal_phrases:
                                if content_lower.startswith(phrase):
                                    is_refusal = True
                                    break
                            if not is_refusal:
                                processed_content = content 
                                log_message += f" succeeded: {content[:80]}..."
                            else:
                                log_message += " returned a refusal message."
                        else:
                            log_message += " failed to return content."
                    else:
                        log_message += " failed or returned empty/invalid result."
                    if elem_type == "visual":
                        element_to_update.vlm_description = processed_content
                    else:
                        element_to_update.vlm_transcription = processed_content
                    print(log_message)
                else:
                     print(f"    Error: Result index {element_index} out of bounds for {elem_type} list (size {len(element_list)}). Skipping result.")
    
    return page_data

async def process_document(pdf_path: str) -> Tuple[LegacyProcessedDocument, ProcessedDocument]:
    """
    Processes the entire PDF document asynchronously.
    Returns both the legacy and new format ProcessedDocument.
    """
    doc = None # Initialize doc to None
    try:
        doc = fitz.open(pdf_path)
        filename = os.path.basename(pdf_path)
        total_pages = len(doc)
        print(f"Processing document: {filename} ({total_pages} pages)")

        # Legacy document format
        legacy_processed_doc = LegacyProcessedDocument(
            metadata=DocumentMetadata(
                filename=filename,
                total_pages=total_pages,
                processing_timestamp=datetime.now().isoformat()
            ),
            pages=[]
        )

        # New document format - initialize basic structure
        doc_id = f"doc_{uuid.uuid4().hex[:8]}"
        
        new_processed_doc = ProcessedDocument(
            document=Document(
                id=doc_id,
                title=filename,
                sections=[],
                character_start=0,
                character_end=0,  # Will be updated after processing
                summary=None  # Extract from last section if available
            ),
            sections=[],
            paragraphs=[],
            annotations=[],
            metadata={
                "filename": filename,
                "total_pages": total_pages,
                "processing_timestamp": datetime.now().isoformat()
            }
        )

        # Process pages normally
        total_text_length = 0
        section_counter = 0
        paragraph_counter = 0
        annotation_counter = 0
        current_section = None
        
        # Section tracking flags
        in_citation_section = False
        in_appendix_section = False
        in_summary_section = False
        summary_sections = [] # Keep track of sections that might be summaries

        for page_index in range(total_pages):
            page = doc[page_index]
            page_data = await process_page_async(page, page_index + 1, total_pages)
            legacy_processed_doc.pages.append(page_data)
            
            # Track if we've processed any content on this page
            any_content_processed = False
            
            # Build paragraphs for new format
            for text_block in page_data.text_blocks:
                block_text = text_block.text.strip()
                if not block_text:
                    continue
                
                # Detect if this is a heading (simplified - can be improved)
                is_heading = False
                if len(block_text) < 100 and (
                    block_text.isupper() or 
                    any(heading in block_text.lower() for heading in [
                        "introduction", "abstract", "summary", "conclusion", 
                        "method", "result", "discussion", "reference", "appendix",
                        "chapter"
                    ])
                ):
                    is_heading = True
                
                # Check for section boundaries
                if is_heading:
                    # Check for citation/reference section
                    if is_citation_section(block_text):
                        in_citation_section = True
                        in_appendix_section = False
                        in_summary_section = False
                        print(f"Detected citation section on page {page_index+1}: '{block_text}'")
                    
                    # Check for appendix section
                    elif is_appendix_section(block_text):
                        in_citation_section = False
                        in_appendix_section = True
                        in_summary_section = False
                        print(f"Detected appendix section on page {page_index+1}: '{block_text}'")
                    
                    # Check for summary section
                    elif is_summary_section(block_text):
                        in_citation_section = False
                        in_appendix_section = False
                        in_summary_section = True
                        print(f"Detected summary section on page {page_index+1}: '{block_text}'")
                
                # Skip citation sections but process appendix and summary
                if in_citation_section and not in_appendix_section and not in_summary_section:
                    continue
                
                # If this is a heading, start a new section
                if is_heading:
                    section_id = f"sec_{section_counter + 1}"
                    section_counter += 1
                    current_section = Section(
                        id=section_id,
                        heading=block_text,
                        character_start=total_text_length,
                        character_end=total_text_length + len(block_text),
                        paragraphs=[]
                    )
                    new_processed_doc.document.sections.append(section_id)
                    new_processed_doc.sections.append(current_section)
                    
                    # Track potential summary sections
                    if in_summary_section or is_summary_section(block_text):
                        summary_sections.append(section_id)
                    
                    # Also create a paragraph for the heading
                    para_id = f"para_{paragraph_counter:03d}"
                    paragraph_counter += 1
                    paragraph = Paragraph(
                        id=para_id,
                        text=block_text,
                        character_start=total_text_length,
                        character_end=total_text_length + len(block_text),
                        annotations=[],
                        bbox=text_block.bbox,
                        page_number=page_index
                    )
                    
                    if current_section:
                        current_section.paragraphs.append(para_id)
                    
                    new_processed_doc.paragraphs.append(paragraph)
                    total_text_length += len(block_text)
                    any_content_processed = True
                else:
                    # Regular paragraph - skip if in citation section
                    if in_citation_section and not in_appendix_section and not in_summary_section:
                        continue
                        
                    para_id = f"para_{paragraph_counter:03d}"
                    paragraph_counter += 1
                    paragraph = Paragraph(
                        id=para_id,
                        text=block_text,
                        character_start=total_text_length,
                        character_end=total_text_length + len(block_text),
                        annotations=[],
                        bbox=text_block.bbox,
                        page_number=page_index
                    )
                    
                    if current_section:
                        current_section.paragraphs.append(para_id)
                    
                    new_processed_doc.paragraphs.append(paragraph)
                    total_text_length += len(block_text)
                    any_content_processed = True
            
            # Process visual elements and equations (especially important for summary)
            if page_data.visual_elements or page_data.equations:
                print(f"  Processing {len(page_data.visual_elements)} visual elements and {len(page_data.equations)} equations on page {page_index+1}")
                
                # Process visual elements with improved handling
                for visual_elem in page_data.visual_elements:
                    # Create a paragraph-like structure for the visual element
                    vis_elem_id = f"para_vis_{paragraph_counter:03d}"
                    paragraph_counter += 1
                    
                    # Determine element type with more detail
                    visual_type = visual_elem.type
                    
                    # Create better description text
                    description = visual_elem.vlm_description or f"Visual element ({visual_type})"
                    
                    # Create a proper paragraph for visual element
                    vis_paragraph = Paragraph(
                        id=vis_elem_id,
                        text=description,
                        character_start=total_text_length,
                        character_end=total_text_length + len(description),
                        annotations=[],
                        bbox=visual_elem.bbox,
                        page_number=page_index
                    )
                    
                    # Create a special annotation to mark this as a visual element
                    viz_annot_id = f"ann_vis_{annotation_counter:02d}"
                    annotation_counter += 1
                    
                    # Preserve original visual element type in semantic_tag
                    viz_annotation = Annotation(
                        id=viz_annot_id,
                        type="visual_element",  # Use distinct type for visual elements
                        color="#3366FF",  # Blue color for visual elements
                        referenced_text=description[:50] + ("..." if len(description) > 50 else ""),
                        referenced_char_start=total_text_length,
                        referenced_char_end=total_text_length + min(50, len(description)),
                        previous_text="",
                        posterior_text="",
                        paragraph_id=vis_elem_id,
                        annotated_text=f"VISUAL: {visual_type}",
                        bbox=visual_elem.bbox,
                        comment_info=None,
                        vertices=None,
                        semantic_tag=visual_type,  # Store original visual type
                        detected_color_name="blue",
                        color_info=None
                    )
                    
                    # Connect visual element to the current section
                    if current_section:
                        current_section.paragraphs.append(vis_elem_id)
                    
                    # Add to document
                    new_processed_doc.paragraphs.append(vis_paragraph)
                    vis_paragraph.annotations.append(viz_annot_id)
                    new_processed_doc.annotations.append(viz_annotation)
                    
                    total_text_length += len(description)
                    any_content_processed = True
                
                # Process equations with improved handling
                for equation in page_data.equations:
                    # Create a paragraph-like structure for the equation
                    eqn_id = f"para_eqn_{paragraph_counter:03d}"
                    paragraph_counter += 1
                    
                    # Create better equation text
                    eqn_text = equation.vlm_transcription or "Mathematical equation"
                    if eqn_text.startswith("```") or eqn_text.startswith("$$"):
                        # Clean up common LaTeX delimiters in VLM output
                        eqn_text = eqn_text.replace("```", "").replace("$$", "$").replace("\\[", "$").replace("\\]", "$")
                    
                    # Create paragraph for equation
                    eqn_paragraph = Paragraph(
                        id=eqn_id,
                        text=eqn_text,
                        character_start=total_text_length,
                        character_end=total_text_length + len(eqn_text),
                        annotations=[],
                        bbox=equation.bbox,
                        page_number=page_index
                    )
                    
                    # Also create a special annotation for this equation
                    eqn_annot_id = f"ann_eqn_{annotation_counter:02d}"
                    annotation_counter += 1
                    
                    # Use specific type for equations to distinguish them
                    eqn_annotation = Annotation(
                        id=eqn_annot_id,
                        type="equation",  # Use distinct type for equations
                        color="#FF6600",  # Orange color for equations
                        referenced_text=eqn_text[:50] + ("..." if len(eqn_text) > 50 else ""),
                        referenced_char_start=total_text_length,
                        referenced_char_end=total_text_length + min(50, len(eqn_text)),
                        previous_text="",
                        posterior_text="",
                        paragraph_id=eqn_id,
                        annotated_text="EQUATION",
                        bbox=equation.bbox,
                        comment_info=None,
                        vertices=None,
                        semantic_tag="equation",
                        detected_color_name="orange",
                        color_info=None
                    )
                    
                    # Add to section
                    if current_section:
                        current_section.paragraphs.append(eqn_id)
                    
                    # Add to document
                    new_processed_doc.paragraphs.append(eqn_paragraph)
                    eqn_paragraph.annotations.append(eqn_annot_id)
                    new_processed_doc.annotations.append(eqn_annotation)
                    
                    total_text_length += len(eqn_text)
                    any_content_processed = True
            
            # Process annotations for new format - prioritize non-citation sections
            if not in_citation_section or in_appendix_section or in_summary_section:
                for annot in page_data.annotations:
                    annotation_id = f"ann_{annotation_counter:02d}"
                    annotation_counter += 1
                    
                    # Find the paragraph this annotation belongs to
                    matched_para_id = None
                    closest_para = None
                    min_distance = float('inf')
                    
                    # First try exact overlap
                    for para in new_processed_doc.paragraphs:
                        # Only consider paragraphs on the same page
                        if para.page_number != page_index:
                            continue
                            
                        if para.bbox and is_rect_overlap(annot.bbox, para.bbox):
                            matched_para_id = para.id
                            break
                    
                    # If can't find by spatial overlap, try to match by content with more robust method
                    if not matched_para_id and annot.text_content:
                        for para in new_processed_doc.paragraphs:
                            # Only consider paragraphs on the same page
                            if para.page_number != page_index:
                                continue
                                
                            # Try exact content match first
                            if annot.text_content in para.text:
                                matched_para_id = para.id
                                break
                            
                            # If still no match, check for partial match with minimum 4-character overlap
                            # to avoid accidental matches on single characters or short sequences
                            if len(annot.text_content) >= 4:
                                # Look for at least 4-character sequences
                                for i in range(len(annot.text_content) - 3):
                                    substr = annot.text_content[i:i+4]
                                    if substr in para.text:
                                        matched_para_id = para.id
                                        break
                                if matched_para_id:
                                    break
                    
                    # If still no match, use the closest paragraph on the same page
                    if not matched_para_id:
                        for para in new_processed_doc.paragraphs:
                            # Only consider paragraphs on the same page
                            if para.page_number != page_index:
                                continue
                                
                            if para.bbox and annot.bbox:
                                # Calculate distance between centers
                                para_center_x = (para.bbox.x0 + para.bbox.x1) / 2
                                para_center_y = (para.bbox.y0 + para.bbox.y1) / 2
                                annot_center_x = (annot.bbox.x0 + annot.bbox.x1) / 2
                                annot_center_y = (annot.bbox.y0 + annot.bbox.y1) / 2
                                
                                distance = ((para_center_x - annot_center_x) ** 2 + 
                                        (para_center_y - annot_center_y) ** 2) ** 0.5
                                
                                if distance < min_distance:
                                    min_distance = distance
                                    closest_para = para
                        
                        if closest_para:
                            matched_para_id = closest_para.id
                    
                    # If a paragraph match was found or created
                    if matched_para_id:
                        para = next(p for p in new_processed_doc.paragraphs if p.id == matched_para_id)
                        
                        # Get referenced text with improved handling
                        referenced_text = annot.text_content or ""
                        
                        # Initialize prev_text and post_text to empty strings by default
                        prev_text = ""
                        post_text = ""
                        
                        # Find character start/end in paragraph
                        char_start = para.character_start
                        char_end = char_start + len(referenced_text) if referenced_text else para.character_end
                        
                        # If text is in paragraph, get exact position
                        if referenced_text and referenced_text in para.text:
                            start_in_para = para.text.find(referenced_text)
                            char_start = para.character_start + start_in_para
                            char_end = char_start + len(referenced_text)
                            
                            # Get context before and after
                            context_size = 50  # Number of characters for context
                            prev_text = para.text[max(0, start_in_para - context_size):start_in_para]
                            post_text = para.text[start_in_para + len(referenced_text):
                                              min(len(para.text), start_in_para + len(referenced_text) + context_size)]
                        elif referenced_text and len(referenced_text) >= 4:
                            # Try to find partial match
                            match_found = False
                            for i in range(len(referenced_text) - 3):
                                substr = referenced_text[i:i+4]
                                if substr in para.text:
                                    start_in_para = para.text.find(substr)
                                    max_match_len = min(len(referenced_text) - i, len(para.text) - start_in_para)
                                    match_len = 4  # Start with minimum match length
                                    
                                    # Extend match as far as possible
                                    while match_len < max_match_len and referenced_text[i:i+match_len+1] == para.text[start_in_para:start_in_para+match_len+1]:
                                        match_len += 1
                                    
                                    char_start = para.character_start + start_in_para
                                    char_end = char_start + match_len
                                    
                                    # Get context
                                    context_size = 50
                                    prev_text = para.text[max(0, start_in_para - context_size):start_in_para]
                                    post_text = para.text[start_in_para + match_len:
                                                    min(len(para.text), start_in_para + match_len + context_size)]
                                    
                                    # Use the matched text
                                    referenced_text = para.text[start_in_para:start_in_para+match_len]
                                    match_found = True
                                    break
                            
                            # If no match was found in the loop, keep defaults
                            if not match_found:
                                # For comments, try to find nearest text to associate with
                                if annot.type == "comment" or annot.type == "cv_drawing":
                                    # Find nearest paragraph text based on spatial proximity
                                    if annot.bbox and para.bbox:
                                        # Calculate proximity points (left, center, right of annotation)
                                        annot_x_center = (annot.bbox.x0 + annot.bbox.x1) / 2
                                        annot_y_bottom = annot.bbox.y1
                                        
                                        # Find nearest text in paragraph - use first 100 chars as context
                                        preview_text = para.text[:min(100, len(para.text))]
                                        referenced_text = preview_text
                                        char_start = para.character_start
                                        char_end = char_start + len(preview_text)
                                        
                                        # Get extended context
                                        context_size = 100
                                        if len(para.text) > context_size:
                                            prev_text = para.text[:context_size//2]
                                            post_text = para.text[context_size//2:context_size]
                                        else:
                                            prev_text = ""
                                            post_text = para.text
                                else:
                                    # No specific text match, use entire paragraph reference as fallback
                                    prev_text = ""
                                    post_text = ""
                        
                        # Determine annotation type with improved handling
                        annot_type = annot.type
                        if annot_type == "cv_highlight":
                            annot_type = "highlight"
                        elif annot_type == "cv_drawing" or annot_type == "ink":
                            # Improved detection logic for comments vs symbols
                            if annot.detected_color_name == "red" and annot.bbox:
                                # Get page dimensions from page_data
                                page_width = page_data.dimensions[0] if page_data.dimensions and len(page_data.dimensions) == 2 else 612
                                
                                # Right margin
                                if annot.bbox.x0 > page_width * 0.8:
                                    annot_type = "comment"
                                # Left margin
                                elif annot.bbox.x1 < page_width * 0.2:
                                    annot_type = "comment"
                                else:
                                    annot_type = "ink" if annot.vertices else "symbol"
                            else:
                                annot_type = "ink" if annot.vertices else "symbol"
                        
                        # Special handling for ink annotations with vertices
                        vertices = None
                        if annot.vertices and annot_type == "ink":
                            vertices = annot.vertices
                        
                        # Determine color with more robust handling
                        color = annot.detected_color_name or "yellow"  # Default to yellow if not specified
                        if annot.color and isinstance(annot.color, dict):
                            if "stroke" in annot.color:
                                r, g, b = annot.color["stroke"]
                                color = f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"
                            elif any(key in annot.color for key in ["fill", "interior"]):
                                key = next(k for k in ["fill", "interior"] if k in annot.color)
                                r, g, b = annot.color[key]
                                color = f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"
                        
                        # Create annotation with complete information
                        annotation = Annotation(
                            id=annotation_id,
                            type=annot_type,
                            color=color,
                            referenced_text=referenced_text,
                            referenced_char_start=char_start,
                            referenced_char_end=char_end,
                            previous_text=prev_text,
                            posterior_text=post_text,
                            paragraph_id=matched_para_id,
                            annotated_text=annot.text_content if annot_type == "comment" else referenced_text,  # Use text_content for comments
                            bbox=annot.bbox,
                            comment_info=annot.comment_info,
                            vertices=vertices,
                            semantic_tag=annot.semantic_tag,
                            detected_color_name=annot.detected_color_name,
                            color_info=annot.color
                        )
                        
                        # Add annotation to list and to paragraph
                        para.annotations.append(annotation_id)
                        new_processed_doc.annotations.append(annotation)
                        any_content_processed = True
                    else:
                        # If no paragraph match, create a special standalone paragraph for this annotation
                        para_id = f"para_{paragraph_counter:03d}"
                        paragraph_counter += 1
                        
                        # Create text for annotation
                        annot_text = annot.text_content or f"Annotation: {annot.type}"
                        
                        # Create paragraph for annotation
                        paragraph = Paragraph(
                            id=para_id,
                            text=annot_text,
                            character_start=total_text_length,
                            character_end=total_text_length + len(annot_text),
                            annotations=[],
                            bbox=annot.bbox,
                            page_number=page_index
                        )
                        
                        # Add paragraph to document
                        if current_section:
                            current_section.paragraphs.append(para_id)
                        new_processed_doc.paragraphs.append(paragraph)
                        total_text_length += len(annot_text)
                        
                        # Create annotation linked to this paragraph
                        annotation = Annotation(
                            id=annotation_id,
                            type=annot.type,
                            color=annot.detected_color_name or "yellow",
                            referenced_text=annot_text,
                            referenced_char_start=total_text_length - len(annot_text),
                            referenced_char_end=total_text_length,
                            previous_text="",
                            posterior_text="",
                            paragraph_id=para_id,
                            annotated_text=annot.text_content if annot.type == "comment" else annot_text,
                            bbox=annot.bbox,
                            comment_info=annot.comment_info,
                            vertices=annot.vertices,
                            semantic_tag=annot.semantic_tag,
                            detected_color_name=annot.detected_color_name,
                            color_info=annot.color
                        )
                        
                        # Add annotation to list and to paragraph
                        paragraph.annotations.append(annotation_id)
                        new_processed_doc.annotations.append(annotation)
                        any_content_processed = True
            
            # Print page processing status
            if any_content_processed:
                print(f"  Processed content from page {page_index+1}")
            else:
                print(f"  No content extracted from page {page_index+1}")
                
            page = None  # Free memory
            
        # Update document's total character count
        new_processed_doc.document.character_end = total_text_length
        
        # If no sections were detected, create a single default section
        if not new_processed_doc.sections:
            default_section = Section(
                id="sec_1",
                heading="Document",
                character_start=0,
                character_end=total_text_length,
                paragraphs=[p.id for p in new_processed_doc.paragraphs]
            )
            new_processed_doc.document.sections.append("sec_1")
            new_processed_doc.sections.append(default_section)
        
        # Extract summary from tracked summary sections or from last section if none found
        summary_text = None
        
        # First try to extract from detected summary sections
        if summary_sections:
            summary_paragraphs = []
            for section_id in summary_sections:
                section = next((s for s in new_processed_doc.sections if s.id == section_id), None)
                if section:
                    for para_id in section.paragraphs:
                        para = next((p for p in new_processed_doc.paragraphs if p.id == para_id), None)
                        if para:
                            summary_paragraphs.append(para.text)
            
            if summary_paragraphs:
                summary_text = " ".join(summary_paragraphs)
                print(f"Extracted summary from explicitly marked summary sections ({len(summary_paragraphs)} paragraphs)")
        
        # If no summary found, try to find from any section with summary in the title
        if not summary_text:
            for section in reversed(new_processed_doc.sections):
                if is_summary_section(section.heading):
                    summary_paragraphs = []
                    for para_id in section.paragraphs:
                        para = next((p for p in new_processed_doc.paragraphs if p.id == para_id), None)
                        if para:
                            summary_paragraphs.append(para.text)
                    
                    if summary_paragraphs:
                        summary_text = " ".join(summary_paragraphs)
                        print(f"Extracted summary from section with summary in title: '{section.heading}' ({len(summary_paragraphs)} paragraphs)")
                        break
        
        # If still no summary, try to extract from last non-citation, non-appendix section
        if not summary_text:
            # Go backwards through sections to find the last main content section
            for section in reversed(new_processed_doc.sections):
                if not is_citation_section(section.heading) and not is_appendix_section(section.heading):
                    summary_paragraphs = []
                    # Get the first few paragraphs only
                    for para_id in section.paragraphs[:3]:  # Limit to first 3 paragraphs
                        para = next((p for p in new_processed_doc.paragraphs if p.id == para_id), None)
                        if para:
                            summary_paragraphs.append(para.text)
                    
                    if summary_paragraphs:
                        summary_text = " ".join(summary_paragraphs)
                        print(f"Extracted summary from last main content section: '{section.heading}' ({len(summary_paragraphs)} paragraphs)")
                        break
        
        new_processed_doc.document.summary = summary_text

        print("--- Document Processing Complete ---")
        return legacy_processed_doc, new_processed_doc

    except Exception as e:
        print(f"Error processing PDF {pdf_path}: {e}")
        # Return empty documents
        return (
            LegacyProcessedDocument(metadata=DocumentMetadata(filename=os.path.basename(pdf_path) if pdf_path else 'unknown', total_pages=0)),
            ProcessedDocument(
                document=Document(
                    id=f"doc_error",
                    title=os.path.basename(pdf_path) if pdf_path else 'unknown',
                    sections=[],
                    character_end=0
                ),
                sections=[],
                paragraphs=[],
                annotations=[]
            )
        )
    finally:
        if doc: # Ensure doc is closed even if errors occurred after opening
            doc.close()

def save_processed_document(processed_doc: Union[LegacyProcessedDocument, ProcessedDocument], output_dir: str, filename: str):
    """Saves the processed document data to a JSON file."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_path = os.path.join(output_dir, filename)

    # Use Pydantic's serialization method for proper handling of types
    json_output = processed_doc.model_dump_json(indent=getattr(config, 'JSON_INDENT', 2))

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(json_output)
    print(f"Processed document saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a PDF document to extract multimodal features.")
    parser.add_argument("--pdf_path", required=True, help="Path to the input PDF file.")
    args = parser.parse_args()

    print("Starting document processing pipeline...")
    if not config.OPENAI_API_KEY:
         print("Warning: OpenAI API Key not set. VLM analysis will be skipped.")

    pdf_file = args.pdf_path
    if not os.path.exists(pdf_file):
        print(f"Error: PDF file not found at {pdf_file}")
    else:
        base_name = os.path.basename(pdf_file)
        file_name_without_ext = os.path.splitext(base_name)[0]
        # Ensure output dir exists for JSON filename generation
        if not os.path.exists(config.OUTPUT_DIR):
            os.makedirs(config.OUTPUT_DIR)
        legacy_output_json_filename = f"{file_name_without_ext}_processed_legacy.json"
        new_output_json_filename = f"{file_name_without_ext}_processed.json"

        legacy_processed_data, new_processed_data = asyncio.run(process_document(pdf_file))

        # Save both formats
        if legacy_processed_data and legacy_processed_data.pages:
            save_processed_document(
                legacy_processed_data,
                config.OUTPUT_DIR,
                legacy_output_json_filename 
            )
        else:
            print("Error: Legacy document processing failed or produced no data. Skipping save.")
        
        if new_processed_data and new_processed_data.document:
            save_processed_document(
                new_processed_data,
                config.OUTPUT_DIR,
                new_output_json_filename 
            )
        else:
            print("Error: New document processing failed or produced no data. Skipping save.")
            
    print("Pipeline finished.")
