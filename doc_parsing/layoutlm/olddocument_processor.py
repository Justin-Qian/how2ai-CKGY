import fitz  # PyMuPDF
import os
import json
from PIL import Image
import io
from datetime import datetime
from typing import List, Optional, Tuple
import argparse
import asyncio # Import asyncio
import re # Import regex for symbol checking

# Import configurations and data structures
from . import config
from .data_structures import (
    ProcessedDocument, DocumentMetadata, PageData,
    TextBlock, BoundingBox, Annotation, VisualElement, EquationElement
)

# Import helper functions
from .layoutlm_utils import extract_layoutlm_features # Keep only necessary layoutlm import
from .vlm_utils import analyze_image_region_with_vlm
from .cv_utils import detect_color_highlight_regions, detect_ink_regions # Import both CV functions

def pdf_coords_to_bbox(rect: fitz.Rect) -> BoundingBox:
    """Converts PyMuPDF Rect coordinates to our BoundingBox model."""
    return BoundingBox(x0=rect.x0, y0=rect.y0, x1=rect.x1, y1=rect.y1)

# Helper function to check if a rectangle is likely too thin or wide (potentially decorative)
def is_extreme_aspect_ratio(rect: fitz.Rect, max_ratio: float = 10.0) -> bool:
    width = rect.width
    height = rect.height
    if width <= 0 or height <= 0:
        return True # Invalid rect
    ratio = max(width / height, height / width)
    return ratio > max_ratio

def interpret_annotation_semantics(annotation_data: Annotation) -> Optional[str]:
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
                annotation_obj = Annotation(
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
                    annotation_obj = Annotation(
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
                    text_in_cv_annot = None 
                    annotation_obj = Annotation(
                        type="cv_drawing", bbox=pdf_coords_to_bbox(pdf_rect),
                        text_content=None, color=None, 
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

async def process_document(pdf_path: str) -> ProcessedDocument:
    """
    Processes the entire PDF document asynchronously.
    """
    doc = None # Initialize doc to None
    try:
        doc = fitz.open(pdf_path)
        filename = os.path.basename(pdf_path)
        total_pages = len(doc)
        print(f"Processing document: {filename} ({total_pages} pages)")

        processed_doc = ProcessedDocument(
            metadata=DocumentMetadata(
                filename=filename,
                total_pages=total_pages,
                processing_timestamp=datetime.now().isoformat()
            ),
            pages=[]
        )

        for page_index in range(total_pages):
            page = doc[page_index]
            page_data = await process_page_async(page, page_index + 1, total_pages)
            processed_doc.pages.append(page_data)
            page = None # Attempt to free memory
            page_data = None # Attempt to free memory

        print("--- Document Processing Complete ---")
        return processed_doc

    except Exception as e:
        print(f"Error processing PDF {pdf_path}: {e}")
        # Return an empty ProcessedDocument or raise error
        return ProcessedDocument(metadata=DocumentMetadata(filename=os.path.basename(pdf_path) if pdf_path else 'unknown', total_pages=0))
    finally:
        if doc: # Ensure doc is closed even if errors occurred after opening
            doc.close()

def save_processed_document(processed_doc: ProcessedDocument, output_dir: str, filename: str):
    """Saves the processed document data to a JSON file."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_path = os.path.join(output_dir, filename)

    # Use Pydantic's serialization method for proper handling of types
    # Use model_dump_json for direct JSON string output
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
        output_json_filename = f"{file_name_without_ext}_processed.json"
        output_json_path = os.path.join(config.OUTPUT_DIR, output_json_filename)

        processed_data = asyncio.run(process_document(pdf_file))

        # Check if processed_data is valid before saving
        if processed_data and processed_data.pages:
            save_processed_document(
                processed_data,
                config.OUTPUT_DIR,
                output_json_filename 
            )
        else:
            print("Error: Document processing failed or produced no data. Skipping save.")
            
    print("Pipeline finished.")