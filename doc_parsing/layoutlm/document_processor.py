import fitz  # PyMuPDF
import os
import json
from PIL import Image
import io
from datetime import datetime
from typing import List
import argparse # Added import

# Import configurations and data structures
import config
from data_structures import (
    ProcessedDocument, DocumentMetadata, PageData,
    TextBlock, BoundingBox, Annotation, VisualElement
)

# Import helper functions
from layoutlm_utils import extract_layoutlm_features, preprocess_image_for_layoutlm
from vlm_utils import analyze_image_region_with_vlm

def pdf_coords_to_bbox(rect: fitz.Rect) -> BoundingBox:
    """Converts PyMuPDF Rect coordinates to our BoundingBox model."""
    return BoundingBox(x0=rect.x0, y0=rect.y0, x1=rect.x1, y1=rect.y1)

def identify_complex_regions(page: fitz.Page, text_blocks: list) -> List[VisualElement]:
    """
    Placeholder function to identify regions potentially needing VLM analysis.
    This needs more sophisticated logic based on heuristics or models.
    """
    complex_regions = []
    page_width, page_height = page.rect.width, page.rect.height

    # Example Heuristic 1: Detect large images
    image_list = page.get_image_info(xrefs=True)
    for img_info in image_list:
        img_bbox_rect = img_info['bbox']
        img_area = (img_bbox_rect[2] - img_bbox_rect[0]) * (img_bbox_rect[3] - img_bbox_rect[1])

        # Check if area exceeds threshold and doesn't significantly overlap with text
        # (Overlap check logic would be needed here)
        if img_area > config.MIN_IMAGE_AREA_FOR_VLM:
            # Check for overlap with text blocks (simplified check)
            is_overlapping = False
            for tb in text_blocks:
                tb_rect = fitz.Rect(tb.bbox.x0, tb.bbox.y0, tb.bbox.x1, tb.bbox.y1)
                if tb_rect.intersects(img_bbox_rect):
                    # More sophisticated overlap calculation might be needed
                    is_overlapping = True
                    break
            
            if not is_overlapping:
                 complex_regions.append(VisualElement(
                     type="figure", # Assume figure for now
                     bbox=pdf_coords_to_bbox(fitz.Rect(img_bbox_rect))
                 ))

    # Example Heuristic 2: Look for keywords in text blocks (e.g., "Figure:", "Table:")
    # ... implementation needed ...

    # Example Heuristic 3: Check annotations pointing to visual areas
    # ... implementation needed ...

    return complex_regions


def process_document(pdf_path: str) -> ProcessedDocument:
    """
    Processes the entire PDF document according to the defined pipeline.
    """
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
        page_num = page_index + 1
        print(f"--- Processing Page {page_num}/{total_pages} ---")

        page_width, page_height = page.rect.width, page.rect.height
        page_dims = (page_width, page_height)

        page_data = PageData(
            page_number=page_num,
            dimensions=page_dims
        )

        # 1. Extract Text Blocks and Words/Boxes for LayoutLM
        words = []
        boxes_original = [] # Store original PDF coordinates for mapping
        text_content = page.get_text("words") # List of [x0, y0, x1, y1, word, block_no, line_no, word_no]

        current_block_text = ""
        current_block_bbox = None

        for x0, y0, x1, y1, word, block_no, line_no, word_no in text_content:
            words.append(word)
            bbox_tuple = (x0, y0, x1, y1)
            boxes_original.append(bbox_tuple)

            # Simple text block aggregation (can be improved)
            word_rect = fitz.Rect(x0, y0, x1, y1)
            if current_block_bbox is None:
                current_block_bbox = word_rect
                current_block_text = word
            elif word_rect.intersects(current_block_bbox) or abs(word_rect.y0 - current_block_bbox.y0) < 5: # Basic proximity check
                current_block_bbox.include_rect(word_rect)
                current_block_text += " " + word
            else:
                # Save previous block
                page_data.text_blocks.append(TextBlock(
                    text=current_block_text.strip(),
                    bbox=pdf_coords_to_bbox(current_block_bbox)
                ))
                # Start new block
                current_block_bbox = word_rect
                current_block_text = word
        
        # Add the last block
        if current_block_bbox:
             page_data.text_blocks.append(TextBlock(
                text=current_block_text.strip(),
                bbox=pdf_coords_to_bbox(current_block_bbox)
            ))


        # 2. Get Page Image for Models
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2)) # Higher resolution rendering
        page_image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

        # 3. Extract LayoutLM Features (Embeddings)
        # Note: Storing raw embeddings in JSON can be very large.
        # Consider alternatives like saving embeddings separately or omitting if not needed.
        if words and boxes_original:
            input_ids, embeddings = extract_layoutlm_features(page_image, words, boxes_original, page_dims)
            # How to map embeddings back to text_blocks needs careful implementation
            # based on tokenization alignment. Skipping embedding storage in JSON for now.
            print(f"  Extracted LayoutLM features (Embeddings shape: {embeddings.shape})")
        else:
             print("  Skipping LayoutLM feature extraction (no words/boxes).")


        # 4. Extract Annotations
        page_annots = page.annots()
        if page_annots:
            for annot in page_annots:
                annot_type_code = annot.type[0]
                annot_type_str = annot.type[1] # e.g., 'Highlight', 'Text', 'Square'
                annot_rect = annot.rect
                text_in_annot = page.get_textbox(annot_rect).strip()
                comment_info = None
                if annot_type_code == fitz.PDF_ANNOT_TEXT: # Comment/Note
                    comment_info = annot.info # Contains 'content', 'title' etc.
                    annot_type_str = "comment" # Standardize type
                elif annot_type_code == fitz.PDF_ANNOT_HIGHLIGHT:
                     annot_type_str = "highlight"
                elif annot_type_code == fitz.PDF_ANNOT_UNDERLINE:
                     annot_type_str = "underline"
                # Add more annotation types as needed

                page_data.annotations.append(Annotation(
                    type=annot_type_str.lower(),
                    bbox=pdf_coords_to_bbox(annot_rect),
                    text_content=text_in_annot if text_in_annot else None,
                    comment_info=comment_info
                ))
            print(f"  Extracted {len(page_data.annotations)} annotations.")

        # 5. Identify Complex Regions (Placeholder)
        complex_regions = identify_complex_regions(page, page_data.text_blocks)
        print(f"  Identified {len(complex_regions)} potential complex regions (placeholder).")

        # 6. Analyze Complex Regions with VLM (Placeholder)
        for region in complex_regions:
            try:
                # Crop the image region
                img_bbox = (region.bbox.x0, region.bbox.y0, region.bbox.x1, region.bbox.y1)
                # Ensure coordinates are within page bounds before cropping
                img_bbox_pil = (
                    max(0, img_bbox[0] * (pix.width / page_width)),
                    max(0, img_bbox[1] * (pix.height / page_height)),
                    min(pix.width, img_bbox[2] * (pix.width / page_width)),
                    min(pix.height, img_bbox[3] * (pix.height / page_height))
                )
                
                if img_bbox_pil[0] < img_bbox_pil[2] and img_bbox_pil[1] < img_bbox_pil[3]: # Check valid crop area
                    region_image = page_image.crop(img_bbox_pil)
                    
                    # Determine appropriate prompt based on region type or context
                    prompt = config.VLM_PROMPT_DESCRIPTION # Default prompt
                    # Add logic here to choose a more specific prompt if possible

                    print(f"    Analyzing region {region.bbox} with VLM...")
                    vlm_result = analyze_image_region_with_vlm(region_image, prompt)

                    if vlm_result and vlm_result.get('choices') and len(vlm_result['choices']) > 0:
                        message = vlm_result['choices'][0].get('message')
                        if message and message.get('content'):
                            region.vlm_description = message['content']
                            print(f"      VLM Result: {region.vlm_description[:100]}...") # Print snippet
                        else:
                             print("      VLM analysis failed to return content.")
                    else:
                        print("      VLM analysis failed or returned no choices.")
                else:
                    print(f"    Skipping invalid crop area for region {region.bbox}")

            except Exception as e:
                print(f"    Error processing region {region.bbox} for VLM: {e}")

            page_data.visual_elements.append(region) # Add region even if VLM fails

        # Add page data to document
        processed_doc.pages.append(page_data)

    print("--- Document Processing Complete ---")
    return processed_doc

def save_processed_document(processed_doc: ProcessedDocument, output_dir: str, filename: str):
    """Saves the processed document data to a JSON file."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_path = os.path.join(output_dir, filename)

    # Use Pydantic's serialization method for proper handling of types
    # Use model_dump_json for direct JSON string output
    json_output = processed_doc.model_dump_json(indent=config.JSON_INDENT)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(json_output)
    print(f"Processed document saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a PDF document to extract multimodal features.")
    parser.add_argument("--pdf_path", required=True, help="Path to the input PDF file.")
    args = parser.parse_args()

    print("Starting document processing pipeline...")
    # Ensure API key is loaded (or handle absence)
    if not config.OPENAI_API_KEY:
         print("Warning: OpenAI API Key not set. VLM analysis will be skipped.")

    pdf_file = args.pdf_path
    if not os.path.exists(pdf_file):
        print(f"Error: PDF file not found at {pdf_file}")
    else:
        # Generate dynamic output filename
        base_name = os.path.basename(pdf_file)
        file_name_without_ext = os.path.splitext(base_name)[0]
        output_json_filename = f"{file_name_without_ext}_processed.json"

        processed_data = process_document(pdf_file)
        save_processed_document(
            processed_data,
            config.OUTPUT_DIR,
            output_json_filename # Use dynamic filename
        )
    print("Pipeline finished.")
