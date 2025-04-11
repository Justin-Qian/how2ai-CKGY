import fitz  # PyMuPDF
import json
import os
import argparse # Added import
from data_structures import ProcessedDocument, BoundingBox # Import necessary Pydantic models
import config

# --- Configuration for Visualization ---
VISUALIZATION_COLORS = {
    "text_block": (0, 0, 1),    # Blue
    "annotation_highlight": (1, 1, 0), # Yellow
    "annotation_underline": (0, 1, 0), # Green
    "annotation_comment": (1, 0, 0),   # Red
    "annotation_other": (0.5, 0.5, 0.5), # Grey
    "visual_element": (0.5, 0, 0.5), # Purple
}
TEXT_COLOR = (1, 0, 0) # Red for text labels
TEXT_FONTSIZE = 6
BOX_WIDTH = 0.5 # Line width for boxes
# OUTPUT_FILENAME = "visualized_output.pdf" # Removed, will be generated dynamically
# --- End Configuration ---

def draw_bbox(page: fitz.Page, bbox: BoundingBox, color: tuple, width: float, label: str = None, text_color: tuple = TEXT_COLOR, font_size: int = TEXT_FONTSIZE):
    """Draws a bounding box and optional label on the page."""
    rect = fitz.Rect(bbox.x0, bbox.y0, bbox.x1, bbox.y1)
    page.draw_rect(rect, color=color, width=width, fill=None) # Use fill=None for transparent rectangle
    if label:
        # Position label slightly above the top-left corner
        label_pos = fitz.Point(rect.x0, rect.y0 - font_size - 1)
        # Ensure label position is within page bounds
        if label_pos.y < 0: label_pos.y = rect.y0 + 1
        if label_pos.x < 0: label_pos.x = 0

        page.insert_text(label_pos, label, fontsize=font_size, color=text_color, fontname="helv") # Use a standard font

def visualize_processed_document(json_path: str, pdf_path: str, output_path: str):
    """
    Loads processed data and original PDF, draws annotations, and saves a new PDF.
    """
    print(f"Loading processed data from: {json_path}")
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        processed_doc = ProcessedDocument(**data) # Load data into Pydantic models
    except FileNotFoundError:
        print(f"Error: Processed JSON file not found at {json_path}")
        return
    except Exception as e:
        print(f"Error loading or parsing JSON file: {e}")
        return

    print(f"Loading original PDF from: {pdf_path}")
    try:
        doc = fitz.open(pdf_path)
    except FileNotFoundError:
        print(f"Error: Original PDF file not found at {pdf_path}")
        return
    except Exception as e:
        print(f"Error opening PDF file: {e}")
        return

    if len(doc) != processed_doc.metadata.total_pages:
        print("Warning: PDF page count doesn't match metadata page count.")

    print("Generating visualization...")
    for i, page_data in enumerate(processed_doc.pages):
        if i >= len(doc):
            print(f"Warning: More pages in JSON ({i+1}) than in PDF ({len(doc)}). Skipping extra pages.")
            break

        page = doc[i] # Get the corresponding page from the PDF
        print(f"  Visualizing page {page_data.page_number}/{processed_doc.metadata.total_pages}")

        # Draw Text Blocks
        for tb in page_data.text_blocks:
            draw_bbox(page, tb.bbox, VISUALIZATION_COLORS["text_block"], BOX_WIDTH)

        # Draw Annotations
        for annot in page_data.annotations:
            color_key = f"annotation_{annot.type}"
            color = VISUALIZATION_COLORS.get(color_key, VISUALIZATION_COLORS["annotation_other"])
            label = f"Annot: {annot.type}"
            if annot.comment_info and annot.comment_info.get('content'):
                 label += f" ({annot.comment_info['content'][:30]}...)" # Add snippet of comment
            elif annot.text_content:
                 label += f" ({annot.text_content[:30]}...)"
            draw_bbox(page, annot.bbox, color, BOX_WIDTH, label=label)

        # Draw Visual Elements
        for ve in page_data.visual_elements:
            label = f"Visual: {ve.type}"
            if ve.vlm_description:
                label += f" (VLM: {ve.vlm_description[:50]}...)" # Add snippet of VLM desc
            draw_bbox(page, ve.bbox, VISUALIZATION_COLORS["visual_element"], BOX_WIDTH, label=label)

    # Save the modified PDF
    output_file_path = os.path.join(os.path.dirname(json_path), output_path) # Save in the same dir as JSON
    try:
        doc.save(output_file_path, garbage=4, deflate=True, clean=True)
        print(f"Visualization saved successfully to: {output_file_path}")
    except Exception as e:
        print(f"Error saving visualized PDF: {e}")
    finally:
        doc.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize processed document annotations on the original PDF.")
    parser.add_argument("--pdf_path", required=True, help="Path to the original PDF file that was processed.")
    args = parser.parse_args()

    original_pdf_file = args.pdf_path

    # Infer input JSON and output PDF filenames
    base_name = os.path.basename(original_pdf_file)
    file_name_without_ext = os.path.splitext(base_name)[0]
    processed_json_filename = f"{file_name_without_ext}_processed.json"
    output_viz_filename = f"{file_name_without_ext}_visualized.pdf"

    processed_json_file = os.path.join(config.OUTPUT_DIR, processed_json_filename)
    output_viz_file_path = os.path.join(config.OUTPUT_DIR, output_viz_filename) # Full path for saving

    if not os.path.exists(original_pdf_file):
         print(f"Error: Original PDF file not found at {original_pdf_file}.")
    elif not os.path.exists(processed_json_file):
        print(f"Error: Processed JSON file not found at {processed_json_file}. Make sure document_processor.py ran successfully for this PDF.")
    else:
        # Pass the full output path to the function
        visualize_processed_document(processed_json_file, original_pdf_file, output_viz_file_path)
