import fitz  # PyMuPDF
import json
import os
import argparse
from typing import Dict, Any, Tuple, Optional

# --- Configuration for Visualization ---
VIZ_COLORS = {
    "text_block": (0, 0, 1),  # Blue
    "figure": (0, 0.7, 0),  # Dark Green
    "drawing": (0.7, 0, 0.7), # Purple
    "equation": (0.2, 0.5, 0.7), # Bluish-Green
    "annotation_default": (1, 0, 0),  # Red
    "highlight": (0.9, 0.9, 0), # Yellowish - Box outline for native
    "cv_highlight": (1.0, 0.75, 0.0), # Orange - Box outline for CV-detected
    "cv_drawing": (1.0, 0.0, 0.0),   # Red - Box outline for CV-detected ink/drawings
    "comment": (0, 0.7, 0.7), # Teal
    "underline": (1, 0.5, 0), # Orange
    "strikeout": (0.5, 0.5, 0.5), # Grey
    "square": (1, 0, 0.5), # Pink
    "circle": (0.5, 0, 1), # Violet
    "ink": (0.5, 0.3, 0), # Brown
}
OUTLINE_COLOR_DEFAULT = (0.2, 0.2, 0.2) # Dark grey outline for elements with fill
TEXT_COLOR = (0, 0, 0) # Black
LINE_WIDTH = 1.0 # Make lines slightly thinner
INK_LINE_WIDTH = 1.5
FONTSIZE = 6 # Slightly smaller default font
TEXT_OFFSET = FONTSIZE + 2 # Offset text above the box

def hex_to_rgb(hex_color):
    """Converts hex color string #RRGGBB to fitz RGB tuple (0-1 range)."""
    hex_color = hex_color.lstrip('#')
    lv = len(hex_color)
    return tuple(int(hex_color[i:i + lv // 3], 16) / 255.0 for i in range(0, lv, lv // 3))

def draw_element_bbox(page: fitz.Page, bbox: Dict[str, float], text: str, color: Tuple[float, float, float], text_color: Tuple[float, float, float], line_width: float = 1, fill_color: Optional[Tuple[float, float, float]] = None):
    """Draws a bounding box and adds text label above it. Optionally fills the box."""
    try:
        rect = fitz.Rect(bbox['x0'], bbox['y0'], bbox['x1'], bbox['y1'])
        page_rect = page.rect
        rect.intersect(page_rect)
        if rect.is_empty or rect.is_infinite:
            return

        page.draw_rect(rect, color=color, fill=fill_color, width=line_width)

        text_pos = fitz.Point(rect.x0, rect.y0 - TEXT_OFFSET)
        if text_pos.y < 5: text_pos.y = rect.y0 + line_width + 1 # Adjust if near top edge
        
        # Simple text wrapping attempt (split by spaces)
        max_chars_per_line = 80 # Adjust as needed
        lines = []
        current_line = ""
        for word in text.split():
            if not current_line:
                current_line = word
            elif len(current_line) + len(word) + 1 < max_chars_per_line:
                current_line += " " + word
            else:
                lines.append(current_line)
                current_line = word
        if current_line:
            lines.append(current_line)
        
        # Draw text line by line
        for i, line in enumerate(lines):
            line_pos = fitz.Point(text_pos.x, text_pos.y + i * (FONTSIZE + 1)) # Adjust line spacing
            # Basic check to avoid drawing text too far down page if wrapped
            if line_pos.y < page_rect.y1 - FONTSIZE:
                 page.insert_text(line_pos, 
                                 line, 
                                 fontsize=FONTSIZE, 
                                 color=text_color)
            else:
                break # Stop drawing if text goes off page

    except Exception as e:
        print(f"Error drawing element with bbox {bbox} and text '{text}': {e}")


def draw_ink_annotation(page: fitz.Page, vertices_list: list, color: Tuple[float, float, float], line_width: float = 1.5):
    """Draws lines based on ink annotation vertices."""
    try:
        for path in vertices_list:
            points = [fitz.Point(p[0], p[1]) for p in path]
            if len(points) > 1:
                page.draw_polyline(points, color=color, width=line_width)
            elif len(points) == 1:
                 # Draw a small circle for a single point (dot)
                 page.draw_circle(points[0], radius=line_width, color=color, fill=color)
    except Exception as e:
        print(f"Error drawing ink annotation: {e}")


def visualize_data_on_pdf(pdf_path: str, json_path: str, output_path: str, draw_text_blocks: bool = False):
    """
    Loads a PDF and its corresponding JSON data, draws visualizations on the PDF pages,
    and saves the result to a new file.
    """
    print(f"Loading PDF: {pdf_path}")
    print(f"Loading JSON data: {json_path}")
    
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        print(f"Error opening PDF '{pdf_path}': {e}")
        return

    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: JSON file not found at '{json_path}'")
        doc.close()
        return
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON file '{json_path}': {e}")
        doc.close()
        return
    except Exception as e:
         print(f"Error loading JSON file '{json_path}': {e}")
         doc.close()
         return

    metadata = data.get("metadata", {})
    pages_data = data.get("pages", [])

    if not pages_data:
        print("Warning: JSON data contains no page information.")
        doc.close()
        return
        
    if metadata.get("total_pages") != len(doc):
         print(f"Warning: Mismatch in page count between PDF ({len(doc)}) and JSON ({metadata.get('total_pages', 'N/A')}). Proceeding anyway.")

    print("Starting visualization...")
    for page_index, page_data in enumerate(pages_data):
        if page_index >= len(doc):
            print(f"Warning: Skipping page data {page_index+1} as it exceeds PDF page count.")
            continue
            
        page_num = page_data.get("page_number", page_index + 1)
        print(f"-- Visualizing Page {page_num} --")
        page = doc[page_index]

        # 1. Visualize Text Blocks (Optional)
        if draw_text_blocks:
            for i, block in enumerate(page_data.get("text_blocks", [])):
                if "bbox" in block:
                    draw_element_bbox(page, block["bbox"], f"TextBlock_{i}", VIZ_COLORS["text_block"], TEXT_COLOR, line_width=1)

        # 2. Visualize Annotations
        for i, annot in enumerate(page_data.get("annotations", [])):
            if "bbox" not in annot: continue
            
            annot_type = annot.get("type", "annotation_default")
            tag = annot.get("semantic_tag", "")
            
            # Extract color info for label and potential fill
            annot_color_info = annot.get("color") or {}
            stroke_color_tuple = annot_color_info.get('stroke')
            fill_color_tuple = annot_color_info.get('fill')
            detected_color = annot.get("detected_color_name") # Get detected color
            
            # Determine outline and fill for drawing
            # Use specific CV color for outline if type is cv_highlight or cv_drawing
            default_color = VIZ_COLORS["annotation_default"]
            if annot_type == "cv_highlight":
                 default_color = VIZ_COLORS.get("cv_highlight")
            elif annot_type == "cv_drawing":
                 default_color = VIZ_COLORS.get("cv_drawing")
                 
            outline_color = VIZ_COLORS.get(annot_type, default_color)
            fill_draw_color = None
            line_draw_width = LINE_WIDTH # Default line width
            
            # Use original fill color for NATIVE highlights if available
            # CV highlights won't have original fill from PyMuPDF
            if annot_type == "highlight" and fill_color_tuple:
                 fill_draw_color = fill_color_tuple
                 # Use a standard, clearer outline when filling highlights
                 outline_color = OUTLINE_COLOR_DEFAULT 
                 line_draw_width = 0.5 # Use thinner line if filling
            elif annot_type != "highlight" and stroke_color_tuple: # For non-highlights, use original stroke if available
                 outline_color = stroke_color_tuple 

            # Format color for label
            color_label = ""
            if stroke_color_tuple:
                color_label += f" S:({stroke_color_tuple[0]:.2f},{stroke_color_tuple[1]:.2f},{stroke_color_tuple[2]:.2f})"
            if fill_color_tuple:
                 color_label += f" F:({fill_color_tuple[0]:.2f},{fill_color_tuple[1]:.2f},{fill_color_tuple[2]:.2f})"
            # Add detected color name to label if present
            if detected_color:
                 color_label += f" [CV: {detected_color}]"
            
            label = f"{annot_type.upper()}{color_label}{f' [{tag}]' if tag else ''}"
            
            # Use the potentially adjusted line width
            draw_element_bbox(page, annot["bbox"], label, outline_color, TEXT_COLOR, line_width=line_draw_width, fill_color=fill_draw_color)

            if annot_type == "ink" and annot.get("vertices"):
                ink_draw_color = stroke_color_tuple if stroke_color_tuple else VIZ_COLORS["ink"]
                draw_ink_annotation(page, annot["vertices"], ink_draw_color, INK_LINE_WIDTH)

        # 3. Visualize Visual Elements (Figures, Drawings)
        for i, element in enumerate(page_data.get("visual_elements", [])):
             if "bbox" not in element: continue
             
             element_type = element.get("type", "figure")
             color = VIZ_COLORS.get(element_type, VIZ_COLORS["figure"])
             desc = element.get("vlm_description")
             safe_desc = desc if desc is not None else ""
             # Show more description, let wrapping handle overflow
             label = f"{element_type.upper()}: {safe_desc}" 
             
             draw_element_bbox(page, element["bbox"], label, color, TEXT_COLOR, line_width=LINE_WIDTH)

        # 4. Visualize Equations
        for i, eqn in enumerate(page_data.get("equations", [])):
             if "bbox" not in eqn: continue

             color = VIZ_COLORS.get("equation", (0.2, 0.5, 0.7))
             transcription = eqn.get("vlm_transcription", "[VLM Transcription Pending or Failed]")
             label = f"EQUATION: {transcription}" # Let wrapping handle length

             draw_element_bbox(page, eqn["bbox"], label, color, TEXT_COLOR, line_width=LINE_WIDTH)

    print(f"Saving visualized PDF to: {output_path}")
    try:
        # Save with garbage collection and compression for smaller file size
        doc.save(output_path, garbage=4, deflate=True)
        print("Visualization complete.")
    except Exception as e:
        print(f"Error saving output PDF '{output_path}': {e}")
    finally:
        doc.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize processed JSON data onto the original PDF.")
    parser.add_argument("--pdf_path", required=True, help="Path to the original input PDF file.")
    parser.add_argument("--json_path", required=True, help="Path to the processed JSON file corresponding to the PDF.")
    parser.add_argument("--output_path", help="Path to save the visualized output PDF. Defaults to '[pdf_path]_visualized.pdf'.")
    parser.add_argument("--draw_text", action="store_true", help="Also draw bounding boxes for text blocks (can be very cluttered).")

    args = parser.parse_args()

    output_pdf_path = args.output_path
    if not output_pdf_path:
        pdf_dir = os.path.dirname(args.pdf_path)
        base_name = os.path.basename(args.pdf_path)
        file_name_without_ext = os.path.splitext(base_name)[0]
        output_pdf_path = os.path.join(pdf_dir, f"{file_name_without_ext}_visualized.pdf")
        # If original PDF was in Input_files, maybe save viz to output dir?
        # Example: output_pdf_path = os.path.join("output", f"{file_name_without_ext}_visualized.pdf")

    visualize_data_on_pdf(args.pdf_path, args.json_path, output_pdf_path, args.draw_text)
