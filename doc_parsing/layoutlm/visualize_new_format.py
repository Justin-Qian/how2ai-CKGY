import fitz  # PyMuPDF
import json
import os
import argparse
from typing import Dict, Any, Tuple, Optional, List

# --- Visualization Configuration ---
VIZ_COLORS = {
    "document": (0.5, 0.5, 0.5),       # Gray
    "section": (0.0, 0.2, 0.6),        # Dark blue
    "paragraph": (0.0, 0.5, 0.2),      # Green
    "annotation_default": (1, 0, 0),   # Red
    "highlight": (0.9, 0.9, 0),        # Yellow
    "comment": (0, 0.7, 0.7),          # Teal
    "symbol": (1.0, 0.0, 0.0),         # Red
    "visual_element": (0.0, 0.3, 0.8), # Blue
    "equation": (0.8, 0.4, 0.0),       # Orange
    "ink": (0.8, 0.0, 0.8),            # Purple
}

# Add corresponding fill colors with transparency for highlighting
VIZ_FILL_COLORS = {
    "highlight": (0.9, 0.9, 0, 0.3),     # Yellow with transparency
    "comment": (0, 0.7, 0.7, 0.2),       # Teal with transparency
    "symbol": (1.0, 0.0, 0.0, 0.1),      # Red with transparency
    "visual_element": (0.0, 0.3, 0.8, 0.2),  # Blue with transparency
    "equation": (0.8, 0.4, 0.0, 0.15),   # Orange with transparency
    "ink": (0.8, 0.0, 0.8, 0.1),         # Purple with transparency
}

OUTLINE_COLOR_DEFAULT = (0.2, 0.2, 0.2)  # Dark gray outline
TEXT_COLOR = (0, 0, 0)  # Black
LINE_WIDTH = 1.0  
FONTSIZE = 6  
TEXT_OFFSET = FONTSIZE + 2  

def hex_to_rgb(hex_color):
    """Convert hexadecimal color string #RRGGBB to fitz RGB tuple (0-1 range)."""
    hex_color = hex_color.lstrip('#')
    lv = len(hex_color)
    return tuple(int(hex_color[i:i + lv // 3], 16) / 255.0 for i in range(0, lv, lv // 3))

def draw_element_bbox(page: fitz.Page, bbox: Dict[str, float], text: str, color: Tuple[float, float, float], text_color: Tuple[float, float, float], line_width: float = 1, fill_color: Optional[Tuple[float, float, float, float]] = None, semantic_tag: Optional[str] = None):
    """Draw a bounding box and add text label above it. Optionally fill the box with transparent color."""
    try:
        rect = fitz.Rect(bbox['x0'], bbox['y0'], bbox['x1'], bbox['y1'])
        page_rect = page.rect
        rect.intersect(page_rect)
        if rect.is_empty or rect.is_infinite:
            return

        # If we have a semantic tag, check if we should use a specific fill color
        if semantic_tag in ["visual_element", "equation"] and not fill_color:
            fill_color = VIZ_FILL_COLORS.get(semantic_tag)
        
        # Draw the rectangle with or without fill
        if fill_color:
            # For filled rectangles, use a fill color with transparency
            if len(fill_color) == 4:  # RGBA
                opacity = fill_color[3]
                rgb_color = fill_color[:3]
                # First draw the fill with transparency
                # Modified to use shape instead of draw_rect with opacity
                shape = page.new_shape()
                shape.draw_rect(rect)
                shape.finish(fill=rgb_color, fill_opacity=opacity, color=color, width=line_width)
                shape.commit()
            else:
                # Use regular draw_rect for RGB without alpha
                page.draw_rect(rect, color=color, fill=fill_color, width=line_width)
        else:
            page.draw_rect(rect, color=color, width=line_width)

        # Text placement logic
        text_pos = fitz.Point(rect.x0, rect.y0 - TEXT_OFFSET)
        if text_pos.y < 5: 
            text_pos.y = rect.y0 + line_width + 1  # Adjust if near top edge
        
        # Simple text wrapping (split by spaces)
        max_chars_per_line = 80  # Adjust as needed
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
            line_pos = fitz.Point(text_pos.x, text_pos.y + i * (FONTSIZE + 1))  # Adjust line spacing
            # Basic check to avoid drawing text too far down page if wrapped
            if line_pos.y < page_rect.y1 - FONTSIZE:
                 page.insert_text(line_pos, 
                                 line, 
                                 fontsize=FONTSIZE, 
                                 color=text_color)
            else:
                break  # Stop drawing if text goes off page

    except Exception as e:
        print(f"Error drawing element with bbox {bbox} and text '{text}': {e}")

def find_page_for_position(char_position: int, paragraphs: List[Dict], pdf_doc: fitz.Document) -> int:
    """查找字符位置对应的PDF页面。"""
    # 首先找到包含该位置的段落
    containing_paragraph = None
    for paragraph in paragraphs:
        start = paragraph.get("character_start", 0)
        end = paragraph.get("character_end", 0)
        if start <= char_position <= end:
            containing_paragraph = paragraph
            break
    
    if not containing_paragraph or "bbox" not in containing_paragraph:
        return 0  # 默认返回第一页
    
    # 使用包含段落的边界框找到页面
    bbox = containing_paragraph["bbox"]
    
    # 遍历所有页面，查找包含该边界框的页面
    for page_num in range(len(pdf_doc)):
        page = pdf_doc[page_num]
        page_rect = page.rect
        if (bbox["x0"] >= 0 and bbox["x0"] <= page_rect.width and 
            bbox["y0"] >= 0 and bbox["y0"] <= page_rect.height):
            return page_num
    
    return 0  # 默认返回第一页

def draw_ink_annotation(page: fitz.Page, vertices_list: list, color: Tuple[float, float, float], line_width: float = 1.5):
    """
    Draw ink annotation consisting of one or multiple paths (strokes) on the page.
    Each path is a list of (x, y) coordinates.
    """
    try:
        if not vertices_list:
            return
            
        for path in vertices_list:
            if not path or len(path) < 2:
                continue
            
            # Draw each stroke
            for i in range(len(path) - 1):
                p1 = fitz.Point(path[i][0], path[i][1])
                p2 = fitz.Point(path[i+1][0], path[i+1][1])
                
                # Check if points are valid
                if (0 <= p1.x <= page.rect.width and 0 <= p1.y <= page.rect.height and
                    0 <= p2.x <= page.rect.width and 0 <= p2.y <= page.rect.height):
                    page.draw_line(p1, p2, color=color, width=line_width)
    except Exception as e:
        print(f"Error drawing ink annotation: {e}")

def visualize_new_format_on_pdf(pdf_path: str, json_path: str, output_path: str, show_sections: bool = True, show_paragraphs: bool = True):
    """
    Load a PDF and its corresponding new format JSON data, draw visualizations on the PDF pages,
    and save the result to a new file.
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

    # Get main data components
    document = data.get("document", {})
    sections = {section["id"]: section for section in data.get("sections", [])}
    paragraphs = {para["id"]: para for para in data.get("paragraphs", [])}
    annotations = {annot["id"]: annot for annot in data.get("annotations", [])}
    
    if not document:
        print("Warning: JSON data does not contain document information.")
        doc.close()
        return
        
    print("Starting visualization...")
    
    # Track which pages have been processed
    processed_pages = set()

    # First process and visualize annotations by type
    print("Processing annotations...")
    for annot_id, annot in annotations.items():
        if "bbox" not in annot and "vertices" not in annot:
            continue
            
        annot_type = annot.get("type", "annotation_default")
        para_id = annot.get("paragraph_id")
        semantic_tag = annot.get("semantic_tag", None)
        
        # Find the paragraph this annotation belongs to
        para = paragraphs.get(para_id)
        if not para:
            print(f"  No paragraph found for annotation {annot_id}")
            continue
            
        # Use the page number from the paragraph
        page_index = para.get("page_number", 0)
        if page_index is None or page_index >= len(doc):
            print(f"  Invalid page number {page_index} for annotation {annot_id}")
            continue
            
        processed_pages.add(page_index)
        page = doc[page_index]
        
        # Special handling for ink annotations with vertices
        if annot_type == "ink" and "vertices" in annot and annot["vertices"]:
            # Determine color
            color_str = annot.get("color", "")
            if color_str and color_str.startswith("#"):
                ink_color = hex_to_rgb(color_str)
            else:
                ink_color = VIZ_COLORS.get("ink", VIZ_COLORS["annotation_default"])
                
            # Draw the ink strokes
            draw_ink_annotation(page, annot["vertices"], ink_color, line_width=1.5)
            
            # Add a label for the ink annotation
            if "bbox" in annot and annot["bbox"]:
                label = f"INK ANNOTATION [{annot_id}]"
                draw_element_bbox(
                    page, 
                    annot["bbox"], 
                    label, 
                    VIZ_COLORS["ink"],
                    TEXT_COLOR, 
                    line_width=LINE_WIDTH,
                )
            print(f"  Drew ink annotation {annot_id} on page {page_index+1}")
            continue
            
        # For all other annotation types
        if "bbox" not in annot or not annot["bbox"]:
            continue
            
        # Determine outline color based on type and semantic tag
        color_str = annot.get("color", "")
        if color_str and color_str.startswith("#"):
            outline_color = hex_to_rgb(color_str)
        else:
            if semantic_tag and semantic_tag in VIZ_COLORS:
                outline_color = VIZ_COLORS[semantic_tag]
            elif annot_type in VIZ_COLORS:
                outline_color = VIZ_COLORS[annot_type]
            else:
                outline_color = VIZ_COLORS["annotation_default"]
        
        # Determine fill color based on type and semantic tag
        fill_color = None
        if semantic_tag and semantic_tag in VIZ_FILL_COLORS:
            fill_color = VIZ_FILL_COLORS[semantic_tag]
        elif annot_type in VIZ_FILL_COLORS:
            fill_color = VIZ_FILL_COLORS[annot_type]
        
        # Build label with more details
        referenced_text = annot.get("referenced_text", "")
        if annot_type == "visual_element":
            label = f"VISUAL: {semantic_tag or 'element'}"
            if referenced_text:
                label += f": {referenced_text[:30]}"
                if len(referenced_text) > 30:
                    label += "..."
        elif annot_type == "equation":
            label = f"EQUATION [{annot_id}]"
            if referenced_text:
                label += f": {referenced_text[:30]}"
                if len(referenced_text) > 30:
                    label += "..."
        else:
            label = f"{annot_type.upper()}"
            if semantic_tag:
                label += f" [{semantic_tag}]"
            label += f" [{annot_id}]: {referenced_text[:50]}"
            if len(referenced_text) > 50:
                label += "..."
            
        # Draw annotation with appropriate fill color
        draw_element_bbox(
            page, 
            annot["bbox"], 
            label, 
            outline_color, 
            TEXT_COLOR, 
            line_width=LINE_WIDTH, 
            fill_color=fill_color,
            semantic_tag=semantic_tag
        )
        print(f"  Drew annotation {annot_id} on page {page_index+1}")
    
    # Visualize paragraphs - but skip those that are represented by annotations already
    if show_paragraphs:
        print("Processing paragraphs...")
        for para_id, para in paragraphs.items():
            # Skip paragraphs that have annotations
            if para.get("annotations"):
                continue
                
            if "bbox" not in para:
                continue
                
            # Get page number
            page_index = para.get("page_number", 0)
            if page_index is None or page_index >= len(doc):
                print(f"  Invalid page number {page_index} for paragraph {para_id}")
                continue
                
            processed_pages.add(page_index)
            page = doc[page_index]
            
            # Build label
            text = para.get("text", "")
            label = f"PARA [{para_id}]: {text[:30]}"
            if len(text) > 30:
                label += "..."
                
            # Draw paragraph
            draw_element_bbox(page, para["bbox"], label, VIZ_COLORS["paragraph"], TEXT_COLOR, line_width=LINE_WIDTH)
            print(f"  Drew paragraph {para_id} on page {page_index+1}")
    
    # Visualize sections (if needed and possible)
    if show_sections:
        print("Processing sections...")
        for section_id, section in sections.items():
            # Sections usually don't have bbox, determine via paragraphs
            section_paras = section.get("paragraphs", [])
            if not section_paras:
                continue
                
            # Use first paragraph to determine visual position
            first_para_id = section_paras[0]
            first_para = paragraphs.get(first_para_id)
            if not first_para or "bbox" not in first_para:
                continue
                
            # Get page number from paragraph
            page_index = first_para.get("page_number", 0)
            if page_index is None or page_index >= len(doc):
                print(f"  Invalid page number {page_index} for section {section_id}")
                continue
                
            processed_pages.add(page_index)
            page = doc[page_index]
            
            # Build label
            heading = section.get("heading", "")
            label = f"SECTION [{section_id}]: {heading}"
            
            # Draw section (using first paragraph's bbox)
            draw_element_bbox(page, first_para["bbox"], label, VIZ_COLORS["section"], TEXT_COLOR, line_width=LINE_WIDTH)
            print(f"  Drew section {section_id} on page {page_index+1}")

    # Print summary of processed pages
    if processed_pages:
        print(f"Processed {len(processed_pages)} pages: {sorted(processed_pages)}")
        print(f"Saving visualized PDF to: {output_path}")
        
        try:
            # Save with garbage collection and compression for smaller file size
            doc.save(output_path, garbage=4, deflate=True)
            print("Visualization complete.")
        except Exception as e:
            print(f"Error saving output PDF '{output_path}': {e}")
    else:
        print("No pages were processed. Check if the JSON data contains valid annotations, paragraphs, or sections.")
    
    doc.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize processed new format JSON data on the original PDF.")
    parser.add_argument("--pdf_path", required=True, help="Path to the original input PDF file.")
    parser.add_argument("--json_path", required=True, help="Path to the processed new format JSON file.")
    parser.add_argument("--output_path", help="Path to save the visualized output PDF. Defaults to '[pdf_path]_new_visualized.pdf'.")
    parser.add_argument("--hide_sections", action="store_true", help="Do not show section boundaries.")
    parser.add_argument("--hide_paragraphs", action="store_true", help="Do not show paragraph boundaries.")

    args = parser.parse_args()

    output_pdf_path = args.output_path
    if not output_pdf_path:
        pdf_dir = os.path.dirname(args.pdf_path)
        base_name = os.path.basename(args.pdf_path)
        file_name_without_ext = os.path.splitext(base_name)[0]
        output_pdf_path = os.path.join(pdf_dir, f"{file_name_without_ext}_new_visualized.pdf")

    visualize_new_format_on_pdf(
        args.pdf_path, 
        args.json_path, 
        output_pdf_path, 
        show_sections=not args.hide_sections,
        show_paragraphs=not args.hide_paragraphs
    )