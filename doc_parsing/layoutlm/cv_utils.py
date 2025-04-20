import cv2
import numpy as np
from PIL import Image
import io
from typing import List, Dict, Tuple, Optional

# Define HSV color range for typical yellow highlights
# These ranges might need adjustment based on the specific yellow shade
DEFAULT_YELLOW_HSV_LOWER = np.array([20, 100, 100])
DEFAULT_YELLOW_HSV_UPPER = np.array([30, 255, 255])

# Morphology kernel size
MORPH_KERNEL_SIZE = (5, 5)
# Minimum contour area to consider (adjust based on expected highlight size)
MIN_CONTOUR_AREA = 100
# Parameters for merging highlight fragments
MAX_HORIZONTAL_GAP_RATIO = 0.25 # Relaxed: Allow larger horizontal gap
MIN_VERTICAL_OVERLAP_RATIO = 0.3 # Relaxed: Require less vertical overlap
# Parameters for Ink Detection
INK_MORPH_KERNEL_SIZE = (3, 3) # Smaller kernel for thinner lines
INK_MIN_CONTOUR_AREA = 30    # Reduced minimum area for ink
INK_MAX_ASPECT_RATIO_DEVIATION = 0.4 # How much box aspect ratio can deviate from contour's minAreaRect ratio (filter thick blobs)
INK_MIN_EXTENT = 0.05 # Relaxed: Allow less dense contours
# Morphology settings specifically for HIGHLIGHTS
HIGHLIGHT_DILATE_ITERATIONS = 5
HIGHLIGHT_ERODE_ITERATIONS = 1 # Keep some erosion to reduce noise after heavy dilation
HIGHLIGHT_KERNEL_SHAPE = (15, 3) # Rectangular kernel: Wide to connect horizontally

def pil_to_cv2(pil_image: Image.Image) -> np.ndarray:
    """Converts a PIL Image (RGB) to an OpenCV image (BGR)."""
    # Convert PIL image to numpy array (RGB)
    rgb_image = np.array(pil_image)
    # Convert RGB to BGR for OpenCV
    bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    return bgr_image

def merge_boxes(box1: Dict[str, float], box2: Dict[str, float]) -> Dict[str, float]:
    """Merges two bounding boxes into one encompassing box."""
    x0 = min(box1['x0'], box2['x0'])
    y0 = min(box1['y0'], box2['y0'])
    x1 = max(box1['x1'], box2['x1'])
    y1 = max(box1['y1'], box2['y1'])
    return {"x0": x0, "y0": y0, "x1": x1, "y1": y1}

def should_merge(box1: Dict[str, float], box2: Dict[str, float], 
                 max_gap_ratio: float = MAX_HORIZONTAL_GAP_RATIO, 
                 min_overlap_ratio: float = MIN_VERTICAL_OVERLAP_RATIO) -> bool:
    """Checks if two boxes should be merged based on proximity and overlap."""
    # Calculate dimensions
    h1 = box1['y1'] - box1['y0']
    w1 = box1['x1'] - box1['x0']
    h2 = box2['y1'] - box2['y0']
    w2 = box2['x1'] - box2['x0']
    if h1 <= 0 or w1 <= 0 or h2 <= 0 or w2 <= 0: return False

    # Vertical overlap
    overlap_y = max(0, min(box1['y1'], box2['y1']) - max(box1['y0'], box2['y0']))
    if overlap_y / min(h1, h2) < min_overlap_ratio:
        return False # Not enough vertical overlap

    # Horizontal gap
    gap_x = max(0, max(box1['x0'], box2['x0']) - min(box1['x1'], box2['x1']))
    avg_width = (w1 + w2) / 2
    if gap_x / avg_width > max_gap_ratio:
        return False # Horizontal gap too large

    return True # Conditions met for merging

def detect_color_highlight_regions(
    pil_image: Image.Image, 
    hsv_lower: np.ndarray = DEFAULT_YELLOW_HSV_LOWER, 
    hsv_upper: np.ndarray = DEFAULT_YELLOW_HSV_UPPER,
    min_area: int = MIN_CONTOUR_AREA
) -> List[Dict[str, float]]:
    """
    Detects regions of a specific color, merges nearby fragments typical of highlights,
    and returns their bounding boxes.

    Args:
        pil_image: The input PIL Image object.
        hsv_lower: Lower bound of the target color in HSV space.
        hsv_upper: Upper bound of the target color in HSV space.
        min_area: Minimum contour area to be considered a valid region.

    Returns:
        A list of dictionaries, where each dictionary represents a bounding box
        ({'x0': ..., 'y0': ..., 'x1': ..., 'y1': ...}) in the original
        PIL image coordinates.
    """
    initial_regions = []
    if pil_image is None: return initial_regions

    try:
        # Image conversion, HSV masking
        cv2_image = pil_to_cv2(pil_image)
        hsv_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv_image, hsv_lower, hsv_upper)
        
        # --- Apply Morphology Optimized for Highlights ---
        # Use a rectangular kernel biased towards horizontal connection
        highlight_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, HIGHLIGHT_KERNEL_SHAPE)
        # Dilate significantly to bridge gaps within highlighted text lines
        mask_dilated = cv2.dilate(mask, highlight_kernel, iterations=HIGHLIGHT_DILATE_ITERATIONS)
        # Erode slightly to clean up noise expanded by dilation
        mask_eroded = cv2.erode(mask_dilated, highlight_kernel, iterations=HIGHLIGHT_ERODE_ITERATIONS)
        cleaned_mask = mask_eroded
        # --- End Morphology ---
        
        contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Get initial bounding boxes for significant contours
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > min_area:
                x, y, w, h = cv2.boundingRect(contour)
                initial_regions.append({"x0": float(x), "y0": float(y), "x1": float(x+w), "y1": float(y+h)})
                
                # --- Debug Drawing (Optional) ---
                # To visualize detected contours during debugging:
                # cv2.rectangle(cv2_image, (x, y), (x + w, y + h), (0, 255, 0), 2) # Draw green box
                # cv2.drawContours(cv2_image, [contour], -1, (255, 0, 0), 1) # Draw blue contour
                # --- End Debug --- 

        if not initial_regions: return []

        # Merge overlapping/nearby bounding boxes iteratively
        merged_regions = initial_regions.copy()
        while True:
            merged_this_pass = False
            new_merged_regions = []
            merged_indices = set()
            merged_regions.sort(key=lambda b: (b['y0'], b['x0']))
            i = 0
            while i < len(merged_regions):
                if i in merged_indices: i += 1; continue
                current_box = merged_regions[i]
                best_merge_j = -1
                merged_box_candidate = None
                j = i + 1
                while j < len(merged_regions):
                    if j in merged_indices: j += 1; continue
                    other_box = merged_regions[j]
                    if other_box['y0'] > current_box['y1'] + (current_box['y1']-current_box['y0']): pass 
                    if should_merge(current_box, other_box):
                        merged_box_candidate = merge_boxes(current_box, other_box)
                        best_merge_j = j
                        merged_this_pass = True
                        break
                    j += 1
                if best_merge_j != -1:
                    new_merged_regions.append(merged_box_candidate)
                    merged_indices.add(i)
                    merged_indices.add(best_merge_j)
                    i += 1 
                else:
                    new_merged_regions.append(current_box)
                    merged_indices.add(i)
                    i += 1
            merged_regions = new_merged_regions
            if not merged_this_pass: break
        final_regions = merged_regions

    except Exception as e:
        print(f"Error during OpenCV color detection/merging: {e}")
        return initial_regions # Return unmerged if error

    return final_regions

def detect_ink_regions(
    pil_image: Image.Image, 
    hsv_lower1: np.ndarray, # First range (e.g., low hues for red)
    hsv_upper1: np.ndarray,
    hsv_lower2: Optional[np.ndarray] = None, # Optional second range (e.g., high hues for red)
    hsv_upper2: Optional[np.ndarray] = None,
    min_area: int = INK_MIN_CONTOUR_AREA,
    min_extent: float = INK_MIN_EXTENT
) -> List[Dict[str, float]]:
    """
    Detects thin regions of a specific color, filters them, and merges nearby fragments.

    Args:
        pil_image: Input PIL Image.
        hsv_lower1, hsv_upper1: Primary HSV color range.
        hsv_lower2, hsv_upper2: Optional secondary HSV range (for colors like red).
        min_area: Minimum contour area.
        min_extent: Minimum ratio of contour area to bounding box area.

    Returns:
        List of bounding box dictionaries for detected ink regions.
    """
    initial_regions = []
    if pil_image is None: return initial_regions

    try:
        cv2_image = pil_to_cv2(pil_image)
        hsv_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2HSV)
        mask1 = cv2.inRange(hsv_image, hsv_lower1, hsv_upper1)
        if hsv_lower2 is not None and hsv_upper2 is not None:
            mask2 = cv2.inRange(hsv_image, hsv_lower2, hsv_upper2)
            mask = cv2.bitwise_or(mask1, mask2)
        else:
            mask = mask1
        kernel = np.ones(INK_MORPH_KERNEL_SIZE, np.uint8)
        mask_cleaned = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2) # Slightly stronger closing
        contours, _ = cv2.findContours(mask_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            bbox_area = w * h
            if bbox_area <= 0: continue
            extent = float(area) / bbox_area
            if extent < min_extent: # Relaxed extent filter
                 continue 
            # Optional thinness filter is still commented out
            initial_regions.append({"x0": float(x), "y0": float(y), "x1": float(x + w), "y1": float(y + h)})
            
        if not initial_regions: return []
            
        # --- Apply Merging Logic to Ink Regions --- 
        merged_regions = initial_regions.copy()
        while True:
            merged_this_pass = False
            new_merged_regions = []
            merged_indices = set()
            merged_regions.sort(key=lambda b: (b['y0'], b['x0']))
            i = 0
            while i < len(merged_regions):
                if i in merged_indices: i += 1; continue
                current_box = merged_regions[i]
                best_merge_j = -1
                merged_box_candidate = None
                j = i + 1
                while j < len(merged_regions):
                    if j in merged_indices: j += 1; continue
                    other_box = merged_regions[j]
                    # Use slightly relaxed merging criteria for ink?
                    if should_merge(current_box, other_box, 
                                    max_gap_ratio=MAX_HORIZONTAL_GAP_RATIO + 0.1, # Allow slightly larger gap for ink
                                    min_overlap_ratio=MIN_VERTICAL_OVERLAP_RATIO - 0.1): # Allow slightly less overlap
                        merged_box_candidate = merge_boxes(current_box, other_box)
                        best_merge_j = j
                        merged_this_pass = True
                        break
                    j += 1
                if best_merge_j != -1:
                    new_merged_regions.append(merged_box_candidate)
                    merged_indices.add(i)
                    merged_indices.add(best_merge_j)
                    i += 1 
                else:
                    new_merged_regions.append(current_box)
                    merged_indices.add(i)
                    i += 1
            merged_regions = new_merged_regions
            if not merged_this_pass: break
        # --- End Merging Logic ---
        
        final_regions = merged_regions

    except Exception as e:
        print(f"Error during OpenCV ink detection/merging: {e}")
        return initial_regions # Return unmerged if error

    return final_regions

# Example usage (for testing this module directly)
if __name__ == '__main__':
    print("CV Utils - Example Color Detection (requires a test image)")
    # Create a dummy image with a yellow rectangle
    width, height = 400, 300
    dummy_pil = Image.new('RGB', (width, height), color='white')
    # Draw a yellow rectangle
    from PIL import ImageDraw
    draw = ImageDraw.Draw(dummy_pil)
    # Approximate yellow coordinates
    rect_coords = [50, 50, 250, 150] 
    draw.rectangle(rect_coords, fill='yellow')
    # Draw some noise
    draw.rectangle([300, 200, 310, 210], fill='yellow') # Small area
    draw.rectangle([10, 10, 30, 30], fill=(255, 255, 100)) # Slightly different yellow

    print("Detecting yellow regions...")
    detected_boxes = detect_color_highlight_regions(dummy_pil)

    if detected_boxes:
        print(f"Detected {len(detected_boxes)} regions:")
        for i, box in enumerate(detected_boxes):
            print(f"  Region {i}: {box}")
    else:
        print("No regions detected.")

    # Optional: Display the dummy image with detected boxes (requires OpenCV)
    # cv2_dummy = pil_to_cv2(dummy_pil)
    # for box in detected_boxes:
    #     cv2.rectangle(cv2_dummy, (int(box['x0']), int(box['y0'])), (int(box['x1']), int(box['y1'])), (0, 255, 0), 2)
    # cv2.imshow("Dummy Image with Detections", cv2_dummy)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows() 