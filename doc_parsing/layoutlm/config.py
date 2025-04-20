import os
from dotenv import load_dotenv
import numpy as np

# Load environment variables from a .env file if it exists
load_dotenv()

# --- Model Configuration ---
LAYOUTLM_MODEL_NAME = "microsoft/layoutlmv3-base"
# Specify the GPT-4 VLM model name if needed, or rely on OpenAI library defaults
# GPT4_VLM_MODEL_NAME = "gpt-4-vision-preview" # Example

# --- API Keys ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    print("Warning: OPENAI_API_KEY environment variable not set. GPT-4 VLM features will not work.")

# --- Processing Configuration ---
# PDF_INPUT_PATH = "doc_parsing/layoutlm/Multimodal_Interfaces_A_Survey_of_Principles_Model-1 (1).pdf" # Removed, will be CLI arg
OUTPUT_DIR = "doc_parsing/layoutlm/output"
# PROCESSED_JSON_FILENAME = "processed_document.json" # Removed, will be generated dynamically
LAYOUTLM_IMAGE_SIZE = (224, 224) # Expected input size for LayoutLMv3 image features

# --- VLM Configuration ---
# Heuristics for identifying complex regions (example thresholds)
MIN_IMAGE_AREA_FOR_VLM = 5000 # Minimum pixel area for an image to be considered for VLM
VLM_PROMPT_DESCRIPTION = "Describe the content of this image region in detail."
VLM_PROMPT_CHART_EXTRACTION = "Extract the data points or key information from this chart/table."
VLM_PROMPT_HANDWRITING = "Transcribe the handwritten text in this image."

# --- Output Configuration ---
JSON_INDENT = 4


ANNOTATION_SEMANTIC_MAP = {
    # Color-based rules (using 'stroke' color for highlights/underlines)
    # Example: Bright Yellow for 'important'
    (0.99, 0.99, 0.0): "important",
    # Example: Bright Pink/Magenta for 'definition'
    (0.99, 0.0, 0.99): "definition",
    # Example: Bright Cyan for 'question'
    (0.0, 0.99, 0.99): "question",
    # Example: Green for 'example'
     (0.0, 0.8, 0.0): "example",

    # Type-based rules (can add more complex logic)
    # "comment": "note", # Maybe tag all comments as 'note' by default?

    # Symbol/Text-based rules (requires more logic in the processor)
    # "**": "very_important" # Example for asterisks
}

# --- CV Highlight Detection Configuration ---
# Map color names to their HSV lower and upper bounds
# Adjust these ranges carefully for your specific highlight colors!
CV_HIGHLIGHT_HSV_RANGES = {
    "yellow": {
        "lower": np.array([20, 100, 100]),
        "upper": np.array([30, 255, 255])
    },
    "pink": { # Example for pink/magenta
        "lower": np.array([140, 100, 100]),
        "upper": np.array([170, 255, 255])
    },
    "blue": { # Example for cyan/light blue
        "lower": np.array([85, 100, 100]),
        "upper": np.array([110, 255, 255])
    },
    "green": { # Example for green
        "lower": np.array([40, 100, 100]),
        "upper": np.array([75, 255, 255])
    },
    "orange": { # Example for orange
        "lower": np.array([5, 100, 100]),
        "upper": np.array([20, 255, 255])
    },
    "red": { # Example for typical red ink
        # Red wraps around HSV 0/180. Need two ranges.
        # Range 1: Low hues (0-10)
        "lower1": np.array([0, 100, 100]), 
        "upper1": np.array([10, 255, 255]),
        # Range 2: High hues (160-180)
        "lower2": np.array([160, 100, 100]),
        "upper2": np.array([180, 255, 255])
    },
    # Add more colors as needed
}
