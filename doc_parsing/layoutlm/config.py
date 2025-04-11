import os
from dotenv import load_dotenv

# Load environment variables from a .env file if it exists
load_dotenv()

# --- Model Configuration ---
LAYOUTLM_MODEL_NAME = "microsoft/layoutlmv3-base"
# Specify the GPT-4 VLM model name if needed, or rely on OpenAI library defaults
# GPT4_VLM_MODEL_NAME = "gpt-4-vision-preview" # Example

# --- API Keys ---
# It's crucial to set the OPENAI_API_KEY environment variable
# You can set it in your system environment or create a .env file
# in the doc_parsing/layoutlm/ directory with the following content:
# OPENAI_API_KEY='your_api_key_here'
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    print("Warning: OPENAI_API_KEY environment variable not set. GPT-4 VLM features will not work.")
    # Depending on requirements, you might want to raise an error here instead:
    # raise ValueError("OPENAI_API_KEY environment variable is required for VLM features.")

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

# You can add other configuration variables here as needed
