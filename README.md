# Multimodal Research Paper Annotation Extractor

This project provides a pipeline to extract rich information from annotated research paper PDFs, combining layout analysis, text extraction, annotation detection (native and CV-based), and Vision Language Model (VLM) analysis for figures, drawings, and equations.

## Overview

The system processes PDF documents page by page to extract:

*   **Text Content & Layout:** Using PyMuPDF for initial text/word extraction and bounding boxes.
*   **Layout Features:** Employing `microsoft/layoutlmv3-base` (via Hugging Face `transformers`) to understand document layout (though embeddings are not stored in the final JSON by default).
*   **Native PDF Annotations:** Extracting standard annotations like highlights, underlines, comments, ink strokes, etc., directly from the PDF structure using PyMuPDF. Captures type, bounding box, associated text, color information (stroke/fill), and vertices (for ink).
*   **CV-Detected Annotations:** For PDFs where annotations are not native (e.g., flattened or drawn highlights/ink), uses OpenCV (`cv2`) to visually detect regions based on color (e.g., yellow highlights, red ink). Stores bounding box, associated text, and detected color name.
*   **Visual Elements:** Identifying figures and drawings using heuristics based on image info and vector graphics extracted by PyMuPDF.
*   **Equations:** Identifying potential equation regions using layout and symbol heuristics.
*   **VLM Analysis:** Using OpenAI's GPT-4o API (via `asyncio` for concurrency) to:
    *   Generate descriptions for detected figures and drawings.
    *   Transcribe detected equations (attempting LaTeX format).
*   **Structured Output:** Saving all extracted information into a detailed JSON file per processed PDF.
*   **Visualization:** Providing a script (`visualize_output.py`) to draw the extracted bounding boxes, labels, VLM descriptions, and annotations onto a copy of the original PDF for review.

## Features

*   Extracts text blocks with bounding boxes.
*   Extracts native annotations (highlights, comments, ink, etc.) with metadata (color, vertices).
*   Detects non-native highlights and ink strokes using OpenCV color detection.
*   Merges fragmented CV-detected highlights and ink strokes for better continuity.
*   Identifies potential figure, drawing, and equation regions.
*   Uses GPT-4o for rich descriptions of visual elements and transcription of equations.
*   Filters VLM refusal messages (e.g., "I cannot...").
*   Outputs structured JSON data.
*   Provides a visualization script to overlay extracted data onto the PDF.
*   Leverages GPU for LayoutLM inference if PyTorch with CUDA is correctly installed.

## Setup

1.  **Clone the Repository:**
    ```bash
    git clone <your-repo-url>
    cd <your-repo-directory>/doc_parsing/layoutlm 
    ```
2.  **Create Virtual Environment:** (Recommended)
    ```bash
    python -m venv .venv 
    ```
3.  **Activate Environment:**
    *   Windows (PowerShell): `.\.venv\Scripts\Activate.ps1`
    *   macOS/Linux: `source .venv/bin/activate`
4.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *   **GPU Support (Recommended):** For significantly faster processing. Follow instructions on the [official PyTorch website](https://pytorch.org/get-started/locally/) to install the correct version matching your CUDA toolkit (e.g., `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121`).

5.  **Set up OpenAI API Key:**
    *   Create a file named `.env` in the `doc_parsing/layoutlm/` directory.
    *   Add your OpenAI API key to the `.env` file:
        ```dotenv
        OPENAI_API_KEY="your_actual_openai_api_key_here"
        ```
    *   **Important:** Ensure `.env` is listed in your `.gitignore` file to avoid committing your key.

## Usage

Run scripts from the **workspace root directory** (`How2AI/how2ai-CKGY/` or equivalent) using the module flag (`-m`) to ensure relative imports work correctly.

1.  **Processing a PDF:**
    ```bash
    python -m doc_parsing.layoutlm.document_processor --pdf_path path/to/your/input_document.pdf
    ```
    *   Replace `path/to/your/input_document.pdf` with the actual path to the PDF you want to process (e.g., `doc_parsing/layoutlm/PDF_data/NLP/your_paper.pdf`).
    *   This will generate a `_processed.json` file in the `doc_parsing/layoutlm/output/` directory containing the extracted data.

2.  **Visualizing the Output:**
    ```bash
    python -m doc_parsing.layoutlm.visualize_output --pdf_path path/to/your/original.pdf --json_path path/to/your/processed.json --output_path path/to/your/visualized.pdf
    ```
    *   Replace the paths accordingly (use the original PDF and the generated JSON).
    *   The output visualized PDF will be saved to the specified path (defaults to the same directory as the original PDF with `_visualized` appended).
    *   Use the `--draw_text` flag to also visualize detected text block boundaries (can be very cluttered).

## Configuration & Tuning

*   **`config.py`:** Modify VLM prompts (`VLM_PROMPT_...`), model names, JSON indentation, and CV detection parameters.
*   **CV Tuning:** The accuracy of CV-based highlight and ink detection heavily depends on the specific colors and rendering in your PDFs. You may need to:
    *   Adjust the **HSV ranges** in `config.py` (`CV_HIGHLIGHT_HSV_RANGES`) to match the exact shades of color used.
    *   Tune **morphology parameters** (`HIGHLIGHT_...`, `INK_...`) and **merging parameters** (`MAX_HORIZONTAL_GAP_RATIO`, `MIN_VERTICAL_OVERLAP_RATIO`) in `cv_utils.py` to optimize detection and continuity.

## Previous Approaches (Briefly)

Earlier iterations of related projects may have explored Graph Neural Networks (GNNs) for document understanding, but this specific pipeline focuses on the combination of LayoutLM, rule-based extraction, CV, and VLM analysis for annotated PDFs.

## Contributing

(Add contribution guidelines if applicable)

## License

(Add license information if applicable)