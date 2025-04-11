from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification
from datasets import load_dataset
from PIL import Image
import torch
import fitz  # PyMuPDF
import json

# Step 1: Load model & processor
processor = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=True)
model = LayoutLMv3ForTokenClassification.from_pretrained("microsoft/layoutlmv3-base")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Step 2: Load PDF & extract layout-aware tokens
pdf_path = "Multimodal_Interfaces_A_Survey_of_Principles_Model-1 (1).pdf"
doc = fitz.open(pdf_path)

triplets = []

for page_index in range(len(doc)):
    page = doc[page_index]
    blocks = page.get_text("dict")["blocks"]

    # Step 3: Extract highlighted text with layout info
    highlights = [annot for annot in page.annots() if annot.type[0] == 8] if page.annots() else []
    
    for annot in highlights:
        bbox = annot.rect
        text = page.get_textbox(bbox)
        
        if not text.strip():
            continue

        # Step 4: Screenshot (for layout model) + tokenize
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
        image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

        encoding = processor(
            image, 
            return_tensors="pt", 
            truncation=True, 
            max_length=512
        ).to(device)

        # Step 5: Predict tokens & retrieve layout-aware context
        outputs = model(**encoding)
        logits = outputs.logits
        predicted_class = logits.argmax(-1)

        # Simulated: build a semantic + user_intent triple (real system would use a head)
        semantic_triple = {
            "type": "semantic",
            "subject": "document_title_here",
            "predicate": "contains_statement",
            "object": text.strip(),
            "attributes": {
                "source": "highlight",
                "position": str(bbox),
                "context": "Text near highlight on the same page.",
                "document_id": f"page_{page_index+1}",
                "timestamp": "2025-04-01T09:00:00Z"
            }
        }

        user_triple = {
            "type": "user_intent",
            "subject": "username1",
            "predicate": "marked_important",
            "object": text.strip(),
            "attributes": {
                "comment": annot.info.get("title", "highlighted"),
                "source": "comment",
                "document_id": f"page_{page_index+1}",
                "timestamp": "2025-04-01T09:00:00Z",
                "tags": ["highlight"]
            }
        }

        triplets.extend([semantic_triple, user_triple])

# Step 6: Save triplets for downstream KG script
with open("layoutlm_triplets.json", "w", encoding="utf-8") as f:
    json.dump(triplets, f, ensure_ascii=False, indent=4)

print("✅ Triplets extracted and saved to layoutlm_triplets.json")
