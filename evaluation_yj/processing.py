import json

## Function to read text from specified pages of a JSON file and save to a text file
def save_text_from_json(input_json_path, output_text_path, start_page=1, end_page=9):
    with open(input_json_path, "r", encoding="utf-8") as file:
        data = json.load(file)
        text_blocks = []
        for page in data.get("pages", []):
            if start_page <= page.get("page_number", 0) <= end_page:
                for block in page.get("text_blocks", []):
                    text_blocks.append(block.get("text", ""))
        text_content = " ".join(text_blocks)
    with open(output_text_path, "w", encoding="utf-8") as text_file:
        text_file.write(text_content)

## Function to read all text from a JSON file and save to a text file
def save_reference_text_from_json(input_json_path, output_text_path):
    with open(input_json_path, "r", encoding="utf-8") as file:
        data = json.load(file)
        text_blocks = []
        for page in data.get("pages", []):
            for block in page.get("text_blocks", []):
                text_blocks.append(block.get("text", ""))
        text_content = " ".join(text_blocks)
    with open(output_text_path, "w", encoding="utf-8") as text_file:
        text_file.write(text_content)

if __name__ == "__main__":
    # Save text from pages 1-9
    input_json_path = "data/mixture_of_million_experts_processed.json"
    output_text_path = "processed_data/extracted_text.txt"
    save_text_from_json(input_json_path, output_text_path)

    # Save reference text from all pages
    summary_json_path = "data/sumary.json"
    reference_text_path = "processed_data/reference_text.txt"
    save_reference_text_from_json(summary_json_path, reference_text_path)
