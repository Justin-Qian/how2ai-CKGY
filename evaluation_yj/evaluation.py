import os
import json

# Function to read text from a file
def read_text_from_file(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()

# Import metric functions from metric.py
from metrics import calculate_bleu, calculate_rouge, evaluate_text

if __name__ == "__main__":
    # Paths to the processed text files
    reference_text_path = "evaluation_yj/processed_data/reference_text.txt"
    summary_text_path = "evaluation_yj/processed_data/summary.txt"

    # Read the reference and summary texts
    reference_text = read_text_from_file(reference_text_path)
    summary_text = read_text_from_file(summary_text_path)

    # Calculate individual metrics
    bleu = calculate_bleu(reference_text, summary_text)
    rouge = calculate_rouge(reference_text, summary_text)

    print(f"BLEU score: {bleu:.4f}")
    print(f"ROUGE-1 F1 score: {rouge['rouge-1']['f']:.4f}")
    print(f"ROUGE-2 F1 score: {rouge['rouge-2']['f']:.4f}")
    print(f"ROUGE-L F1 score: {rouge['rouge-l']['f']:.4f}")


    # Save results to a JSON file
    results = {
        "bleu": bleu,
        "rouge-1": rouge['rouge-1']['f'],
        "rouge-2": rouge['rouge-2']['f'],
        "rouge-l": rouge['rouge-l']['f'],
    }
    with open("evaluation_yj/processed_data/evaluation_results.json", "w", encoding="utf-8") as json_file:
        json.dump(results, json_file, ensure_ascii=False, indent=4)
