import json
from openai import OpenAI
import os

## Set OpenAI API Key
os.environ["OPENAI_API_KEY"] = "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"

## Read text from a file
def read_text_from_file(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()

## Get the response
def get_response(prompt, model="gpt-4"):
    client = OpenAI()
    response = client.responses.create(
        model=model,
        input=[
            {
                "role": "system",
                "content": "You are a helpful assistant answering user questions."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
    )
    return response.output_text

## Generate summary from text
def generate_summary(text):
    prompt = f"Summarize the following text: {text}"
    summary = get_response(prompt)
    return summary

## Save summary to a file
def save_summary_to_file(summary, output_path):
    with open(output_path, "w", encoding="utf-8") as file:
        file.write(summary)

if __name__ == "__main__":
    # Example usage
    extracted_text_path = "processed_data/extracted_text.txt"  # Path to the processed text file
    text = read_text_from_file(extracted_text_path)
    summary = generate_summary(text)
    print("Generated Summary:", summary)

    # Save the summary to a file
    output_summary_path = "processed_data/summary.txt"
    save_summary_to_file(summary, output_summary_path)
    # You can now use `summary` for further metric calculations
