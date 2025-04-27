import json
from openai import OpenAI
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

## Set OpenAI API Key
OpenAI.api_key = os.getenv("OPENAI_API_KEY")

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
    extracted_text_path = "evaluation_yj/processed_data/extracted_text.txt"  # Path to the processed text file
    text = read_text_from_file(extracted_text_path)
    summary = generate_summary(text)
    print("Generated Summary:", summary)

    # Save the summary to a file
    output_summary_path = "evaluation_yj/processed_data/summary.txt"
    save_summary_to_file(summary, output_summary_path)
    # You can now use `summary` for further metric calculations
