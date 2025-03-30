import json
from openai import OpenAI
import os
from collections import Counter

## Load the json data
def load_data(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

## Generate Prompt Template
def generate_prompt(data_point, use_annotation=True):
    doc = data_point['document_text']
    question = data_point['question']
    prompt = f"Document:\n{doc}\n"

    if use_annotation and 'annotations' in data_point:
        ann_texts = "\n".join([f"- {ann['highlight']}: {ann['comment']}" for ann in data_point['annotations']])
        prompt += f"\nUser Annotations:\n{ann_texts}\n"

    prompt += f"\nQuestion:\n{question}\n\nAnswer:"
    return prompt

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

## Calculate Persona F1
def get_unigram_f1(pred_text, persona_fields):
    pred_tokens = pred_text.lower().split()
    truth_text = " ".join(persona_fields['interests'] +
                          persona_fields['confusions'] +
                          persona_fields['known_facts'])
    truth_tokens = truth_text.lower().split()

    pred_counts = Counter(pred_tokens)
    truth_counts = Counter(truth_tokens)
    overlap = sum((pred_counts & truth_counts).values())

    precision = overlap / (len(pred_tokens) + 1e-8)
    recall = overlap / (len(truth_tokens) + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    return f1

## Save output to JSON
def save_output_to_json(output_path, prompt, response, f1_score):
    result = {
        "prompt": prompt,
        "response": response,
        "persona_f1": f"{f1_score:.3f}"
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

## Main function
if __name__ == "__main__":
    data = load_data("sample_data.json")

    prompt = generate_prompt(data, use_annotation=True)
    response = get_response(prompt)
    f1 = get_unigram_f1(response, data['persona'])

    save_output_to_json("output_result.json", prompt, response, f1)
