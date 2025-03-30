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

    prompt += f"\nQuestion:\n{question}\n"
    instrcution1 = "\nPlease answer this question considering what the user is interested in, confused about, and has known."
    instrcution2 = "\nPlease answer this question considering what the user is interested in, confused about, and has known."
    prompt += instrcution1
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

## Save output to TXT
def save_output_to_txt(output_path, prompt, response, f1_score):
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("Prompt:\n")
        f.write(prompt.strip() + "\n\n")
        f.write("Response:\n")
        f.write(response.strip() + "\n\n")
        f.write(f"Persona F1 Score: {f1_score:.3f}\n")

## Total Pipeline
def total_pipeline(data_path, output_path, use_annotation=True):
    data = load_data(data_path)
    prompt = generate_prompt(data, use_annotation)
    response = get_response(prompt)
    f1 = get_unigram_f1(response, data['persona'])
    save_output_to_txt(output_path, prompt, response, f1)
    return f1

## Experiment
def experiment(data_path, output_base_path, use_annotation=True, repeat=5):
    f1_scores = []

    for i in range(repeat):
        run_output_path = f"{output_base_path}_run{i+1}.txt"
        f1 = total_pipeline(data_path, run_output_path, use_annotation)
        f1_scores.append(f1)

    avg_f1 = sum(f1_scores) / repeat

    summary_path = f"{output_base_path}_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"Average Persona F1 Score over {repeat} runs: {avg_f1:.3f}\n")
        f.write("Individual scores:\n")
        for i, score in enumerate(f1_scores):
            f.write(f"Run {i+1}: {score:.3f}\n")

## Main function
if __name__ == "__main__":
    experiment(
        data_path="sample_data.json",
        output_base_path=os.path.join("Output", "output_with_annotation"),
        use_annotation=True,
        repeat=5
    )

    experiment(
        data_path="sample_data.json",
        output_base_path=os.path.join("Output", "output_without_annotation"),
        use_annotation=False,
        repeat=5
    )
