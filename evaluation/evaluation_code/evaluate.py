import json
import numpy as np
import openai
from chatKG import load_kg, retrieve_relevant_triples, construct_prompt, generate_answer
import re
from sklearn.metrics import f1_score
from collections import defaultdict

# Load API key
openai.api_key = ""

# Load test data (you would create this file with your test cases)
def load_test_data(filename="test_data.json"):
    with open(filename, "r", encoding="utf-8") as f:
        return json.load(f)

# Persona F1 calculation
def calculate_persona_f1(response, persona):
    """
    Calculate F1 score between response and persona elements
    """
    interests_f1 = calculate_text_overlap(response, persona["interests"])
    confusions_f1 = calculate_text_overlap(response, persona["confusions"])
    known_facts_f1 = calculate_text_overlap(response, persona["known_facts"])
    
    overall_f1 = (interests_f1 + confusions_f1 + known_facts_f1) / 3
    
    return {
        "interests_f1": interests_f1,
        "confusions_f1": confusions_f1,
        "known_facts_f1": known_facts_f1,
        "overall_f1": overall_f1
    }

def calculate_text_overlap(response, target_items):
    """
    Calculate unigram F1 overlap between response and target items
    """
    if not target_items:
        return 0.0
        
    response_words = set(re.sub(r'[^\w\s]', '', response.lower()).split())
    
    # Combine all target items and get unique words
    target_words = set()
    for item in target_items:
        target_words.update(re.sub(r'[^\w\s]', '', item.lower()).split())
    
    if not target_words or not response_words:
        return 0.0
    
    # Calculate precision, recall, and F1
    intersection = target_words.intersection(response_words)
    precision = len(intersection) / len(response_words) if response_words else 0
    recall = len(intersection) / len(target_words) if target_words else 0
    
    if precision + recall == 0:
        return 0.0
    
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1

# Methods for generating responses

def baseline_a(test_case):
    """
    Baseline A: Document only, Text-Prompt
    """
    document = test_case["document"]
    question = test_case["question"]
    
    prompt = f"Document content:\n{document}\n\nUser question: {question}\n\nPlease provide a detailed answer to the question based on the document content."
    
    response = openai.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful educational assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.5,
        max_tokens=250
    )
    
    return response.choices[0].message.content.strip()

def baseline_b(test_case):
    """
    Baseline B: Document + Annotation, Text-Prompt
    """
    document = test_case["document"]
    question = test_case["question"]
    annotations = test_case["annotations"]
    
    # Format annotations as text
    annotations_text = ""
    for annotation in annotations:
        annotations_text += f"- Highlighted: \"{annotation['highlight']}\"\n"
        annotations_text += f"  Comment: \"{annotation['comment']}\"\n"
    
    prompt = f"Document content:\n{document}\n\nUser annotations:\n{annotations_text}\n\nUser question: {question}\n\nPlease provide a detailed, personalized answer to the question based on both the document content and the user's annotations."
    
    response = openai.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful educational assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.5,
        max_tokens=250
    )
    
    return response.choices[0].message.content.strip()

def baseline_c(test_case):
    """
    Baseline C: Document + User Info, KG (no annotations)
    """
    question = test_case["question"]
    
    # Load the KG without annotations (would need to create this variant)
    G = load_kg("kg_no_annotations.json")
    
    if G is None:
        return "Error: Knowledge graph could not be loaded."
    
    # Use the existing chatKG approach but with a KG that doesn't have annotations
    retrieved = retrieve_relevant_triples(question, G, top_k=5)
    
    if not retrieved:
        return "No relevant information found in the knowledge graph."
    
    prompt_text = construct_prompt(question, retrieved)
    answer = generate_answer(prompt_text)
    
    return answer

def our_method(test_case):
    """
    Our Method: Document + Annotation, KG
    """
    question = test_case["question"]
    
    # Load the full KG with annotations
    G = load_kg("kg.json")
    
    if G is None:
        return "Error: Knowledge graph could not be loaded."
    
    # Use the existing chatKG approach
    retrieved = retrieve_relevant_triples(question, G, top_k=5)
    
    if not retrieved:
        return "No relevant information found in the knowledge graph."
    
    prompt_text = construct_prompt(question, retrieved)
    answer = generate_answer(prompt_text)
    
    return answer

def run_evaluation():
    # Load test data
    test_data = load_test_data()
    
    # Results storage
    results = {
        "baseline_a": defaultdict(list),
        "baseline_b": defaultdict(list),
        "baseline_c": defaultdict(list),
        "our_method": defaultdict(list)
    }
    
    # Process each test case
    for i, test_case in enumerate(test_data):
        print(f"Processing test case {i+1}/{len(test_data)}...")
        
        # Generate responses for each method
        response_a = baseline_a(test_case)
        response_b = baseline_b(test_case)
        response_c = baseline_c(test_case)
        response_our = our_method(test_case)
        
        # Calculate metrics
        persona = test_case["persona"]
        metrics_a = calculate_persona_f1(response_a, persona)
        metrics_b = calculate_persona_f1(response_b, persona)
        metrics_c = calculate_persona_f1(response_c, persona)
        metrics_our = calculate_persona_f1(response_our, persona)
        
        # Store results
        for metric, value in metrics_a.items():
            results["baseline_a"][metric].append(value)
        for metric, value in metrics_b.items():
            results["baseline_b"][metric].append(value)
        for metric, value in metrics_c.items():
            results["baseline_c"][metric].append(value)
        for metric, value in metrics_our.items():
            results["our_method"][metric].append(value)
        
        # Optional: Save responses for qualitative analysis
        with open(f"responses_case_{i+1}.json", "w", encoding="utf-8") as f:
            json.dump({
                "question": test_case["question"],
                "baseline_a": response_a,
                "baseline_b": response_b,
                "baseline_c": response_c,
                "our_method": response_our
            }, f, ensure_ascii=False, indent=4)
    
    # Calculate averages
    final_results = {}
    for method, metrics in results.items():
        final_results[method] = {}
        for metric, values in metrics.items():
            final_results[method][metric] = sum(values) / len(values) if values else 0
    
    # Save final results
    with open("evaluation_results.json", "w", encoding="utf-8") as f:
        json.dump(final_results, f, ensure_ascii=False, indent=4)
    
    # Print summary
    print("\nEvaluation Results (Persona F1):")
    print("Method | Interests F1 | Confusions F1 | Known Facts F1 | Overall F1")
    print("-" * 70)
    for method, metrics in final_results.items():
        print(f"{method} | {metrics['interests_f1']:.3f} | {metrics['confusions_f1']:.3f} | {metrics['known_facts_f1']:.3f} | {metrics['overall_f1']:.3f}")

if __name__ == "__main__":
    run_evaluation() 