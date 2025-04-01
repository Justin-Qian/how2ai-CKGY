import json
import numpy as np
import openai
from chatKG import retrieve_relevant_triples, construct_prompt, generate_answer
import re
from collections import defaultdict
import networkx as nx
from networkx.readwrite import json_graph

# Load API key
openai.api_key = ""

def load_kg(filename):
    """Load knowledge graph from JSON file"""
    try:
        with open(filename, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Create a new DiGraph
        G = nx.DiGraph()
        
        # Add nodes from the node-link format
        for node in data["nodes"]:
            G.add_node(node["id"])
        
        # Add edges from the node-link format
        for link in data["links"]:
            source = link["source"]
            target = link["target"]
            # Extract predicate and other attributes
            predicate = link.get("predicate", "")
            triple_type = link.get("triple_type", "semantic")
            # Remove these keys to avoid duplication as they'll be specified separately
            attrs = {k: v for k, v in link.items() 
                    if k not in ["source", "target", "predicate", "triple_type"]}
            
            # Add the edge with all attributes
            G.add_edge(source, target, predicate=predicate, triple_type=triple_type, **attrs)
            
        return G
    except Exception as e:
        print("Error loading KG:", e)
        return None

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

def kg_method(test_case, kg_file):
    """
    Generate response using the specified KG variant
    """
    question = test_case["question"]
    
    # Load the specified KG
    G = load_kg(kg_file)
    
    if G is None:
        return "Error: Knowledge graph could not be loaded."
    
    # Use the existing chatKG approach
    retrieved = retrieve_relevant_triples(question, G, top_k=5)
    
    if not retrieved:
        return "No relevant information found in the knowledge graph."
    
    prompt_text = construct_prompt(question, retrieved)
    answer = generate_answer(prompt_text)
    
    return answer

def run_ablation_evaluation():
    # Load test data
    test_data = load_test_data()
    
    # Define KG variants to test
    kg_variants = {
        "full_method": "kg.json",
        "wo_user_intent": "kg_no_user_intent.json",
        "wo_semantic": "kg_no_semantic.json",
        "wo_attributes": "kg_no_attributes.json",
        "text_only": "kg_no_annotations.json"  # Reusing baseline C for text-only
    }
    
    # Results storage
    results = {variant: defaultdict(list) for variant in kg_variants}
    
    # Process each test case
    for i, test_case in enumerate(test_data):
        print(f"Processing ablation test case {i+1}/{len(test_data)}...")
        
        # Generate responses for each KG variant
        responses = {}
        for variant_name, kg_file in kg_variants.items():
            responses[variant_name] = kg_method(test_case, kg_file)
        
        # Calculate metrics
        persona = test_case["persona"]
        for variant_name, response in responses.items():
            metrics = calculate_persona_f1(response, persona)
            
            # Store results
            for metric, value in metrics.items():
                results[variant_name][metric].append(value)
        
        # Optional: Save responses for this test case
        with open(f"ablation_responses_case_{i+1}.json", "w", encoding="utf-8") as f:
            json.dump({
                "question": test_case["question"],
                **responses
            }, f, ensure_ascii=False, indent=4)
    
    # Calculate averages
    final_results = {}
    for variant, metrics in results.items():
        final_results[variant] = {}
        for metric, values in metrics.items():
            final_results[variant][metric] = sum(values) / len(values) if values else 0
    
    # Save final results
    with open("ablation_results.json", "w", encoding="utf-8") as f:
        json.dump(final_results, f, ensure_ascii=False, indent=4)
    
    # Print summary
    print("\nAblation Study Results (Persona F1):")
    print("Method Configuration | Overall F1 | Relative Change")
    print("-" * 70)
    
    # Get baseline for relative change calculation
    baseline = final_results["full_method"]["overall_f1"]
    
    for variant, metrics in final_results.items():
        overall_f1 = metrics["overall_f1"]
        if variant == "full_method":
            rel_change = "-"
        else:
            rel_change = f"{((overall_f1 - baseline) / baseline) * 100:.1f}%"
        
        print(f"{variant} | {overall_f1:.3f} | {rel_change}")

if __name__ == "__main__":
    run_ablation_evaluation() 