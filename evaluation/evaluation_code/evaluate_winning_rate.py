import json
import openai
import os
from pathlib import Path
import numpy as np
import random

# Load API key
openai.api_key = ""

def load_responses():
    """Load all responses from the evaluation."""
    responses = []
    
    # Find all response files
    response_files = list(Path('.').glob('responses_case_*.json'))
    
    for file_path in response_files:
        with open(file_path, 'r', encoding='utf-8') as f:
            response_data = json.load(f)
            responses.append(response_data)
    
    return responses

def evaluate_with_gpt(response_a, response_b, question, persona, method_a, method_b):
    """Use GPT to evaluate which response better aligns with the user persona."""
    
    interests = "\n".join([f"- {item}" for item in persona["interests"]])
    confusions = "\n".join([f"- {item}" for item in persona["confusions"]])
    known_facts = "\n".join([f"- {item}" for item in persona["known_facts"]])
    
    prompt = f"""You are evaluating two AI responses to a user question. 
Choose the response that better aligns with the user's persona.

USER PERSONA:
Interests:
{interests}

Points of confusion:
{confusions}

Known facts:
{known_facts}

USER QUESTION:
{question}

RESPONSE A:
{response_a}

RESPONSE B:
{response_b}

Based solely on how well each response addresses the user's specific interests, confusions, and builds upon their known facts, which response is better? 
Answer with either "A" or "B" and provide a brief explanation of your choice.
"""
    
    response = openai.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are an objective evaluator of personalized educational content."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
        max_tokens=150
    )
    
    answer = response.choices[0].message.content.strip()
    winner = "A" if answer.startswith("A") else "B"
    
    # Map the winner back to the method name
    winning_method = method_a if winner == "A" else method_b
    
    return winning_method, answer

def evaluate_winning_rate():
    """Evaluate winning rate between different methods using GPT."""
    responses = load_responses()
    
    # Define method pairs to compare
    method_pairs = [
        ("our_method", "baseline_a"),
        ("our_method", "baseline_b"),
        ("our_method", "baseline_c")
    ]
    
    # Results storage
    results = {pair: [] for pair in method_pairs}
    
    # Process each test case
    for i, case in enumerate(responses):
        print(f"Evaluating case {i+1}/{len(responses)}...")
        
        question = case["question"]
        
        # Load test data to get persona
        with open("test_data.json", "r", encoding="utf-8") as f:
            test_data = json.load(f)
            persona = test_data[i]["persona"]
        
        # Evaluate each method pair
        for method_a, method_b in method_pairs:
            # Randomize order to avoid position bias
            if random.random() > 0.5:
                response_a = case[method_a]
                response_b = case[method_b]
                first, second = method_a, method_b
            else:
                response_a = case[method_b]
                response_b = case[method_a]
                first, second = method_b, method_a
            
            winner, explanation = evaluate_with_gpt(response_a, response_b, question, persona, first, second)
            
            # Record if our_method won
            pair = (method_a, method_b)
            results[pair].append(1 if winner == "our_method" else 0)
            
            # Save detailed results
            with open(f"winning_rate_case_{i+1}_{method_a}_vs_{method_b}.txt", "w", encoding="utf-8") as f:
                f.write(f"Question: {question}\n\n")
                f.write(f"Method {first} response:\n{response_a}\n\n")
                f.write(f"Method {second} response:\n{response_b}\n\n")
                f.write(f"Winner: {winner}\n")
                f.write(f"Explanation: {explanation}\n")
    
    # Calculate winning rates
    winning_rates = {}
    for pair, outcomes in results.items():
        method_a, method_b = pair
        winning_rates[f"{method_a}_vs_{method_b}"] = sum(outcomes) / len(outcomes) if outcomes else 0
    
    # Save final results
    with open("winning_rate_results.json", "w", encoding="utf-8") as f:
        json.dump(winning_rates, f, ensure_ascii=False, indent=4)
    
    # Print summary
    print("\nWinning Rate Results:")
    print("Comparison | GPT Evaluation")
    print("-" * 40)
    for comparison, rate in winning_rates.items():
        print(f"{comparison} | {rate*100:.1f}%")

if __name__ == "__main__":
    evaluate_winning_rate() 