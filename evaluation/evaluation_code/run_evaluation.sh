#!/bin/bash

# Run Evaluation Pipeline for Annotation-Enhanced Knowledge Mapping Project

echo "Starting evaluation pipeline..."

# Step 1: Create variants of the knowledge graph for testing
echo "Creating knowledge graph variants..."
python create_kg_variants.py

# Step 2: Run the main evaluation comparing all methods
echo "Running main evaluation (comparing all methods)..."
python evaluate.py

# Step 3: Run the ablation study evaluation
echo "Running ablation study evaluation..."
python evaluate_ablation.py

# Step 4: Run the winning rate evaluation
echo "Running winning rate evaluation..."
python evaluate_winning_rate.py

# Step 5: Summarize results
echo "Evaluation complete! Results are stored in:"
echo "- evaluation_results.json (Persona F1 scores)"
echo "- ablation_results.json (Ablation study results)"
echo "- winning_rate_results.json (Winning rate comparison)"
echo "- responses_case_*.json (Example responses for qualitative analysis)"

echo -e "\nSummary of Persona F1 scores:"
if [ -f evaluation_results.json ]; then
    python -c "
import json
with open('evaluation_results.json', 'r') as f:
    results = json.load(f)
print('Method | Interests F1 | Confusions F1 | Known Facts F1 | Overall F1')
print('-' * 70)
for method, metrics in results.items():
    print(f'{method} | {metrics[\"interests_f1\"]:.3f} | {metrics[\"confusions_f1\"]:.3f} | {metrics[\"known_facts_f1\"]:.3f} | {metrics[\"overall_f1\"]:.3f}')
"
fi

echo -e "\nSummary of Winning Rate results:"
if [ -f winning_rate_results.json ]; then
    python -c "
import json
with open('winning_rate_results.json', 'r') as f:
    results = json.load(f)
print('Comparison | GPT Evaluation')
print('-' * 40)
for comparison, rate in results.items():
    print(f'{comparison} | {rate*100:.1f}%')
"
fi

echo -e "\nSummary of Ablation Study results:"
if [ -f ablation_results.json ]; then
    python -c "
import json
with open('ablation_results.json', 'r') as f:
    results = json.load(f)
baseline = results['full_method']['overall_f1']
print('Method Configuration | Overall F1 | Relative Change')
print('-' * 70)
for variant, metrics in results.items():
    overall_f1 = metrics['overall_f1']
    if variant == 'full_method':
        rel_change = '-'
    else:
        rel_change = f'{((overall_f1 - baseline) / baseline) * 100:.1f}%'
    print(f'{variant} | {overall_f1:.3f} | {rel_change}')
"
fi

echo "Done!" 