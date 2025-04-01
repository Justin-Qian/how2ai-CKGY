# Evaluation Framework for Annotation-Enhanced Knowledge Mapping

This directory contains the scripts needed to evaluate the performance of our annotation-enhanced knowledge graph approach compared to baselines.

## Overview

The evaluation framework consists of several components:

1. **Main Evaluation**: Compares four approaches (our method and three baselines) using the Persona F1 metric
2. **Winning Rate Evaluation**: Uses GPT-4 to determine which approach produces better responses
3. **Ablation Study**: Tests variations of our approach to understand component contributions

## Setup

Before running the evaluation, make sure you have:

1. Generated your knowledge graph using `AttTripleGraph.py` and saved it as `kg.json`
2. Created test data in `test_data.json` (a sample is provided)
3. Installed required dependencies:
   ```
   pip install networkx matplotlib openai numpy scikit-learn
   ```

## Running the Evaluation

You can run the complete evaluation pipeline using:

```bash
chmod +x run_evaluation.sh
./run_evaluation.sh
```

Or run individual components separately:

```bash
# Create KG variants for testing
python create_kg_variants.py

# Run main evaluation comparing all methods
python evaluate.py

# Run ablation study
python evaluate_ablation.py

# Run winning rate evaluation
python evaluate_winning_rate.py
```

## Understanding the Results

The evaluation produces several output files:

- **evaluation_results.json**: Contains Persona F1 scores for all methods
- **winning_rate_results.json**: Contains winning rates from the GPT evaluation
- **ablation_results.json**: Contains results from the ablation study
- **responses_case_*.json**: Individual responses for qualitative analysis

## Metrics Details

### Persona F1

The Persona F1 metric measures how well the generated response aligns with the user's persona:
- **Interests F1**: Overlap with user's stated interests
- **Confusions F1**: Addressing user's points of confusion
- **Known Facts F1**: Building upon user's prior knowledge
- **Overall F1**: Average of the three components

### Winning Rate

The winning rate compares two methods head-to-head, calculating the percentage of times our method produces a better response than the baseline according to GPT-4's evaluation.

### Methods Being Evaluated

1. **Baseline A (Document, Text)**: Uses only the document content in text format
2. **Baseline B (Document + Annotation, Text)**: Uses document content and annotations in text format
3. **Baseline C (Document + User Info, KG)**: Uses a knowledge graph without annotations
4. **Our Method (Document + Annotation, KG)**: Uses both annotations and knowledge graph structure

## Ablation Study Variants

The ablation study tests the following variants:

1. **Full Method**: Complete implementation with all components
2. **Without User Intent Triples**: Removes user intent triples
3. **Without Semantic Triples**: Removes semantic triples
4. **Without Attribute Context**: Removes attribute metadata
5. **Text-only Representation**: Uses a simple text representation (equivalent to Baseline B)

## Customizing the Evaluation

You can modify the test data in `test_data.json` to include more test cases or adjust the persona information. The structure of each test case should include:
- `document`: The source document text
- `question`: The user's question
- `annotations`: List of user annotations (highlight + comment)
- `persona`: User's interests, confusions, and known facts 