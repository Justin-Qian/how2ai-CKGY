from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge import Rouge
import numpy as np

def calculate_bleu(reference, candidate):
    """
    Calculate BLEU score between reference and candidate texts.

    Args:
        reference (str): Reference text
        candidate (str): Candidate text

    Returns:
        float: BLEU score
    """
    reference_tokens = [reference.split()]
    candidate_tokens = candidate.split()

    smoothing_function = SmoothingFunction().method1
    try:
        bleu_score = sentence_bleu(reference_tokens, candidate_tokens,
                                  smoothing_function=smoothing_function)
        return bleu_score
    except:
        return 0.0

def calculate_rouge(reference, candidate):
    """
    Calculate ROUGE scores between reference and candidate texts.

    Args:
        reference (str): Reference text
        candidate (str): Candidate text

    Returns:
        dict: Dictionary with ROUGE-1, ROUGE-2 and ROUGE-L scores
    """
    rouge = Rouge()
    try:
        scores = rouge.get_scores(candidate, reference)
        return scores[0]
    except:
        return {"rouge-1": {"f": 0.0}, "rouge-2": {"f": 0.0}, "rouge-l": {"f": 0.0}}

def evaluate_text(reference, candidate):
    """
    Evaluate similarity between reference and candidate texts.

    Args:
        reference (str): Reference text
        candidate (str): Candidate text

    Returns:
        dict: Dictionary with all evaluation metrics
    """
    bleu = calculate_bleu(reference, candidate)
    rouge = calculate_rouge(reference, candidate)

    # Extract ROUGE scores
    rouge_1 = rouge["rouge-1"]["f"]
    rouge_2 = rouge["rouge-2"]["f"]
    rouge_l = rouge["rouge-l"]["f"]

    # Calculate overall score
    metrics = {
        "bleu": bleu,
        "rouge-1": rouge_1,
        "rouge-2": rouge_2,
        "rouge-L": rouge_l,
        "overall": np.mean([bleu, rouge_1, rouge_2, rouge_l])
    }

    return metrics


if __name__ == "__main__":
    # Example usage
    reference_text = "The quick brown fox jumps over the lazy dog."
    candidate_text = "The quick brown fox jumps over a lazy dog."

    # Calculate individual metrics
    bleu = calculate_bleu(reference_text, candidate_text)
    rouge = calculate_rouge(reference_text, candidate_text)

    print(f"BLEU score: {bleu:.4f}")
    print(f"ROUGE-1 F1 score: {rouge['rouge-1']['f']:.4f}")
    print(f"ROUGE-2 F1 score: {rouge['rouge-2']['f']:.4f}")
    print(f"ROUGE-L F1 score: {rouge['rouge-l']['f']:.4f}")

    # Using the combined evaluation function
    print("\nUsing combined evaluation:")
    metrics = evaluate_text(reference_text, candidate_text)
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
