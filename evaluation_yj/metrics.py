from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import numpy as np

def calculate_bleu(reference, candidate):
    """
    Calculate BLEU score between reference and candidate texts.
    Note: This function is kept for reference but not used in the current evaluation.

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
        dict: Dictionary with ROUGE-1 and ROUGE-L recall scores
    """
    try:
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
        scores = scorer.score(reference, candidate)
        return {
            "rouge-1": {"recall": scores['rouge1'].recall},
            "rouge-l": {"recall": scores['rougeL'].recall}
        }
    except:
        return {"rouge-1": {"recall": 0.0}, "rouge-l": {"recall": 0.0}}

def highlight_R_P_F1(highlight, candidate):
    """
    Calculate Precision, Recall and F1 score for highlight coverage.

    Args:
        highlight (str): Highlight text
        candidate (str): Candidate text

    Returns:
        dict: Dictionary containing highlight-p (precision), highlight-r (recall) and highlight-f1 scores
    """
    try:
        # Read stopwords
        with open('stopwords.txt', 'r', encoding='utf-8') as f:
            stopwords = set(line.strip() for line in f)

        # Filter stopwords using set difference
        highlight_tokens = set(highlight.lower().split()) - stopwords
        candidate_tokens = set(candidate.lower().split()) - stopwords

        # Calculate intersection
        intersection = highlight_tokens.intersection(candidate_tokens)
        intersection_size = len(intersection)

        # Calculate precision, recall and f1
        recall = intersection_size / len(highlight_tokens) if highlight_tokens else 0.0
        precision = intersection_size / len(candidate_tokens) if candidate_tokens else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        return {
            "highlight-p": precision,
            "highlight-r": recall,
            "highlight-f1": f1
        }
    except Exception as e:
        print(f"Error in highlight calculation: {str(e)}")
        return {
            "highlight-p": 0.0,
            "highlight-r": 0.0,
            "highlight-f1": 0.0
        }

def evaluate_text(reference, candidate, highlight):
    """
    Evaluate similarity between reference and candidate texts, and calculate highlight coverage.

    Args:
        reference (str): Reference text
        candidate (str): Candidate text
        highlight (str): Highlight text to check coverage against candidate

    Returns:
        dict: Dictionary with ROUGE scores and highlight coverage metrics
    """
    rouge = calculate_rouge(reference, candidate)
    highlight_scores = highlight_R_P_F1(highlight, candidate)

    # Extract ROUGE scores
    rouge_1 = rouge["rouge-1"]["recall"]
    rouge_l = rouge["rouge-l"]["recall"]

    # Return metrics without BLEU score
    metrics = {
        "rouge-1": rouge_1,
        "rouge-L": rouge_l,
        "highlight-p": highlight_scores["highlight-p"],
        "highlight-r": highlight_scores["highlight-r"],
        "highlight-f1": highlight_scores["highlight-f1"],
    }

    return metrics


if __name__ == "__main__":
    # Example usage
    reference_text = "The quick brown fox jumps over the lazy dog."
    candidate_text = "The quick brown fox jumps over a lazy dog."
    highlight_text = "quick fox lazy"  # 示例重点文本

    # Using the combined evaluation function
    print("\nUsing combined evaluation:")
    metrics = evaluate_text(reference_text, candidate_text, highlight_text)
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
