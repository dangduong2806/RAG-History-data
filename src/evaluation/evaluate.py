import sys
import collections
import string

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return text.replace(" a ", " ").replace(" an ", " ").replace(" the ", " ")

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def get_tokens(s):
    if not s:
        return []
    return normalize_answer(s).split()

def compute_exact(a_gold, a_pred):
    return int(normalize_answer(a_gold) == normalize_answer(a_pred))

def compute_f1(a_gold, a_pred):
    gold_toks = get_tokens(a_gold)
    pred_toks = get_tokens(a_pred)
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        return int(gold_toks == pred_toks)
        
    if num_same == 0:
        return 0
        
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def compute_recall(a_gold, a_pred):
    gold_toks = get_tokens(a_gold)
    pred_toks = get_tokens(a_pred)
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0:
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0
    return 1.0 * num_same / len(gold_toks)

def evaluate(gold_file, pred_file):
    with open(gold_file, 'r', encoding='utf-8') as fg:
        golds = [line.strip() for line in fg]
        
    with open(pred_file, 'r', encoding='utf-8') as fp:
        preds = [line.strip() for line in fp]
        
    if len(golds) != len(preds):
        print(f"Warning: Number of predictions ({len(preds)}) does not match number of gold answers ({len(golds)}).")
        
    exact_scores = []
    f1_scores = []
    recall_scores = []
    
    for i in range(min(len(golds), len(preds))):
        gold = golds[i]
        pred = preds[i]
        
        # SQuAD evaluation usually takes the max score over all possible references
        # Since our references might be separated by semicolon, we split them
        possible_golds = [g.strip() for g in gold.split(";")]
        
        best_exact = max([compute_exact(g, pred) for g in possible_golds])
        best_f1 = max([compute_f1(g, pred) for g in possible_golds])
        best_recall = max([compute_recall(g, pred) for g in possible_golds])
        
        exact_scores.append(best_exact)
        f1_scores.append(best_f1)
        recall_scores.append(best_recall)
        
    avg_exact = sum(exact_scores) / len(exact_scores) if exact_scores else 0
    avg_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0
    avg_recall = sum(recall_scores) / len(recall_scores) if recall_scores else 0
    
    print(f"Total evaluated instances: {len(exact_scores)}")
    print(f"Exact Match: {avg_exact:.4f}")
    print(f"F1 Score:    {avg_f1:.4f}")
    print(f"Recall:      {avg_recall:.4f}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python evaluate.py <gold_file> <pred_file>")
        sys.exit(1)
        
    evaluate(sys.argv[1], sys.argv[2])
