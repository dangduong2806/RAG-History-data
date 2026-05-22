import sys

try:
    from src.evaluation.qa_metrics import (
        aggregate_mean,
        squad_compute_exact,
        squad_compute_f1,
        squad_compute_recall,
    )
except ImportError:
    from qa_metrics import (
        aggregate_mean,
        squad_compute_exact,
        squad_compute_f1,
        squad_compute_recall,
    )

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
        
        best_exact = max([squad_compute_exact(g, pred) for g in possible_golds])
        best_f1 = max([squad_compute_f1(g, pred) for g in possible_golds])
        best_recall = max([squad_compute_recall(g, pred) for g in possible_golds])
        
        exact_scores.append(best_exact)
        f1_scores.append(best_f1)
        recall_scores.append(best_recall)
        
    avg_exact = aggregate_mean(exact_scores)
    avg_f1 = aggregate_mean(f1_scores)
    avg_recall = aggregate_mean(recall_scores)
    
    print(f"Total evaluated instances: {len(exact_scores)}")
    print(f"Exact Match: {avg_exact:.4f}")
    print(f"F1 Score:    {avg_f1:.4f}")
    print(f"Recall:      {avg_recall:.4f}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python evaluate.py <gold_file> <pred_file>")
        sys.exit(1)
        
    evaluate(sys.argv[1], sys.argv[2])
