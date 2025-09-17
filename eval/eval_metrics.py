import json
from backend.rag_engine import query_rag_with_sources
from pathlib import Path
from backend.rag_engine import retrieve_sources

def _norm(name: str) -> str:
    return Path(str(name)).name.lower()

# --- Metrics ---
def recall_at_k(pred_sources, relevant_docs, k=3):
    retrieved = [_norm(s["doc_id"]) for s in pred_sources[:k]]
    relevant = {_norm(doc) for doc in relevant_docs}
    return int(len(set(retrieved) & relevant) > 0)

def precision_at_k(pred_sources, relevant_docs, k=3):
    retrieved = [_norm(s["doc_id"]) for s in pred_sources[:k]]
    if not retrieved:
        return 0.0
    relevant = {_norm(doc) for doc in relevant_docs}
    hits = len(set(retrieved) & relevant)
    return hits / len(retrieved)

def evaluate(dataset_path="eval/eval_dataset.json", k=3, debug=False):
    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    recalls, precisions = [], []

    for i, item in enumerate(dataset, 1):
        q = item["question"]
        gt_docs = item["relevant_docs"]

        pred_sources = retrieve_sources(q, top_k=max(k, 10))  # get more, we'll cut to k
        r = recall_at_k(pred_sources, gt_docs, k)
        p = precision_at_k(pred_sources, gt_docs, k)
        recalls.append(r)
        precisions.append(p)

        if debug:
            topk = [s["doc_id"] for s in pred_sources[:k]]
            print(f"\n#{i} Q: {q}")
            print(f"  GT docs: {gt_docs}")
            print(f"  GT norm: {[ _norm(x) for x in gt_docs ]}")
            print(f"  Pred@{k}: {topk}")
            print(f"  Pred norm: {[ _norm(x) for x in topk ]}")
            print(f"  Recall@{k}={r}  Precision@{k}={p:.2f}")

    print(f"\nRecall@{k}: {sum(recalls)/len(recalls):.2f}")
    print(f"Precision@{k}: {sum(precisions)/len(precisions):.2f}")

if __name__ == "__main__":
    evaluate(debug=True)

