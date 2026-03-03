def precision_recall_at_k(recommended, relevant, k):

    if k <= 0:
        return 0.0, 0.0

    recommended = list(recommended)
    relevant = set(relevant)

    top_k = recommended[:k]

    hits = sum(1 for item in top_k if item in relevant)

    precision = hits / len(top_k) if top_k else 0.0
    recall = hits / len(relevant) if relevant else 0.0

    return [float(precision), float(recall)]