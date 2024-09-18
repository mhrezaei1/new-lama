import torch

def precision_at_1(log_probs, masked_indices, correct_token_ids):
    """Calculate Precision@1 for masked predictions."""
    correct_predictions = 0
    total_predictions = len(masked_indices)
    
    for i, mask_index in enumerate(masked_indices):
        top_prediction = torch.argmax(log_probs[mask_index])
        if top_prediction == correct_token_ids[i]:
            correct_predictions += 1
    
    return correct_predictions / total_predictions if total_predictions > 0 else 0

def get_ranking(log_probs, masked_indices, vocab, label_index=None, topk=1000):
    """Rank the masked tokens' probabilities and calculate MRR, Precision@k."""
    rankings = []
    for mask_index in masked_indices:
        sorted_probs, sorted_indices = torch.topk(log_probs[mask_index], topk)
        ranking = (sorted_indices == label_index).nonzero(as_tuple=True)
        if len(ranking) > 0:
            rank_position = ranking[0].item() + 1
            rankings.append(1 / rank_position)
        else:
            rankings.append(0)
    return sum(rankings) / len(rankings) if rankings else 0
