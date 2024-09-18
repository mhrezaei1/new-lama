from models import LanguageModel
from utils import load_jsonl_dataset, batchify, save_results
from evaluation import precision_at_1, get_ranking
import argparse
import os

def run_experiment(relations, data_dir, model_info, use_negated_probes=False):
    """Run masked language model experiments for a set of relations."""
    model = LanguageModel(model_info["model_name"])
    all_precision1 = []
    results = []
    results_dir = f"results/{model_info['label']}"
    os.makedirs(results_dir, exist_ok=True)

    for relation in relations:
        dataset_path = os.path.join(data_dir, relation["relation"] + ".jsonl")
        dataset = load_jsonl_dataset(dataset_path)

        for sample in dataset:
            sentences = sample["masked_sentences"]
            log_probs, tokenized_inputs = model.get_contextual_embeddings(sentences)

            masked_indices = [i for i, token in enumerate(tokenized_inputs['input_ids'][0]) if token == model.tokenizer.mask_token_id]
            correct_token_ids = [model.tokenizer.convert_tokens_to_ids([sample["obj_label"]])[0]]

            precision1 = precision_at_1(log_probs[0], masked_indices, correct_token_ids)
            all_precision1.append(precision1)

        # Save result
        mean_precision1 = sum(all_precision1) / len(all_precision1)
        print(f"Mean Precision@1: {mean_precision1}")

        results.append({
            "relation": relation["relation"],
            "mean_precision1": mean_precision1
        })

    # Save final results
    save_results(os.path.join(results_dir, "results.json"), results)
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run masked language model experiments.")
    parser.add_argument("--dataset", required=True, help="Dataset name (GoogleRE, TREx, ConceptNet, SQuAD)")
    parser.add_argument("--use_negated_probes", action="store_true", help="Include negated probes.")
    
    args = parser.parse_args()

    model_info = {"label": "bert_base", "model_name": "bert-base-uncased"}
    run_experiment(get_dataset_parameters(args.dataset), "datasets/", model_info, args.use_negated_probes)
