import json

def load_jsonl_dataset(filepath):
    """Load a dataset from a jsonl file."""
    data = []
    with open(filepath, "r") as f:
        for line in f.readlines():
            data.append(json.loads(line))
    return data

def save_results(filepath, results):
    """Save experiment results to a file."""
    with open(filepath, "w") as f:
        json.dump(results, f, indent=4)

def batchify(data, batch_size):
    """Batch data for efficient processing."""
    return [data[i:i + batch_size] for i in range(0, len(data), batch_size)]
