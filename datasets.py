def get_TREx_parameters(data_path_pre="datasets/"):
    """Return TREx dataset parameters."""
    relations = load_jsonl_dataset(f"{data_path_pre}relations.jsonl")
    data_path_pre += "TREx/"
    data_path_post = ".jsonl"
    return relations, data_path_pre, data_path_post

def get_GoogleRE_parameters():
    """Return Google-RE dataset parameters."""
    relations = [
        {"relation": "place_of_birth", "template": "[X] was born in [Y] .", "template_negated": "[X] was not born in [Y] ."},
        {"relation": "date_of_birth", "template": "[X] (born [Y]).", "template_negated": "[X] (not born [Y])."},
        {"relation": "place_of_death", "template": "[X] died in [Y] .", "template_negated": "[X] did not die in [Y] ."},
    ]
    data_path_pre = "datasets/Google_RE/"
    data_path_post = "_test.jsonl"
    return relations, data_path_pre, data_path_post

def get_ConceptNet_parameters(data_path_pre="datasets/"):
    """Return ConceptNet dataset parameters."""
    relations = [{"relation": "test"}]
    data_path_pre += "ConceptNet/"
    data_path_post = ".jsonl"
    return relations, data_path_pre, data_path_post

def get_Squad_parameters(data_path_pre="datasets/"):
    """Return SQuAD dataset parameters."""
    relations = [{"relation": "test"}]
    data_path_pre += "Squad/"
    data_path_post = ".jsonl"
    return relations, data_path_pre, data_path_post

def get_dataset_parameters(dataset_name):
    """Return dataset parameters based on dataset name."""
    if dataset_name == "TREx":
        return get_TREx_parameters()
    elif dataset_name == "GoogleRE":
        return get_GoogleRE_parameters()
    elif dataset_name == "ConceptNet":
        return get_ConceptNet_parameters()
    elif dataset_name == "SQuAD":
        return get_Squad_parameters()
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
