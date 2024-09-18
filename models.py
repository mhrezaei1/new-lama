from transformers import BertForMaskedLM, BertTokenizer, RobertaForMaskedLM, RobertaTokenizer
import torch

class LanguageModel:
    def __init__(self, model_name):
        if "bert" in model_name:
            self.model = BertForMaskedLM.from_pretrained(model_name)
            self.tokenizer = BertTokenizer.from_pretrained(model_name)
        elif "roberta" in model_name:
            self.model = RobertaForMaskedLM.from_pretrained(model_name)
            self.tokenizer = RobertaTokenizer.from_pretrained(model_name)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()

    def tokenize_sentences(self, sentences):
        return self.tokenizer(sentences, padding=True, truncation=True, return_tensors="pt").to(self.device)

    def get_contextual_embeddings(self, batch_sentences):
        tokenized_inputs = self.tokenize_sentences(batch_sentences)
        with torch.no_grad():
            outputs = self.model(**tokenized_inputs)
            log_probs = torch.log_softmax(outputs.logits, dim=-1)
        return log_probs, tokenized_inputs
