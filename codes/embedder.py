import torch
from transformers import AutoTokenizer



class Embedder:
    def __init__(self, embeddings: list[str], model_name: str):
        tr_embeddings = []

        for embedding in embeddings:
            tr_embeddings.append(torch.load(embedding))

        self.embeddings = torch.cat(tr_embeddings)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)


    def embed(self, text):
        if not hasattr(self, 'tokenizer'):
            raise Exception('You need to load a tokenizer first!')
        input_ids = self.tokenizer(text)['input_ids']
        embeddings = []
        for i in input_ids:
            embeddings.append(self.embeddings[i])

        list_of_embeddings = torch.stack(embeddings)
        mean = torch.mean(list_of_embeddings, axis=0)
        return mean.tolist()