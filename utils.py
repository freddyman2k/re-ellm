import torch
from sentence_transformers import SentenceTransformer

class TextEmbedder:
    """Uses a pretrained SBERT model to embed text into a vector representation that can be used by a SB3 Policy network. 
    Caches embeddings for each input to avoid recomputing them.
    """
    def __init__(self, model_name='paraphrase-MiniLM-L3-v2', device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        self.model = SentenceTransformer(model_name, device=device)
        self.embeddings = {}
    
    def embed(self, text):
        if text in self.embeddings:
            return self.embeddings[text]
        else:
            embedding = self.model.encode(text)
            self.embeddings[text] = embedding
            return embedding