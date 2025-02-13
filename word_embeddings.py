import gensim.downloader as api
from sklearn.metrics.pairwise import cosine_similarity

class WordEmbeddings:
    
    def __init__(self):
        """
        Initialize the word embedding model.
        """
        self.model = api.load("word2vec-google-news-300")
        
    def get_word_similarity(self, w1: str, w2: str) -> float:
        """
        Calculate cosine similarity between two words.
        
        Args:
            w1: First word
            w2: Second word
            
        Returns:
            Cosine similarity score, or 0 if words not in vocabulary
        """
        if w1 in self.model and w2 in self.model:
            emb1 = self.model[w1].reshape(1, -1)
            emb2 = self.model[w2].reshape(1, -1)
            sim = cosine_similarity(emb1, emb2)
            return round(sim.item(), 6)
        return 0