import keras
from keras import layers

class SectorEmbedding(layers.Layer):
    def __init__(self, n_sector, embed_dim):
        """
        Parameters:
            - n_sector: the number of sectors
            - embed_dim: the dimension of the embedding vector
        """
        super().__init__()
        self.sec_emb = layers.Embedding(input_dim = n_sector,
                                        output_dim = embed_dim)
    
    def call(self, x):
        return self.sec_emb(x)