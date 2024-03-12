import keras
from keras import layers
from keras import ops

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
    
class StockEmbedding(layers.Layer):
    def __init__(self, embed_dim):
        """
        Parameters:
            - n_sector: the number of sectors
            - embed_dim: the dimension of the embedding vector
        """
        super().__init__()
        self.stock_emb = layers.Dense(units = embed_dim)
    
    def call(self, x):
        return self.stock_emb(x)

class PositionalEmbedding(layers.Layer):
    def __init__(self, maxlen, embed_dim):
        """
        Parameters:
            - maxlen: the maximal length of the look-back period 
            - embed_dim: the dimension of the embedding vector
        """
        super().__init__()
        self.maxlen = maxlen
        self.pos_emb = layers.Embedding(input_dim = maxlen,
                                        output_dim = embed_dim)
    
    def call(self, x):
        maxlen = ops.shape(x)[-1]
        # use reverse order like 4, 3, 2, 1, 0 to encode the time order
        positions = ops.arange(start = maxlen-1, stop = -1, step = -1)
        return self.pos_emb(positions)