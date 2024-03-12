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
    
class InputEmbedding(layers.Layer):
    def __init__(self, maxlen, n_sector, embed_dim):
        """
        Parameters:
            - maxlen: the maximal length of the look-back period 
            - n_sector: the number of sectors
            - embed_dim: the dimension of the embedding vector
        """
        super().__init__()
        self.pos_emb = PositionalEmbedding(maxlen, embed_dim)
        self.stock_emb = StockEmbedding(embed_dim)
        self.sec_emb = SectorEmbedding(n_sector,embed_dim)

    def call(self, x):
        """
        Parameters:
            - x: input tensor of shape (n_stock, period, dim). 
                - x[i,t,0]: sector of stock i.
                - x[i,t,1]: return of stock i in period t
                - x[i,t,2]: market cap change of stock i in period t
                - x[i,t,3]: volume change of stock i in period t
        """
        positions = self.pos_emb(x)
        sectors = self.sec_emb(x[:,:,0])
        stocks = self.stock_emb(x[:,:,1:])
        return positions + sectors + stocks