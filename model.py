import keras
from keras import layers
from keras import ops

class SectorEmbedding(layers.Layer):
    def __init__(self, n_sector, embed_dim, **kwargs):
        """
        Parameters:
            - n_sector: the number of sectors
            - embed_dim: the dimension of the embedding vector
        """
        super(SectorEmbedding, self).__init__(**kwargs)
        self.sec_emb = layers.Embedding(input_dim = n_sector,
                                        output_dim = embed_dim)
    
    def call(self, x):
        return self.sec_emb(x)
    
class StockEmbedding(layers.Layer):
    def __init__(self, embed_dim, **kwargs):
        """
        Parameters:
            - embed_dim: the dimension of the embedding vector
        """
        super(StockEmbedding, self).__init__(**kwargs)
        self.stock_emb = layers.Dense(units = embed_dim)
    
    def call(self, x):
        """
        Assuming the input x is of shape (stock, dim, period)
        """
        # transpose the last two axes
        x = ops.transpose(x,axes = [0,1,3,2])
        return self.stock_emb(x)

class PositionalEmbedding(layers.Layer):
    def __init__(self, maxlen, embed_dim, **kwargs):
        """
        Parameters:
            - maxlen: the maximal length of the look-back period 
            - embed_dim: the dimension of the embedding vector
        """
        super(PositionalEmbedding, self).__init__(**kwargs)
        self.maxlen = maxlen
        self.pos_emb = layers.Embedding(input_dim = maxlen,
                                        output_dim = embed_dim)
    
    def call(self, x):
        maxlen = ops.shape(x)[-1]
        # use reverse order like 4, 3, 2, 1, 0 to encode the time order
        positions = ops.arange(start = maxlen-1, stop = -1, step = -1)
        return self.pos_emb(positions)
    
class InputEmbedding(layers.Layer):
    def __init__(self, maxlen, n_sector, embed_dim, **kwargs):
        """
        Parameters:
            - maxlen: the maximal length of the look-back period 
            - n_sector: the number of sectors
            - embed_dim: the dimension of the embedding vector
        """
        super(InputEmbedding, self).__init__(**kwargs)
        self.pos_emb = PositionalEmbedding(maxlen, embed_dim)
        self.stock_emb = StockEmbedding(embed_dim)
        self.sec_emb = SectorEmbedding(n_sector,embed_dim)

    def call(self, x):
        """
        Parameters:
            - x: input tensor of shape (n_stock, dim, period). 
                - x[i,0,t]: sector of stock i.
                - x[i,1,t]: return of stock i in period t
                - x[i,2,t]: market cap change of stock i in period t
                - x[i,3,t]: volume change of stock i in period t
        """
        positions = self.pos_emb(x)
        sectors = self.sec_emb(x[:,:,0,:])
        stocks = self.stock_emb(x[:,:,1:,:])
        return positions + sectors + stocks
    
class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, **kwargs):
        super(TransformerBlock, self).__init__(**kwargs)
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output)
        return self.layernorm2(out1 + ffn_output)
    
class MeanLayer(layers.Layer):
    def __init__(self,rate = 0.1, **kwargs):
        super(MeanLayer, self).__init__(**kwargs)
        self.layer1 = layers.Dense(1)
        self.layer2 = layers.Dense(1)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs):
        x = self.layer1(inputs)
        x = ops.squeeze(x,axis=-1)
        x = self.dropout1(x)
        x = self.layer2(x)
        x = ops.squeeze(x,axis=-1)
        return self.dropout2(x)

class StandardDeviationLayer(layers.Layer):
    def __init__(self,rate = 0.1, **kwargs):
        super(StandardDeviationLayer, self).__init__(**kwargs)
        self.layer1 = layers.Dense(1)
        self.layer2 = layers.Dense(1)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs):
        x = self.layer1(inputs)
        x = ops.squeeze(x,axis=-1)
        x = self.dropout1(x)
        x = self.layer2(x)
        x = ops.squeeze(x,axis=-1)
        return self.dropout2(x)

class OutputLayer(layers.Layer):
    def __init__(self,rate = 0.1, **kwargs):
        super(OutputLayer, self).__init__(**kwargs)
        self.mu_layer = MeanLayer(rate)
        self.sd_layer = StandardDeviationLayer(rate)
        self.concat = layers.Concatenate()
    
    def call(self, inputs):
        mu = self.mu_layer(inputs)
        sd = self.sd_layer(inputs)
        return self.concat([mu,sd])

def build_model(n_stock: int = 263,
                n_sector: int = 9,
                maxlen: int = 5,
                emb_dim: int = 32,
                num_heads: int = 2,
                ff_dim: int = 32,
                rate = 0.1):
    inputs = layers.Input(shape=(n_stock,4,maxlen))
    embedding_layer = InputEmbedding(maxlen,n_sector,emb_dim)
    x = embedding_layer(inputs)
    transformer = TransformerBlock(emb_dim,num_heads,ff_dim,rate)
    x = transformer(x)
    output_layer = OutputLayer(rate)
    outputs = output_layer(x)
    return keras.Model(inputs=inputs, outputs=outputs)

@keras.saving.register_keras_serializable()
def gaussian_log_like_loss(y_true,y_pred):
    """
    Parameters:
        - y_true: The next-day return of each stock
        - y_pred: the predicted next-day mean and standard deviation of return
    """
    n = ops.shape(y_pred)[-1]
    d = n//2
    mu = y_pred[:,:d]
    sd = y_pred[:,d:]
    return ops.mean(ops.square(y_true-mu)/(1e-6+ops.square(sd))+ops.log(1e-6+ops.square(sd)),axis=-1)
