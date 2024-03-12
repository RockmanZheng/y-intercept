import numpy as np
import keras
import torch
import matplotlib.pyplot as plt


def calc_opt_portfolio(model: keras.Model,
                       inputs: np.ndarray,
                       target_return: float = 0.1):
    outputs = model.predict(inputs)
    n = outputs.shape[-1]//2
    mu = outputs[:,:n]
    sd = outputs[:,n:]
    return mu/sd**2*target_return/np.sum(mu**2/sd**2,axis=-1).reshape(-1,1)

def calc_return(portfolios: np.ndarray,
                rates: np.ndarray):
    return np.cumprod(1+np.sum(portfolios * rates,axis = -1))-1

if __name__ == '__main__':
    model = keras.models.load_model("model.keras")
    _,_,_,_,x_test,y_test = torch.load('data/data.pt')
    portfolios = calc_opt_portfolio(model, x_test)
    RoR = calc_return(portfolios,np.array(y_test))
    plt.plot(RoR)