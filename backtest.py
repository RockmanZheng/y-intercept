import numpy as np
import keras
import torch
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

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

    stock_df = pd.read_csv('data/data_last.csv')
    tmp = stock_df[stock_df['ticker']=='1332 JT']
    date_list = tmp['date'].apply(lambda x: datetime.strptime(x,'%Y-%m-%d'))
    time_range = date_list.iloc[-len(RoR):]
    print(time_range)
    plt.plot(time_range,RoR*100)
    plt.title('Transformer-Based Mean-Variance Portfolio Strategy')
    plt.ylabel('Return [%]')
    plt.savefig('backtest.png')