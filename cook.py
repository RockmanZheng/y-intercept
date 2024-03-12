import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import torch

sector_df = pd.read_csv('data/data_sector.csv')
stock_df = pd.read_csv('data/data_last.csv')
mktcap_df = pd.read_csv('data/data_mkt_cap.csv')
vol_df = pd.read_csv('data/data_volume.csv')

# get all symbols that are not active
name_list = [symbol for symbol in sector_df['ticker'] if vol_df['volume'][vol_df['ticker']==symbol].min()==0]
n_stock = len(sector_df)-len(name_list)
print('# Active stock = %i'%(n_stock))

# discard those symbols
sector_df = sector_df[~sector_df['ticker'].isin(name_list)]
stock_df = stock_df[~stock_df['ticker'].isin(name_list)]
mktcap_df = mktcap_df[~mktcap_df['ticker'].isin(name_list)]
vol_df = vol_df[~vol_df['ticker'].isin(name_list)]

# encode sectors
le = LabelEncoder()
le.fit(sector_df['bics_sector'])
sector_encoded = le.transform(sector_df['bics_sector'])

# calculate stock return
return_df = pd.DataFrame(data = {'ticker':stock_df['ticker'].iloc[1:],
                                 'date':stock_df['date'].iloc[1:],
                                 'return': (stock_df['last'].iloc[1:].to_numpy()/stock_df['last'].iloc[0:-1].to_numpy()-1)})
# remove invalid return: the first date of each symbol

# calculate market cap change
mktcap_change_df = pd.DataFrame(data = {'ticker':mktcap_df['ticker'].iloc[1:],
                                 'date':mktcap_df['date'].iloc[1:],
                                 'mcap_change': (mktcap_df['mkt_cap'].iloc[1:].to_numpy()/mktcap_df['mkt_cap'].iloc[0:-1].to_numpy()-1)})

# calculate volume change
vol_change_df = pd.DataFrame(data = {'ticker':vol_df['ticker'].iloc[1:],
                                 'date':vol_df['date'].iloc[1:],
                                 'vol_change': (vol_df['volume'].iloc[1:].to_numpy()/vol_df['volume'].iloc[0:-1].to_numpy()-1)})

date_list = sorted(list(set(return_df['date'].apply(lambda x:datetime.strptime(x,'%Y-%m-%d')).to_list())))

n_sector = 9
maxlen = 5

x_data = []
y_data = []
return_ptr = maxlen
mktcap_ptr = maxlen
vol_ptr = maxlen

for i in tqdm(range(maxlen+1,len(date_list))):
    dt = date_list[i].strftime('%Y-%m-%d')
    y_data.append(return_df['return'][return_df['date']==dt].to_numpy())
    buffer = []
    for j in range(i-maxlen,i):
        ddt = date_list[j].strftime('%Y-%m-%d')
        buf = []
        buf.append(sector_encoded)
        buf.append(return_df['return'][return_df['date']==ddt].to_numpy())
        buf.append(mktcap_change_df['mcap_change'][mktcap_change_df['date']==ddt].to_numpy())
        buf.append(vol_change_df['vol_change'][vol_change_df['date']==ddt].to_numpy())
        buffer.append(np.stack(buf,axis = 1))
    x_data.append(np.stack(buffer,axis = 2))

x_data = np.stack(x_data,axis=0)
y_data = np.stack(y_data,axis=0)

sample_size = y_data.shape[0]
train_size = round(sample_size*0.7)
validation_size = round(sample_size*0.15)
test_size = sample_size - train_size - validation_size
x_train = torch.tensor(x_data[:train_size,:,:,:])
y_train = torch.tensor(y_data[:train_size,:])
t = train_size+validation_size
x_val = torch.tensor(x_data[train_size:t,:,:,:])
y_val = torch.tensor(y_data[train_size:t,:])
x_test = torch.tensor(x_data[t:,:,:,:])
y_test = torch.tensor(y_data[t:,:])
torch.save((x_train,y_train,x_val,y_val,x_test,y_test),'data/data.pt')
print('Cooked data saved to data/data.pt')

