import torch
from model import *

x_train,y_train,x_val,y_val,x_test,y_test = torch.load('data/data.pt')
model = build_model(n_stock=x_train.shape[1])
model.compile(optimizer="adam", loss=gaussian_log_like_loss)
history = model.fit(
    x_train, y_train, batch_size=32, epochs=11, validation_data=(x_val, y_val)
)