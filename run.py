from prepare_data import train_features, train_targets, val_features, val_targets, scaled_features, \
                         test_targets, test_features, test_data, rides
from neuralnet import NeuralNetwork

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#  HYPERPARAMETERS
EPOCHS = 2500
LEARNING_RATE = 1.
BATCH_SIZE = 200
HIDDEN_NODES = 8


network = NeuralNetwork(train_features.shape[1], 1, HIDDEN_NODES, LEARNING_RATE)
network.train(train_features.values, train_targets['cnt'].values,
             EPOCHS, val_features.values, val_targets['cnt'].values, BATCH_SIZE)

losses = network.losses

plt.plot(losses['train'], label='Training loss', color='red')
plt.plot(losses['validation'], label='Validation loss', color='tomato')
plt.legend()
_ = plt.ylim()


fig, ax = plt.subplots(figsize=(8, 4))
fig.suptitle('Predictions for December')

mean, std = scaled_features['cnt']
predictions = np.array(network.predict(test_features)).T * std + mean
ax.plot(predictions[0], label='Prediction', color='red')

ax.plot((test_targets['cnt']*std + mean).values, label='Data', color='royalblue')
ax.set_xlim(right=len(predictions))
ax.legend()

dates = pd.to_datetime(rides.iloc[test_data.index]['dteday'])
dates = dates.apply(lambda d: d.strftime('%b %d'))
ax.set_xticks(np.arange(len(dates))[12::24])
_ = ax.set_xticklabels(dates[12::24].apply(lambda x: x[-2:]), rotation=45)
ax.set_xlabel('Date')

plt.show()