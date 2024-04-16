import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

# load the training set
path = "data/tvmarketing.csv"
adv = pd.read_csv(path)


# Split the dataset into training 
X = adv["TV"].values.reshape(-1, 1)
y = adv["Sales"].values

# Train the model
lr_model = LinearRegression()
lr_model.fit(X, y)

# Save the model using pickle
with open('model.pkl', 'wb') as file:
    pickle.dump(lr_model, file)