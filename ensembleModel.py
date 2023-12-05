import requests
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers,models
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.optimizers import Adam
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score



# Soft Voting Ensemble Model

# This Soft Voting Ensemble combines the predictions from the MLP, GBM, and third model. 
# Each model is independently trained on the training data, and the probability distributions for each class 
# are averaged across all models.
# The class with the highest average probability is chosen as the final prediction. 
# The ensemble's performance is evaluated against the test set.


# Function to get all the class probabilities from a model
def get_probabilities(model, data):
    predictions = model.predict(data)
    return predictions


# Train the models
mlp_model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val))
gbm_model.fit(X_train, y_train)
model_3.fit(X_train, y_train)

# Get class probabilities from each model
probabilities_model_mlp = get_probabilities(mlp_model, X_test)
probabilities_model_gbm = get_probabilities(gbm_model, X_test)
probabilities_model_3 = get_probabilities(model_3, X_test)

# Combine class probabilities from all models
all_probabilities = [probabilities_model_mlp, probabilities_model_gbm, probabilities_model_3]
average_probabilities = np.mean(all_probabilities, axis=0)


# Choose the class with the highest average probability 
final_predictions = np.argmax(average_probabilities, axis=1)

# Evaluate the ensemble's accuracy
accuracy = accuracy_score(y_test, final_predictions)
print(f'Ensemble Accuracy: {accuracy}')

