import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
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


def ensemble_model(X_train,y_train,X_val,y_val,X_test,y_test):
# Train the models
    from Models.mlp import mlp_model

    mlp = mlp_model(X_train)
    mlp.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val))

    from Models.gbm import gbm_model

    gbm = gbm_model(X_train,y_train)
    gbm.fit(X_train, y_train)


    from Models.lasso import lasso_regression_model
    lasso = lasso_regression_model()
    lasso.fit(X_train, y_train)

    # Get class probabilities from each model
    probabilities_model_mlp = get_probabilities(mlp, X_test)
    probabilities_model_gbm = get_probabilities(gbm, X_test)
    probabilities_model_lasso = get_probabilities(lasso, X_test)

    # Combine class probabilities from all models
    all_probabilities = [probabilities_model_mlp, probabilities_model_gbm, probabilities_model_lasso]
    average_probabilities = np.mean(all_probabilities, axis=0)


    # Choose the class with the highest average probability 
    final_predictions = np.argmax(average_probabilities, axis=1)

    # Evaluate the ensemble's accuracy
    accuracy = accuracy_score(y_test, final_predictions)
    print(f'Ensemble Accuracy: {accuracy}')


def main(data_size):

    from preprocess import run
    X_train,y_train,X_val,y_val,X_test,y_test = run(data_size)
    ensemble_model(X_train,y_train,X_val,y_val,X_test,y_test)


main(2000)