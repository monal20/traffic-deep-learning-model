import numpy as np
import tensorflow as tf
import pandas as pd
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score
from sklearn.ensemble import VotingClassifier
from sklearn.neural_network import MLPClassifier



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


    mlp = mlp_model()
   

    from Models.gbm import gbm_model

    gbm = gbm_model()

    

    from Models.lasso import lasso_regression_model
    lasso = lasso_regression_model()

    
    voting_classifier = VotingClassifier(estimators=[('lr', mlp), ('svc', lasso), ('dt', gbm)], voting='soft')

    # Train the ensemble model
    voting_classifier.fit(X_train, y_train)
    return voting_classifier


def get_probabilities_with(voting_classifier,X_test,y_test):
    # Make predictions
    probabilities = voting_classifier.predict_proba(X_test)
    # Calculate accuracy
    

    correct_top3_count = 0

    # Display the top 3 probabilities for each prediction and check if the true label is in the top 3
    for i, (true_label, probs) in enumerate(zip(y_test, probabilities)):
        top3_indices = np.argsort(probs)[-15:][::-1]  # Get indices of top 3 probabilities
        top3_probs = probs[top3_indices]
        
        print(f"Instance {i + 1}: True Label: {true_label}")
        
        for j, (index, prob) in enumerate(zip(top3_indices, top3_probs), 1):            
            # Check if the true label is in the top 3
            if index == true_label:
                correct_top3_count += 1

    # Calculate accuracy for the true label being in the top 3 on the test set
    accuracy_top3 = correct_top3_count / len(y_test)
    print(f'\nAccuracy for True Label in Top 5: {accuracy_top3 * 100:.2f}%')



def get_probabilities_without(model, X_test, top_k=3):
    # Make predictions
    probabilities = model.predict_proba(X_test)

    # Display the top k probabilities for each prediction
    top_indices = np.argsort(probabilities[0])[-top_k:][::-1]  # Get indices of top k probabilities
    top_probs = probabilities[0][top_indices]


    array = []
    for j, (index, prob) in enumerate(zip(top_indices, top_probs), 1):
        print(f"  Top {j}: Class {index}, Probability: {prob:.4f}")
        data = index,prob
        array.append(data)

    return array
            
    

def testing(data_size,test_to_compute):

    from preprocess import run

    X_train,y_train,X_val,y_val,X_test,y_test = run(data_size)
    model = ensemble_model(X_train,y_train,X_val,y_val,X_test,y_test)
    array = get_probabilities_without(model,test_to_compute, 10)

    return array
    

def main(data_size):

    from preprocess import run
    X_train,y_train,X_val,y_val,X_test,y_test = run(data_size)
    model = ensemble_model(X_train,y_train,X_val,y_val,X_test,y_test)
    get_probabilities_with(model,X_test,y_test)

