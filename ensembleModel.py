import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score
from sklearn.ensemble import VotingClassifier



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
    #mlp.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_val, y_val))

    from Models.gbm import gbm_model

    gbm = gbm_model()
    #gbm.fit(X_train, y_train)

    from Models.lasso import lasso_regression_model
    lasso = lasso_regression_model()
    #lasso.fit(X_train, y_train)

    """
    # Get class probabilities from each model
    probabilities_model_mlp = get_probabilities(mlp, X_test)
    probabilities_model_gbm = get_probabilities(gbm, X_test)
    probabilities_model_lasso = get_probabilities(lasso, X_test)

    # Combine class probabilities from all models
    all_probabilities = [
        np.round(probabilities_model_mlp).astype(int).reshape(-1),  # Reshape to 1D array
        probabilities_model_gbm.flatten(),  # Flatten to 1D array
        probabilities_model_lasso.flatten()  # Flatten to 1D array
    ]    
    average_probabilities = np.mean(all_probabilities, axis=0)
    print(all_probabilities)

    
    # Choose the class with the highest average probability 
    final_predictions = np.argmax(average_probabilities)
    print(final_predictions)
    
    accuracy = accuracy_score(y_test, final_predictions)
    print(f'Ensemble Accuracy: {accuracy * 100:.2f}%')"""


    voting_classifier = VotingClassifier(estimators=[('lr', gbm), ('svc', lasso), ('dt', gbm)], voting='soft')

    # Train the ensemble model
    voting_classifier.fit(X_train, y_train)

    # Make predictions
    predictions = voting_classifier.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, predictions)
    print(f'Ensemble Accuracy: {accuracy * 100:.2f}%')

    # Evaluate the ensemble's accuracy
    """
    accuracy = accuracy_score(y_test, np.round(probabilities_model_mlp).astype(int))
    print(f'mlp Accuracy: {accuracy * 100:.2f}%')

    accuracy = accuracy_score(y_test, probabilities_model_gbm)
    print(f'gbm Accuracy: {accuracy * 100:.2f}%')


    accuracy = accuracy_score(y_test, probabilities_model_lasso)
    print(f'lasso Accuracy: {accuracy * 100:.2f}%')"""


    

def main(data_size):

    from preprocess import run
    X_train,y_train,X_val,y_val,X_test,y_test = run(data_size)
    ensemble_model(X_train,y_train,X_val,y_val,X_test,y_test)


main(10000)