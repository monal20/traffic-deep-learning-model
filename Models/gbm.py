import sys
from pathlib import Path
from catboost import CatBoostClassifier
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
sys.path.append(str(Path(__file__).resolve().parent.parent))
from DeepEnsemble import X, y, X_train,y_train, X_test, y_test, X_val, y_val

#cat_features = [i for i in X.columns if X[i].dtype == 'object'] not sure if I should add this line, we have the columns as 

# Initialize the CatBoost Classifier
model = CatBoostClassifier(
    iterations=1000, 
    learning_rate=0.1,
    depth=6,
    loss_function='MultiClass',
    #cat_features=cat_features, bleh
    verbose=True
)

# Train the model, directly passing the validation set
model.fit(X_train, y_train) # here, I tried to add eval_set =(X_val, y_val) but keep getting the Dataset test #0 contains class label "SPAULDING AVE" that is not present in the learn dataset


# Prediction
predictions = model.predict_proba(X_test) #Print probabilities

predictionss = model.predict(X_test) #Prints one street (the most risky one)

print(predictions[:2]) #prints probabilities of first 2 tests
print("Streets: ")
print(predictionss[:2]) #Prints street name for firt 2 tests

# Convert predictions to class labels for evaluation
#y_pred = np.argmax(predictions, axis=1)
#y_pred = [model.classes_[index] for index in y_pred]

# Evaluate the model
#print("\n--- Model Evaluation ---\n")

# Confusion Matrix
#conf_matrix = confusion_matrix(y_test, y_pred)
#print("Confusion Matrix:\n", conf_matrix)

# Accuracy
#accuracy = accuracy_score(y_test, y_pred)
#print(f"\nAccuracy: {accuracy}")

# Precision, Recall, F1 Score
#print("\nClassification Report:\n")
#print(classification_report(y_test, y_pred))

