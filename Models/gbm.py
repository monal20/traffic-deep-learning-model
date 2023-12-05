import sys
from pathlib import Path
from catboost import CatBoostClassifier
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
sys.path.append(str(Path(__file__).resolve().parent.parent))


def gbm(X_train,y_train):
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





