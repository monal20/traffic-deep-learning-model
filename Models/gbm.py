import sys
from pathlib import Path
from catboost import CatBoostClassifier

sys.path.append(str(Path(__file__).resolve().parent.parent))


def gbm_model():
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
    return model
    # Train the model, directly passing the validation set





