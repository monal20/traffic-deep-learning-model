from sklearn.linear_model import LogisticRegression


 ################## Lasso Regression with L2 Regularization ###############


def lasso_regression_model():
    #Penalty
    alpha_reg = 0.01

    #Build the model
    lasso_reg = LogisticRegression(penalty='l2', C= alpha_reg, solver='saga')
    return lasso_reg