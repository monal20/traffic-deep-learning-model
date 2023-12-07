from sklearn.linear_model import LogisticRegression


 ################## Lasso Regression with L2 Regularization ###############


def lasso_regression_model():
    #Penalty
    alpha_reg = 0.01

    #Build the model
    lasso_reg = LogisticRegression(penalty='l2', C= alpha_reg, solver='saga',verbose=1,max_iter=1000)
    return lasso_reg