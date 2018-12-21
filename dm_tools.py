import defines
import numpy as np
import matplotlib.pyplot as plt
import plots
import statsmodels.api as sm
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge


# check corelation - basic, plot scatter_plot
def correlation_graphs(data_frame,Ef,Ebg):
    graph_ind = 1;
    for x in data_frame:
        if x == "E_bg" or x == "E_f": continue
        plt.figure(graph_ind)
        if Ef==1:
            plots.scatter_plot(data_frame=data_frame,
                               x_name=x, y_name="E_f",
                               color='blue',
                               marker='o',
                               title="Corelation with " + x,
                               xlabel="blue for Ef and red for E_bg",
                               ylabel="blue for Ef and red for E_bg",
                               ind=graph_ind)
        if Ebg==1:
            plots.scatter_plot(data_frame=data_frame,
                               x_name=x,
                               y_name="E_bg",
                               color='red',
                               marker='o',
                               title="Corelation with " + x,
                               xlabel="blue for Ef and red for E_bg",
                               ylabel="blue for Ef and red for E_bg",
                               ind=graph_ind + 1)
        graph_ind += 2

#simple linear regression
def linearReg(param_df,target_df):
    X = param_df            #input variables
    y = target_df           #target variable
    lm = linear_model.LinearRegression()
    model = lm.fit(X, y)
    predictions = lm.predict(X)
    return predictions, model

#simple log regression - dosnt work,
# from net: use polynomial https://stackoverflow.com/questions/51043573/logistic-regression-valueerror-unknown-label-type-continuous
def log(param_df,target_df):
    X = param_df            #input variables
    y = target_df           #target variable
    logreg = LogisticRegression()
    print X
    model=logreg.fit(X, y)
    predictions = logreg.predict(X)
    return predictions, model

#simple polynomial regression
def poly(param_df,target_df,deg):
    X = param_df  # input variables
    y=target_df
    poly = PolynomialFeatures(deg)
    x_transform=poly.fit_transform(X)
    lin_regressor = linear_model.LinearRegression()
    model=lin_regressor.fit(x_transform, y)
    predictions = lin_regressor.predict(x_transform)
    return predictions, model

#Ordinary Least Squares regression (linear type)
def OLS_MODEL(param_df,target_df):
    X = param_df            #input variables
    y = target_df           #target variable
    # X = sm.add_constant(X)  #let's add an intercept (beta_0) to our model
    model = sm.OLS(y, X).fit()  ## sm.OLS(output, input)
    predictions = model.predict(X)
    summary= model.summary()
    # print summary         # Print out the statistics
    return predictions, model

#lasso regression
def lasso(param_df,target_df,alpha):
    X=param_df
    y=target_df
    lassoreg = Lasso(alpha=alpha, normalize=False, max_iter=2000,random_state=None)#1e5)
    model=lassoreg.fit(param_df, target_df)
    predictions = lassoreg.predict(param_df)
    return predictions, model

#lasso regression
def ridge(param_df,target_df,alpha):
    X=param_df
    y=target_df
    ridgereg = Ridge(alpha=alpha, normalize=True)
    model=ridgereg.fit(param_df,target_df)
    predictions = ridgereg.predict(param_df)
    return predictions, model





