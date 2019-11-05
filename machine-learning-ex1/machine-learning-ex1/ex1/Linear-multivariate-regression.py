# To add a new cell, type '#%%'
# To add a new markdown cell, type '#%% [markdown]'
#%%
from IPython import get_ipython

#%% [markdown]
# # Multivariate Linear Regression
# ### Hypothesis
# $$ h_{\theta}(x_i) = \theta_0 + \theta_1 x_1 + \theta_2 x_2 + ... + \theta_n x_n $$
# ### Cost Function
# $$ J(\theta) = \frac{1}{2 m} \sum_{i=1}^{m} ( h_{\theta}(x_i) - y_i)^2 $$
# 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


#%%
data = pd.read_csv('ex1data2.txt', sep=',', header=None)
data.columns = ["House Size", "No of Bedrooms", "Price"]
m = data.size
data.head()


#%%
data.describe()

#%% [markdown]
# ## Feature Scaling and Mean Normalization
# ### Mean Normalization
# $ x_j := x_j - \mu_j $ where $ \mu_j $ is the average of all the values for feature
# 
# ### Feature Scaling
# $ x_j := \frac{x_j}{s_j} $ where $ s_j $ is the range/standard deviation of all the values for feature j
# 
# We do both, so
# $$ x_j := \frac{x_j - \mu_j}{s_j} $$

#%%
X = data.iloc[:, 0:2]
y = data.iloc[:, 2]
theta = np.array([0.0]*3)

mean = X.mean()
std_ = X.std()
X = X-mean
for col in X:
    X.loc[:, col] = X[col]/std_[col]
X.insert(0, 'x_0', 1.0)
X.head()


#%%
def computeCost(X, y, theta):
    m = y.size
    return 1/(2*m) * np.sum( np.power( X.dot(theta) - y, 2 ) )


#%%
def grad_desc(X, y, theta, iter=200, alpha=0.1):
    m = y.size
    cost_history = np.zeros(iter)
    theta_history = np.zeros((iter, theta.size))
    for it in range(iter):
        theta = theta - alpha/m * X.T.dot((X.dot(theta)-y))
        theta_history[it, :] = theta.T
        cost_history[it] = computeCost(X, y, theta)
    return theta, theta_history, cost_history


#%%



#%%
theta, thetas, costs = grad_desc(X, y, theta, iter= 50, alpha=0.3)
print("Cost:", computeCost(X, y, theta))
theta

#%% [markdown]
# ## Learning Curve

#%%
plt.plot(np.arange(0, 50, 1), costs)
plt.show()

#%% [markdown]
# ## Predicting Prices

#%%
price = theta['x_0'] + (1650-mean['House Size'])/std_['House Size']*theta['House Size'] + (3-mean['No of Bedrooms'])/std_['No of Bedrooms']*theta['No of Bedrooms']
print(price)

#%% [markdown]
# ## Solving with Normal Equation

#%%
X_ = data.iloc[:, 0:2]
X_.insert(0, 'x_0', 1.0)
y_ = data.iloc[:, 2]
theta_n = np.linalg.inv(X_.T.dot(X_)).dot(X_.T).dot(y_)
price_n = theta_n[0] + 1650 * theta_n[1] + 3 * theta_n[2]
print(theta, theta_n)
print(price_n)


#%%
# hs = 800-4500, #b = 2-6
hs = np.linspace(800, 4500, 100)
b_ = np.linspace(1, 6, 6)
hs, b_ = np.meshgrid(hs, b_)
prices_ = hs+b_
for i in range(6):
    for j in range(100):
        prices_[i][j] = theta['x_0'] + (hs[i][j]-mean['House Size'])/std_['House Size']*theta['House Size'] + (b_[i][j]-mean['No of Bedrooms'])/std_['No of Bedrooms']*theta['No of Bedrooms']


#%%
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(hs, b_, prices_)
ax.scatter(X_.iloc[:, 1], X_.iloc[:, 2], y, c='r')
plt.show()


#%%


