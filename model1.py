import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.activations import linear, relu, sigmoid
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import copy
import math

#variables to control what we want the code to do
generatePlots = False

#selects data used to predict bike rentals and the actual value measured
def load_data(df, year):
    # Select features and target variable
    feature_cols = ['season','mnth','holiday','weekday','workingday','weathersit','temp', 'atemp', 'hum', 'windspeed']
    target_col = 'cnt'
    df = df[df['yr'] == year] #select only 1 year
    
    x = df[feature_cols].values
    y = df[target_col].values
    
    return x, y


path = "bike+sharing+dataset/day.csv"
df = pd.read_csv(path)
print(df.shape)
print(df.columns.tolist())

#x is data used to predict y
x, y = load_data(df, year=0)


# #check properties of x
# print("PROPERTIES OF X")
# print("Type of x:",type(x))
# print("First five elements of x are:\n", x[:5]) 
# print("Shape of x:", x.shape)

# print("-------------------------")

# #check properties of y
# print("PROPERTIES OF Y")
# print("Type of y:",type(y))
# print("First five elements of y are:\n", y[:5])
# print("Shape of y:", y.shape)


#Generate plots
if generatePlots:
    # # 1. Correlation heatmap
    # plt.figure(figsize=(10, 6))
    # sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
    # plt.title("Feature Correlation Heatmap")
    # plt.show()

    # 2. Pairplot for selected features
    sns.pairplot(df, vars=['season','yr','mnth','holiday','weekday','workingday','weathersit','temp', 'atemp', 'hum', 'windspeed'], kind='scatter', diag_kind='kde')
    plt.show()

#LINEAR REGRESSION

#normalize features
def zscore_normalize_features(X):
    """
    computes  X, zcore normalized by column
    
    Args:
      X (ndarray (m,n))     : input data, m examples, n features
      
    Returns:
      X_norm (ndarray (m,n)): input normalized by column
      mu (ndarray (n,))     : mean of each feature
      sigma (ndarray (n,))  : standard deviation of each feature
    """
    # find the mean of each column/feature
    mu     = np.mean(X, axis=0)                 # mu will have shape (n,)
    # find the standard deviation of each column/feature
    sigma  = np.std(X, axis=0)                  # sigma will have shape (n,)
    # element-wise, subtract mu for that column from each example, divide by std for that column
    X_norm = (X - mu) / sigma      

    return (X_norm, mu, sigma)
 


#cost function
def compute_cost(x, y, w, b): 
    """
    Computes the cost function for linear regression.
    
    Args:
        x (ndarray): Shape (m,) Input to the model (Population of cities) 
        y (ndarray): Shape (m,) Label (Actual profits for the cities)
        w (ndarray): Shape given by number of features Parameters of the model
        b (scalar): Bias parameter of the model 
    
    Returns
        total_cost (float): The cost of using w,b as the parameters for linear regression
               to fit the data points in x and y
    """
    # number of training examples
    m = x.shape[0] 
    total_cost = 0

    for i in range(0,m):
        predicted_i = np.dot(x[i], w)  + b 
        cost_i = (predicted_i - y[i]) ** 2
        total_cost = total_cost + cost_i
    total_cost = total_cost / (2*m)

    return total_cost

def compute_gradient(x, y, w, b): 
    """
    Computes the gradient for linear regression 
    Args:
      x (ndarray): Shape (m,n) Input to the model (Population of cities) 
      y (ndarray): Shape (m,) Label (Actual profits for the cities)
      w, b (scalar): Parameters of the model  
    Returns
      dj_dw (scalar): The gradient of the cost w.r.t. the parameters w
      dj_db (scalar): The gradient of the cost w.r.t. the parameter b     
     """
    
    # Number of training examples
    m = x.shape[0]
    n = x.shape[1]
    gradient_w = np.zeros(n)
    gradient_b = 0

    ### START CODE HERE ### 
    for i in range(m):
        predicted_i = np.dot(w, x[i]) + b
        gradient_w += (predicted_i - y[i]) * x[i]
        gradient_b += (predicted_i - y[i])
    ### END CODE HERE ### 
    dj_db = gradient_b / m
    dj_dw = gradient_w / m
    return dj_dw, dj_db

def gradient_descent(x, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters): 
    """
    Performs batch gradient descent to learn theta. Updates theta by taking 
    num_iters gradient steps with learning rate alpha
    
    Args:
      x :    (ndarray): Shape (m,n)
      y :    (ndarray): Shape (m,)
      w_in: (ndarray): Shape (n,) Initial values of parameters of the model 
      b_in : (scalar) Initial values of parameters of the model
      cost_function: function to compute cost
      gradient_function: function to compute the gradient
      alpha : (float) Learning rate
      num_iters : (int) number of iterations to run gradient descent
    Returns
      w : (ndarray): Shape (1,) Updated values of parameters of the model after
          running gradient descent
      b : (scalar)                Updated value of parameter of the model after
          running gradient descent
    """
    
    # number of training examples
    m = x.shape[0]
    
    # number of features
    n = x.shape[1]
    
    # An array to store cost J and w's at each iteration — primarily for graphing later
    J_history = []
    w_history = []
    w = copy.deepcopy(w_in)  #avoid modifying global w within function
    b = float(b_in)
    
    for i in range(num_iters):

        # Calculate the gradient and update the parameters
        dj_dw, dj_db = gradient_function(x, y, w, b )  

        # Update Parameters using w, b, alpha and gradient
        w = w - alpha * dj_dw               
        b = b - alpha * dj_db               

        # Save cost J at each iteration
        if i<100000:      # prevent resource exhaustion 
            cost =  cost_function(x, y, w, b)
            J_history.append(cost)

        # Print cost every at intervals 10 times or as many iterations if < 10
        if i % math.ceil(num_iters/10) == 0:
            w_history.append(w)
            print(f"Iteration {i:4}: Cost {float(cost):8.2f}   ")
        
    return w, b, J_history, w_history #return w and J,w history for graphing




#run linear regression
# initialize fitting parameters. Recall that the shape of w is (n,)
initial_w = np.zeros(x.shape[1], dtype=float)
initial_b = 0.0

# some gradient descent settings
iterations = 1500
alpha = 0.05

x_norm, mu, sigma = zscore_normalize_features(x)
print("mu:", mu)
print("sigma:", sigma)
w,b,_,_ = gradient_descent(x_norm ,y, initial_w, initial_b, 
                     compute_cost, compute_gradient, alpha, iterations)
print("w,b found by gradient descent:", w, b)

m = x.shape[0]
predicted = np.zeros(m)

x, y = load_data(df, year=1)



predicted = np.dot(x, w) + b

plt.scatter(y, predicted, alpha=0.6)
plt.xlabel("Actual values")
plt.ylabel("Predicted values")
plt.title("Actual vs Predicted")
plt.plot([y.min(), y.max()], [predicted.min(), predicted.max()], 'r--')
plt.show()