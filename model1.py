import numpy as np
import tensorflow as tf
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import copy
import math

#variables to control what we want the code to do
generatePlots = False

#selects data used to predict bike rentals and the actual value measured (uses one-hot encoding for categorical variables)
def load_data(df, year):
    # Select features and target variable
    feature_cols = ['season','mnth','holiday','weekday','workingday','weathersit','temp', 'atemp', 'hum', 'windspeed']
    target_col = 'cnt'

    # extract categorical columns and apply one-hot encoding
    categorical_cols = ['season','mnth','weekday','weathersit','hr']
    encoder = OneHotEncoder(drop='first', sparse_output=False).set_output(transform='pandas')
    category_encoded = encoder.fit_transform(df[categorical_cols])

    #extract numeric columns
    numeric_cols = ['holiday','workingday','temp', 'atemp', 'hum', 'windspeed']
    numeric_data = df[numeric_cols].to_numpy()

    #combine encoded categorical and numeric data
    columns_to_drop = ['instant', 'dteday', 'casual', 'registered', 'cnt', 'season','mnth','weekday','weathersit']
    x = pd.concat([df, category_encoded], axis=1).drop(columns= columns_to_drop).to_numpy()
    y = df[target_col].values

    return x, y, encoder


path = "bike+sharing+dataset/hour.csv"
df = pd.read_csv(path)
print(df.shape)
print(df.columns.tolist())

#x is data used to predict y
x, y, _ = load_data(df, year=0)


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
    # avoid division by zero on constant features
    sigma_safe = np.where(sigma == 0, 1, sigma)

    X_norm = (X - mu) / sigma_safe
    return X_norm, mu, sigma_safe
 


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

    #faster vectorized version of calculations
    predicted = x @ w + b 
    total_cost = (predicted - y) ** 2
    total_cost = np.sum(total_cost) / (2*m)
    # for i in range(0,m):
    #     predicted_i = np.dot(x[i], w)  + b 
    #     cost_i = (predicted_i - y[i]) ** 2
    #     total_cost = total_cost + cost_i
    # total_cost = total_cost / (2*m)

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

    #faster version using vectorization
    predicted = x @ w + b
    error = predicted - y
    dj_dw = (x.T @ error) / m
    dj_db = np.sum(error) / m
    # ### START CODE HERE ### 
    # for i in range(m):
    #     predicted_i = np.dot(w, x[i]) + b
    #     gradient_w += (predicted_i - y[i]) * x[i]
    #     gradient_b += (predicted_i - y[i])
    # ### END CODE HERE ### 
    # dj_db = gradient_b / m
    # dj_dw = gradient_w / m
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

def eval_mse(y, yhat):
    """ 
    Calculate the mean squared error on a data set.
    Args:
      y    : (ndarray  Shape (m,) or (m,1))  target value of each example
      yhat : (ndarray  Shape (m,) or (m,1))  predicted value of each example
    Returns:
      err: (scalar)             
    """
    m = len(y)
    err = 0.0
    for i in range(m):
    ### START CODE HERE ### 
        err += (yhat[i] - y[i])**2
    ### END CODE HERE ### 
    err/= 2*m
    return(err)



#run linear regression

# some gradient descent settings
iterations = 1500 #in all cases at around 4000 iterations the cost converges for alpha = 0.001
alpha = 0.001

# x_norm, mu, sigma = zscore_normalize_features(x)

# Split input data into training, cross-validation, and test sets - 60% train, 20% cv, 20% test
x_train, X_, y_train, Y_ = train_test_split(x, y, test_size=0.4, random_state=42)
x_cv, x_test, y_cv, y_test = train_test_split(X_, Y_, test_size=0.5, random_state=42)   


#want to have a loop that generates multiple polynomial degrees and selects the best one based on cv error
max_degree = 3 # degree 3 performed the best till now
err_train = np.zeros(max_degree)
err_cv = np.zeros(max_degree)

#for alpha 0.01, iterations 1500 best degree is 3

findDegree = True
sklearnModelOnly = True 
if findDegree:
    for degree in range(1,max_degree):
        if sklearnModelOnly: #is faster
            poly = PolynomialFeatures(degree=degree, include_bias=False)
            x_train_poly = poly.fit_transform(x_train)
            model = LinearRegression()
            model.fit(x_train_poly, y_train)
            y_train_pred_model = model.predict(x_train_poly)
            mse_model = mean_squared_error(y_train, y_train_pred_model)/2
            print("Sklearn model train MSE for degree", degree, ":", mse_model)
            x_cv_poly = poly.transform(x_cv)
            y_cv_pred_model = model.predict(x_cv_poly)
            mse_model_cv = mean_squared_error(y_cv, y_cv_pred_model)/2
            print("Sklearn model cv MSE for degree", degree, ":", mse_model_cv)

#final model with best degree

# some gradient descent settings
iterations = 1500 #in all cases at around 4000 iterations the cost converges for alpha = 0.001
alpha = 0.05

best_degree = 2 #obtained from skelearn model above
poly = PolynomialFeatures(degree=best_degree, include_bias=False)
x_train_poly = poly.fit_transform(x_train)
x_train_norm, mu, sigma = zscore_normalize_features(x_train_poly)
initial_w = np.zeros(x_train_poly.shape[1], dtype=float)
initial_b = 0.0
w,b,J_history,_ = gradient_descent(x_train_norm ,y_train, initial_w, initial_b,
                    compute_cost, compute_gradient, alpha, iterations)
print("Gradient results for weight vector w:", w)

plt.plot(J_history)
plt.xlabel("Iteration")
plt.ylabel("Cost (J)")
plt.title("Gradient Descent Convergence")
plt.grid(True)
plt.show()

# Evaluate on test set
x_test_poly = poly.transform(x_test)
x_test_poly_norm = (x_test_poly - mu) / sigma
m = x_test_poly.shape[0]
predicted_test = x_test_poly_norm.dot(w) + b
test_error = mean_squared_error(y_test, predicted_test)/2
print("Test set Mean Squared Error for degree", best_degree, ":", test_error)
plt.figure(figsize=(7,7))

plt.scatter(y_test, predicted_test, alpha=0.4)
plt.xlabel("Actual Count")
plt.ylabel("Predicted Count")
plt.title(f"Predicted vs Actual (degree {best_degree})")
plt.grid(True)

# Add a perfect-fit reference line
max_val = max(y_test.max(), predicted_test.max())
plt.plot([0, max_val], [0, max_val], color='red', linewidth=2)

plt.show()