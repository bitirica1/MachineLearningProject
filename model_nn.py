import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. REUSED CODE: Data Loading & Preprocessing ---
def load_data(df, year):
    # Select features and target variable
    feature_cols = ['season','mnth','holiday','weekday','workingday','weathersit','temp', 'atemp', 'hum', 'windspeed']
    target_col = 'cnt'

    # extract categorical columns and apply one-hot encoding
    categorical_cols = ['season','mnth','weekday','weathersit','hr']
    # Note: 'hr' is in the dataset but not in feature_cols list above, 
    # but let's stick to your colleague's exact logic for consistency.
    # If 'hr' causes an error because it's missing from input, we will ignore it, 
    # but based on the provided snippet, this logic extracts categorical data well.
    encoder = OneHotEncoder(drop='first', sparse_output=False).set_output(transform='pandas')
    category_encoded = encoder.fit_transform(df[categorical_cols])

    #extract numeric columns
    numeric_cols = ['holiday','workingday','temp', 'atemp', 'hum', 'windspeed']
    # numeric_data = df[numeric_cols].to_numpy() # Not strictly needed if we concat below

    #combine encoded categorical and numeric data
    columns_to_drop = ['instant', 'dteday', 'casual', 'registered', 'cnt', 'season','mnth','weekday','weathersit']
    x = pd.concat([df, category_encoded], axis=1).drop(columns= columns_to_drop).to_numpy()
    y = df[target_col].values

    return x, y, encoder

# Load the dataset
path = "bike+sharing+dataset/hour.csv"
df = pd.read_csv(path)

# Prepare X and y using the reused function
print("Loading and encoding data...")
x, y, _ = load_data(df, year=0)

# --- 2. SPLITTING & SCALING (Crucial for Neural Networks) ---

# Split: 60% train, 20% validation, 20% test (Same ratios as colleague)
x_train, x_temp, y_train, y_temp = train_test_split(x, y, test_size=0.4, random_state=42)
x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.5, random_state=42)

# Scale the data
# Neural Networks struggle if data isn't normalized (e.g., counts are 0-1000, temp is 0-1).
# We use StandardScaler to make mean=0 and std=1
scaler = StandardScaler()

# Fit only on training data to avoid data leakage
x_train_scaled = scaler.fit_transform(x_train)
x_val_scaled = scaler.transform(x_val)
x_test_scaled = scaler.transform(x_test)

print(f"Data Shapes: Train: {x_train_scaled.shape}, Val: {x_val_scaled.shape}, Test: {x_test_scaled.shape}")

# --- 3. BUILD THE TENSORFLOW NEURAL NETWORK ---

# We use a 'Sequential' model (a stack of layers)
model = keras.Sequential([
    # Input Layer: Implicitly defined by input_shape in the first layer
    
    # Hidden Layer 1: 
    # 32 Neurons. 'relu' activation makes it non-linear (better than linear regression)
    layers.Dense(32, activation='relu', input_shape=(x_train_scaled.shape[1],)),
    
    # Hidden Layer 2:
    # 16 Neurons. Adds a bit more depth to learn complex patterns.
    layers.Dense(16, activation='relu'),
    
    # Output Layer:
    # 1 Neuron because we are predicting ONE number (cnt).
    # No activation (linear) because we want the raw number, not a probability.
    layers.Dense(1)
])

# Compile the model
# Optimizer: Adam is the standard "good default" for NNs
# Loss: Mean Squared Error (MSE) - same math as your colleague's cost function
model.compile(
    optimizer='adam',
    loss='mse',
    metrics=['mae'] # We also track Mean Absolute Error to see how many bikes we are off by on average
)

# --- 4. TRAIN THE MODEL ---
print("Starting training...")
history = model.fit(
    x_train_scaled, 
    y_train,
    validation_data=(x_val_scaled, y_val),
    epochs=50,          # How many times to go through the dataset
    batch_size=32,      # Update weights after every 32 samples
    verbose=1
)

# --- 5. EVALUATION & PLOTTING (Adapted from colleague's code) ---

# Evaluate on test set
loss, mae = model.evaluate(x_test_scaled, y_test, verbose=0)
print(f"\nTest Set Results:")
print(f"Mean Squared Error (MSE): {loss:.2f}")
print(f"Mean Absolute Error (MAE): {mae:.2f} (On average, predictions are off by this many bikes)")

# Make predictions
predictions = model.predict(x_test_scaled).flatten()

# Plot Training History (Cost over time)
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss (MSE) over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

# Plot Predictions vs Actual (Scatter Plot)
plt.subplot(1, 2, 2)
plt.scatter(y_test, predictions, alpha=0.4, s=10)
plt.xlabel("Actual Count")
plt.ylabel("Predicted Count")
plt.title("Neural Network: Predicted vs Actual")
plt.grid(True)

# Add a perfect-fit reference line
max_val = max(y_test.max(), predictions.max())
plt.plot([0, max_val], [0, max_val], color='red', linewidth=2, label='Perfect Fit')
plt.legend()

plt.tight_layout()
plt.show()