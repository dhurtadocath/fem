import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from pdb import set_trace

# Generate some sample data
def generate_data(num_points):
    # Generate random 3D coordinates
    X = np.random.rand(num_points, 3)
    
    # Generate corresponding scalar values (you can replace this with your own scalar field)
    y = np.sin(X[:, 0]) + np.cos(X[:, 1]) + X[:, 2]**2
    
    return X, y

# Create the neural network model
def create_model():
    model = Sequential()
    model.add(Dense(64, input_dim=3, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='linear'))
    model.compile(loss='mean_squared_error', optimizer=Adam(lr=0.001))
    return model

# Generate training data
num_points = 1000
X_train, y_train = generate_data(num_points)

# Create the model
model = create_model()

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=1)

# Now you can use the trained model to predict scalar values for new 3D coordinates
# For example, if you have new_data as a NumPy array with shape (num_samples, 3):
# predictions = model.predict(new_data)

set_trace()


