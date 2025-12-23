import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.losses import MeanSquaredLogarithmicError
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pdb import set_trace
import math


def car_to_sph(x, y, z, orig=(0.0, 0.0, 0.0)):
    x -= orig[0]
    y -= orig[1]
    z -= orig[2]
    
    r = math.sqrt(x**2 + y**2 + z**2)
    theta = math.atan2(y, x)
    phi = math.acos(z / r)
    
    return r, theta, phi	

def car_to_sph_list(X,orig=(0.0,0.0,0.0)):
    X_sph = np.zeros_like(X)
    for i,point in enumerate(X):
        X_sph[i] = car_to_sph(point[0], point[1], point[2], orig=orig)
    return X_sph

# scaler_X = StandardScaler()
# scaler_y = StandardScaler()


# Sample data preparation for 3D (replace with your actual dataset)
# X_train should contain (x, y, z) coordinates, and y_train should contain binary labels (0 or 1).
# data = np.genfromtxt('./GeneratedData20231024/GeneratedData.csv', delimiter=',')
data = np.genfromtxt('./Points.csv', delimiter=',')
X_train = data[:,:3]
x0,y0,z0 = np.mean(X_train,axis=0)



# X_train0 = car_to_sph_list(X_train,orig=(x0,y0,z0))
X_train0 = X_train



# X_tain0 = scaler_X.fit_transform(X_train0)
gns = data[:,5]
y_train = (gns>0).astype(int)
# set_trace()
# y_train2=scaler_y.fit_transform(gns.reshape(-1, 1))
y_train2=gns
idxs = np.where(np.abs(gns)<1e-2)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_train0, y_train, test_size=0.2, random_state=42)
X_train2, X_test2, y_train2, y_test2 = train_test_split(X_train0, y_train2, test_size=0.2, random_state=42)
# X_train = X_train[idxs]
# y_train = y_train[idxs]


# Define the neural network architecture for 3D
# model1 = keras.Sequential([
#     keras.layers.Dense(16, activation='relu', input_shape=(3,)),  # Input shape is (3,) for (x, y, z) coordinates
#     keras.layers.Dense(8, activation='relu'),
#     keras.layers.Dense(1, activation='sigmoid')
# ])


model2 = keras.Sequential([
    keras.layers.Dense(32, activation='sigmoid', input_shape=(3,)),  # Input shape is (3,) for (x, y, z) coordinates
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(1)
])



# Compile the model
# model1.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# model2.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
model2.compile(optimizer='adam', loss=MeanSquaredLogarithmicError(), metrics=['mae'])

# Train the model
# model1.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)


# set_trace()

history = model2.fit(X_train2, y_train2, epochs=10, batch_size=32, validation_data=(X_test2, y_test2))

# Evaluate the model on the test set
# test_loss, test_accuracy = model1.evaluate(X_test, y_test)
test_loss2, test_accuracy2 = model2.evaluate(X_test2, y_test2)
# print(f"Test Accuracy: {test_accuracy}")


predictions = model2.predict(X_test2)


set_trace()



print(f"Test Accuracy: {test_accuracy2}")

# Make predictions for 3D data
# Generate 3D data for prediction
X_prediction = np.random.rand(10, 3)  # Replace with your 3D data
# predictions = model1.predict(X_prediction)
