import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
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

# Create the neural network model
def create_model_old():
    model = Sequential()
    model.add(Dense(128, input_dim=3, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1, activation='linear'))
    model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=0.0001))
    return model
def create_model():
    model = Sequential()
    model.add(Dense(512, input_dim=3, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='linear'))
    model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=0.0001))
    return model



# data = np.genfromtxt('./GeneratedData20231024/GeneratedData.csv', delimiter=',')
# X_train = data[:,:3]
# y_train = data[:,3]

data = np.genfromtxt('./Points.csv', delimiter=',')
X_train = data[:,:3]
y_train = data[:,5]


# To spherical
x0,y0,z0 = np.mean(X_train,axis=0)
X_train = car_to_sph_list(X_train,orig=(x0,y0,z0))

# Normalize input data
scaler = StandardScaler()
yscaler = MinMaxScaler(feature_range=(-1, 1))
X_train_scaled = scaler.fit_transform(X_train)
y_train_scaled = yscaler.fit_transform(y_train.reshape(-1, 1))


# Create the model
model = create_model()

# Train the model
# history = model.fit(X_train_scaled, y_train_scaled, epochs=50, batch_size=16, verbose=1)
history = model.fit(X_train_scaled, y_train_scaled, epochs=1000, batch_size=250, verbose=1)

pred = yscaler.inverse_transform(model.predict(scaler.transform(np.array([X_train[123],X_train[567],X_train[1234]]))))
matc = y_train[123],y_train[567],y_train[1234]
print("some predictions:")
print(pred)
print("which should match:")
print(matc)

print("relative error for these 3:")
print(np.divide(pred.T-matc,np.array(matc)))

set_trace()

