import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense


N = 100
t = np.arange(0, N)
y = np.sin(0.05 * t * 10) + 0.8 * np.random.rand(N)  


split_ratio = 0.7  
n_train = int(N * split_ratio)  

train = y[:n_train]  
test = y[n_train:]   

def convertToMatrix(data, step=1):
    X, Y = [], []
    for i in range(len(data) - step):
        X.append(data[i:i + step])
        Y.append(data[i + step])
    return np.array(X), np.array(Y)

step = 5
x_train, y_train = convertToMatrix(train, step)
x_test, y_test = convertToMatrix(test, step)

x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))

model = Sequential()
model.add(SimpleRNN(units=32, input_shape=(step, 1), activation="relu"))
model.add(Dense(units=1))
model.compile(optimizer="adam", loss="mse", metrics=["accuracy"])

model.fit(x_train, y_train, epochs=30, batch_size=4, verbose=1)

y_pred = []
current_input = y[:step].reshape(1, step, 1) 
for _ in range(N - step):
    next_pred = model.predict(current_input)
    next_pred = np.reshape(next_pred, (1, 1, 1))  
    y_pred.append(next_pred[0, 0, 0]) 
    current_input = np.append(current_input[:, 1:, :], next_pred, axis=1)

y_pred = np.concatenate((y[:step], y_pred))

def plot_comparison(original, predict, n_train):
    plt.figure(figsize=(10, 5))
    plt.plot(original, label="Original", color="blue")
    plt.plot(predict, label="Predict", color="red", linestyle="--")
    plt.axvline(x=n_train, color="purple", linestyle="-", linewidth=2, label="Train/Test Split")
    plt.legend()
    plt.title("Comparison of Original and Predicted Values")
    plt.show()

plot_comparison(y, y_pred, n_train)
