import sys
import numpy as np
import pandas as pd
import time

class LinearRegression():
    
    def __call__(self, inputs):
        m = inputs.shape[0]
        inputs_ = np.append(np.ones((m,1)), inputs, axis=1)
        H = np.dot(inputs_, self.weights)
        return H

    def predict(self, inputs):
        H = self(inputs)
        return H
       
    def costFunc(self, X, Y):
        m = X.shape[0]
        H = self(X)
        J = (1/(2*m))*np.sum((H - Y)*(H - Y))
        return J
    
    def gradFunc(self, X, Y):
        m = X.shape[0]
        X_ = np.append(np.ones((m,1)), X, axis=1)
        H = self(X)
        grad = (1/m)*np.dot(X_.T,H - Y)
        return grad
                      
    def fit(self, X_train, Y_train, epochs, learning_rate):
        self.weights = np.random.normal(loc=0.0, scale=0.05, size=(X_train.shape[1]+1, Y_train.shape[1]))
        print('Training...')
        start_time = time.time()
        for _ in range(epochs):
            grad = self.gradFunc(X_train, Y_train)
            self.weights = self.weights - learning_rate*grad
        loss = self.costFunc(X_train, Y_train)
        print('Loss: {}'.format(loss))
        now = time.time()
        duration = now - start_time
        print('----- {}s -----'.format(np.round(1000*duration)/1000))

def main():
    input_csv = sys.argv[1]
    output_csv = sys.argv[2]
    data = pd.read_table(input_csv, sep=',', header=None, dtype='float32')
    X_train = data.iloc[:,[0,1]].values
    Y_train = data.iloc[:,[2]].values
    X_train_ = (X_train - np.mean(X_train, axis=0))/np.std(X_train, axis=0)
    learning_rates = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 0.75]
    epochs = [100, 100, 100, 100, 100, 100, 100, 100, 100, 50]
    with open(output_csv,'w') as f:
        for i in range(len(learning_rates)):
            linear_regression = LinearRegression()
            linear_regression.fit(X_train_, Y_train, epochs=epochs[i], learning_rate=learning_rates[i])
            f.write(str(learning_rates[i]))
            f.write(',')
            f.write(str(epochs[i]))
            f.write(',')
            f.write(str(linear_regression.weights[0,0]))
            f.write(',')
            f.write(str(linear_regression.weights[1,0]))
            f.write(',')
            f.write(str(linear_regression.weights[2,0]))
            f.write('\n')

if __name__ == '__main__':
    main()

