import sys
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt

class LogisticRegression():

    def __call__(self, inputs):
        m = inputs.shape[0]
        inputs_ = np.append(np.ones((m,1)), inputs, axis=1)
        H = 1.0/(1.0+np.exp(-np.dot(inputs_, self.weights)))
        return H

    def predict_proba(self, inputs):
        H = self(inputs)
        return H
    
    def predict(self, inputs):
        H = self(inputs)
        return (H>0.5).astype(float)
       
    def costFunc(self, X, Y):
        m = X.shape[0]
        H = self(X)
        J = -(1/m)*np.sum(Y*np.log(H)+(1-Y)*np.log(1-H))
        return J
    
    def gradFunc(self, X, Y):
        m = X.shape[0]
        X_ = np.append(np.ones((m,1)), X, axis=1)
        H = self(X)
        grad = (1/m)*np.dot(X_.T,H-Y)
        return grad
    
    def metricFunc(self, X, Y):
        H = self(X)
        P = (H>0.5).astype(float)
        return np.mean(P == Y).astype(float)
                      
    def fit(self, X_train, Y_train, epochs=1000, learning_rate=0.001):
        self.weights = np.random.normal(loc=0.0, scale=0.05, size=(X_train.shape[1]+1, Y_train.shape[1]))
        self.weights_history = []
        z = np.random.normal(loc=0.0, scale=0.05, size=(X_train.shape[1]+1, Y_train.shape[1]))
        print('Training...')
        for epoch in range(epochs):
            start_time = time.time()
            grad = self.gradFunc(X_train, Y_train)
            z = 0.99*z + grad
            self.weights = self.weights - learning_rate*z
            loss = self.costFunc(X_train, Y_train)
            metric = self.metricFunc(X_train, Y_train)
            now = time.time()
            duration = now - start_time
            print('Epochs {}/{} - Loss: {} - Metric: {}'.format(epoch+1, epochs, loss, metric))
            print('----- {}s -----'.format(np.round(1000*duration)/1000))
            self.weights_history.append(self.weights)

    def plot_model(self, X_train, Y_train):
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(1, 1, 1)
        ax.axis([np.min(X_train[:,0])-1., np.max(X_train[:,0])+1., np.min(X_train[:,1])-1., np.max(X_train[:,1])+1.])
        u = np.linspace(np.min(X_train[:,0])-1., np.max(X_train[:,0])+1., 100)
        v = np.linspace(np.min(X_train[:,1])-1., np.max(X_train[:,1])+1., 100)
        z = []
        for i in range(100):
            for j in range(100):
                z.append([u[i], v[j]])
        z = np.array(z)
        w = self.predict(z)
        model_pos = np.where(w == 1)[0]
        model_neg = np.where(w == 0)[0]
        ax.plot(z[model_pos,0], z[model_pos,1], marker='+', color='#C7C3FF', markersize=10, linestyle='')
        ax.plot(z[model_neg,0], z[model_neg,1], marker='+', color='#FFECFB', markersize=10, linestyle='')
        pos = np.where(Y_train[:,0] == 1)[0]
        neg = np.where(Y_train[:,0] == 0)[0]
        ax.plot(X_train[pos,0], X_train[pos,1], marker='.', color='#0F00FF', markersize=10, linestyle='') 
        ax.plot(X_train[neg,0], X_train[neg,1], marker='.', color='#FF00AE', markersize=10, linestyle='')
        plt.show()

def main():
    input_csv = sys.argv[1]
    output_csv = sys.argv[2]
    data = pd.read_table(input_csv, sep=',', header=None, dtype='float32')
    X_train = data.iloc[:,[0,1]].values
    Y_train = data.iloc[:,[2]].values
    logistic_regression = LogisticRegression()
    logistic_regression.fit(X_train, Y_train)
    logistic_regression.plot_model(X_train, Y_train)
    with open(output_csv,'w') as f: 
        for w in logistic_regression.weights_history:
            f.write(str(w[1,0]))
            f.write(',')
            f.write(str(w[2,0]))
            f.write(',')
            f.write(str(w[0,0]))
            f.write('\n')
            
if __name__ == '__main__':
    main()