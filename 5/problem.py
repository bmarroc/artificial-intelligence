import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt

class NeuralNetwork():
    
    def __init__(self, hidden_units):
        self.hidden_units = hidden_units
    
    def __call__(self, inputs):
        m = inputs.shape[0]
        inputs_ = np.append(np.ones((m,1)), inputs, axis=1)
        A = 1.0/(1.0+np.exp(-np.dot(inputs_, self.Theta1.T)))
        A_ = np.append(np.ones((m,1)), A, axis=1)
        H = 1.0/(1.0+np.exp(-np.dot(A_, self.Theta2.T)))
        return H,A

    def predict(self, inputs):
        H,_ = self(inputs)
        return H
           
    def costFunc(self, X, Y):
        m = X.shape[0]
        H,_ = self(X)
        J = -(1/m)*np.sum(Y*np.log(H)+(1-Y)*np.log(1-H))
        return J
    
    def gradFunc(self, X, Y):
        Theta1 = self.weights[:(self.input_dim+1)*self.hidden_units].reshape((self.hidden_units, self.input_dim+1))
        Theta2 = self.weights[(self.input_dim+1)*self.hidden_units:].reshape((self.output_dim, self.hidden_units+1))
        m = X.shape[0]
        X_ = np.append(np.ones((m,1)), X, axis=1)
        H,A = self(X)
        A_ = np.append(np.ones((m,1)), A, axis=1)
        D2 = H-Y
        D1 = np.dot(D2, Theta2[:,1:])*(A*(1-A))
        L1 = np.zeros(shape=Theta1.shape)
        L2 = np.zeros(shape=Theta2.shape)
        for t in range(m):
            L2 = L2 + np.dot(D2[[t],:].T,A_[[t],:])
            L1 = L1 + np.dot(D1[[t],:].T,X_[[t],:])
        Theta1_grad = (1/m)*L1
        Theta2_grad = (1/m)*L2
        grad = np.concatenate((Theta1_grad.flatten(), Theta2_grad.flatten()))
        return grad
    
    def metricFunc(self, X, Y):
        H,_ = self(X)
        P = (H>0.5).astype(float)
        return np.mean(P == Y).astype(float)
    
    def fit(self, X_train, Y_train, epochs=1000, learning_rate=0.5):
        self.input_dim = X_train.shape[1]
        self.output_dim = Y_train.shape[1]
        self.Theta1 = np.random.normal(loc=0.0, scale=0.05, size=(self.hidden_units, self.input_dim+1))
        self.Theta2 = np.random.normal(loc=0.0, scale=0.05, size=(self.output_dim, self.hidden_units+1))
        self.weights = np.concatenate((self.Theta1.flatten(), self.Theta2.flatten()))
        print('Training...')
        start_time = time.time()
        for epoch in range(epochs):
            grad = self.gradFunc(X_train, Y_train)
            self.weights = self.weights - learning_rate*grad
            self.Theta1 = self.weights[:(self.input_dim+1)*self.hidden_units].reshape((self.hidden_units, self.input_dim+1))
            self.Theta2 = self.weights[(self.input_dim+1)*self.hidden_units:].reshape((self.output_dim, self.hidden_units+1))
        loss = self.costFunc(X_train, Y_train)
        metric = self.metricFunc(X_train, Y_train)
        print('Loss: {} - Metric: {}'.format(loss,metric))
        now = time.time()
        duration = now - start_time
        print('----- {}s -----'.format(np.round(1000*duration)/1000))

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
        w = (self.predict(z)[:,0]>0.5).astype(int)
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
    data = pd.read_csv('input.csv', sep=',', index_col=0, dtype='float32')
    X_train = data.iloc[:320,[0,1]].values
    Y_train = data.iloc[:320,[2]].values
    X_test = data.iloc[320:,[0,1]].values
    Y_test = data.iloc[320:,[2]].values
    hidden_units = [4,8,16,32,64,128,256,512]
    with open('output.csv','w') as f:
        for h in hidden_units:
            neural_network = NeuralNetwork(hidden_units=h)
            neural_network.fit(X_train, Y_train)
            neural_network.plot_model(X_train, Y_train)
            train_accuracy = neural_network.metricFunc(X_train,Y_train)
            test_accuracy = neural_network.metricFunc(X_test,Y_test)
            f.write(str(h))
            f.write(',')
            f.write(str(train_accuracy))
            f.write(',')
            f.write(str(test_accuracy))
            f.write('\n')
            
if __name__ == '__main__':
    main()