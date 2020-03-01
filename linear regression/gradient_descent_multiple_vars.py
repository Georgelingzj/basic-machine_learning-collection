"""
simple multiple variables linear regression implement in python

Author: george ling
"""
import numpy as np
from sklearn.datasets import load_boston


class LinearRegression():
    def __init__(self):
        self.X,self.y = load_boston(return_X_y=True)

        ones = np.ones(shape=(self.X.shape[0],1))
        X = np.append(ones, self.X, axis=1)
        self.X = X
        self.y = np.reshape(self.y,newshape=(self.y.shape[0],1))

        self.m = X.shape[0] #row
        self.n = X.shape[1] #col
        #randomly initialize the theta
        self.theta = np.random.randn(X.shape[1])

    def normalize_data(self):
        #data = (data-average)/(max-min)
        #deal with x data
        for i in range(1,self.n):
            max = np.max(self.X[:,i])
            min = np.min(self.X[:,i])
            average = np.average(self.X[:,i])
            for j in range(self.m):
                self.X[j,i] = (float)(self.X[j,i]-average)/(float)(max-min)
        #deal with y data
        max_y = np.max(self.y)
        min_y = np.min(self.y)
        average = np.average(self.y)
        for k in range(self.m):
            self.y[k] = (float)(self.y[k]-average)/(float)(max_y-min_y)

    def test_train_split(self,test_size = 0.2):
        #combine x value and y value into whole dateset
        data =  np.hstack((self.X,self.y))


        np.random.shuffle(data)

        test = data[:int(test_size*len(data)),]
        train = data[int(test_size*len(data)): ,]

        self.train_X = train[:,0:14]
        self.train_y = train[:, 14:15]
        self.test_X = test[:,0:14]
        self.test_y = test[:,14:15]



    def L2_loss(self,y_pred,y):
        return np.power((y_pred-y),2)


    def gradient_descent(self,learning_rate = 0.01,epochs = 1000):
        #loss on training set
        cost = 0
        #row vector to col vector
        self.theta = np.reshape(self.theta,newshape=(len(self.theta),1))

        theta_derivative = [0 for i in range(len(self.theta))]

        for epoch in range(epochs):
            for i in range(len(self.train_y)):
                y_pred = np.matmul(self.train_X[i],self.theta)
                cost += self.L2_loss(y_pred,self.train_y[i])

                for k in range(len(theta_derivative)):
                    theta_derivative[k] += (y_pred-self.train_y[i])*self.train_X[i,k]

            cost = (1/(2*self.m))*cost
            #refresh thetas
            for j in range(self.theta.shape[0]):
                self.theta[j] -= learning_rate*(1/self.m)*theta_derivative[j]

            if (epoch + 1) % 50 == 0:
                print("No.{},cost is {}".format(epoch + 1, cost))




    def predict(self):
        #MSE
        loss = 0
        for i in range(len(self.test_y)):
            y_pred = np.matmul(self.test_X[i],self.theta)
            loss += np.power((y_pred-self.test_y[i]),2)


        print("The loss on test set is {}".format(loss))



if __name__ == '__main__':
    linear = LinearRegression()
    linear.normalize_data()
    linear.test_train_split()
    linear.gradient_descent(learning_rate=0.01,epochs=1000)
    linear.predict()
