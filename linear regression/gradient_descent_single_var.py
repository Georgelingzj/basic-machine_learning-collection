import numpy as np
from sklearn.datasets import load_boston
import matplotlib.pyplot as plt

class LinearRegression():
    def __init__(self):
        self.X,self.y = load_boston(return_X_y=True)

        #X.shape[1] = 13
        #for some reason, just choice one x to do linear regression

        self.X = self.X[:,4]
        self.X = np.transpose(self.X)
        self.X = np.reshape(self.X,newshape=(len(self.X),1))
        self.y = np.reshape(self.y,newshape=(len(self.y),1))

        ones = np.ones(self.X.shape)
        X = np.append(ones, self.X, axis=1)
        self.X = X

        self.m = X.shape[0]
        self.n = X.shape[1]
        self.theta = np.random.randn(X.shape[1])

    def test_train_split(self,test_size = 0.2):
        data =  np.concatenate((self.X,self.y),axis=1)

        np.random.shuffle(data)

        test = data[:int(test_size*len(data)),]
        train = data[int(test_size*len(data)): ,]

        self.train_X = train[:,0:2]
        self.train_y = train[:, 2:3]
        self.test_X = test[:,0:2]
        self.test_y = test[:,2:3]

    """
    def L1_loss(self,y_pred,y):
        return np.abs(y_pred-y)
    """

    def L2_loss(self,y_pred,y):
        return np.power((y_pred-y),2)


    def gradient_descent(self,learning_rate = 0.01,epochs = 1000,perform = False):
        #loss on training set
        cost = 0
        J_theta0 = 0
        J_theta1 = 0
        for epoch in range(epochs):
            for i in range(len(self.train_y)):
                y_pred = np.matmul(self.train_X[i],self.theta)
                cost += self.L2_loss(y_pred,self.train_y[i])

                #theta0
                J_theta0 += (y_pred-self.train_y[i])
                #theta1
                J_theta1 += (y_pred-self.train_y[i])*self.train_X[i,1]
            cost = (1/(2*self.m))*cost
            #intercept
            self.theta[0] -= learning_rate*(1/self.m)*J_theta0

            #coef
            self.theta[1] -= learning_rate*(1/self.m)*J_theta1

            if (epoch + 1) % 50 == 0:
                print("No.{},cost is {}".format(epoch + 1, cost))

            if perform == True:
                #see exactly gradient descent
                if (epoch+1)%100 == 0:
                    plt.scatter(self.test_X[:, 1], self.test_y, color='green')

                    x = np.linspace(0.3, 1, 1000)
                    y = self.theta[1] * x + self.theta[0]
                    plt.plot(x, y, color='red')
                    plt.show()


    def predict(self):
        #MSE
        loss = 0
        for i in range(len(self.test_y)):
            y_pred = np.matmul(self.test_X[i],self.theta)
            loss += np.power((y_pred-self.test_y[i]),2)


        print("The loss on test set is {}".format(loss))


        #draw result
        plt.scatter(self.test_X[:,1],self.test_y,color = 'green')

        x = np.linspace(0.3,1,1000)
        y = self.theta[1]*x + self.theta[0]
        plt.plot(x,y,color = 'red')
        plt.show()



if __name__ == '__main__':
    linear = LinearRegression()
    linear.test_train_split()
    linear.gradient_descent(learning_rate=0.01,epochs=1000,perform=True)
    linear.predict()
