import tensorflow as tf
import numpy as np
from sklearn.datasets import load_boston

class LRregression():
    def __init__(self):
        self.X,self.y = load_boston(return_X_y=True)
        '''X.shape = (506, 13)
           y.shape = (506,)
        '''
        self.X = self.X[:,0]

    def train(self,learning_rate = 0.01,epochs = 10000,batch_size = 50):
        #define placeholder
        x = tf.placeholder(dtype=tf.float32,shape=[None,1])
        y = tf.placeholder(dtype=tf.float32,shape=[None,1])

        #give random coef and intercept
        k = tf.Variable(tf.random_normal([1,1],stddev=0.1),name='coef')
        b = tf.Variable(tf.random_normal([1,1],stddev=0.1),name = 'intercept')



        y_pred = tf.add(tf.matmul(x,k),b)

        cost = tf.reduce_mean(tf.pow(y_pred-y, 2))

        train = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

        with tf.Session() as sess:
            tf.global_variables_initializer().run()

            for i in range(epochs):
                rand = np.random.choice(len(self.X),batch_size)

                rand_x = np.reshape(self.X[rand],(50,1))
                rand_y = np.reshape(self.y[rand],(50,1))


                sess.run(train,feed_dict={x:rand_x,y:rand_y})

                if (i+1) %50 == 0:
                    cost_ = sess.run(cost,feed_dict={x:rand_x,y:rand_y})

                    print("NO.{},cost is {}".format(i+1,cost_))


if __name__ == '__main__':
    reg = LRregression()
    reg.train()






