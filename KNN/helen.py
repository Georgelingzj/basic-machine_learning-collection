import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import collections

#K-近邻
class KNN:
    def __init__(self,k):
        self.K = k


    def load_based_data(self,filename):
        Data = np.loadtxt(filename)
        self.data = Data[:,0:3]
        self.label = Data[:,3:4]


        #normalize data 使用线性归一化

        max_value_data1 = np.max(self.data[:,0:1])
        min_value_data1 = np.min(self.data[:,0:1])
        self.data[:,0:1] = (self.data[:,0:1]-min_value_data1)/(max_value_data1-min_value_data1)


        max_value_data2 = np.max(self.data[:,1:2])
        min_value_data2 = np.min(self.data[:,1:2])
        self.data[:,1:2] = (self.data[:,1:2]-min_value_data2)/(max_value_data2-min_value_data2)

        max_value_data3 = np.max(self.data[:,2:3])
        min_value_data3 = np.min(self.data[:,2:3])
        self.data[:,2:3] = (self.data[:,2:3]-min_value_data3)/(max_value_data3-min_value_data3)


        '''
        #visualize data
        cols=[]
        for l in self.label:
            if l== 1:
                cols.append('red')
            elif l== 2:
                cols.append('blue')
            else:
                cols.append('green')

        color = cols


        plt.title("flight length and ratio in playing computer games")
        plt.scatter(self.data[:, 0:1],self.data[:,1:2],color = cols)
        plt.xlabel("flight length")
        plt.ylabel("game time")
        plt.show()

        plt.title("game time and iceream per week")
        plt.scatter(self.data[:, 1:2],self.data[:,2:3],color = cols)
        plt.xlabel("game time")
        plt.ylabel("iceream per week")
        plt.show()

        plt.title("icream per week and flight length")
        plt.scatter(self.data[:, 2:3],self.data[:,0:1],color = cols)
        plt.xlabel("icream per week")
        plt.ylabel("flight length")
        plt.show()
        '''


        val1 = int(input("please enter the flight length you take per year(5k-9w) "))
        val2 = float(input("please enter the ratio of time that you play games(0-15) "))
        val3 = float(input("please enter how many iceream you take per week(0.5-1.5) "))

        self.val1 = (val1-min_value_data1)/(max_value_data1-min_value_data1)
        self.val2 = (val2-min_value_data2)/(max_value_data2-min_value_data2)
        self.val3 = (val3-min_value_data3)/(max_value_data3-min_value_data3)
        #numpy array for user input
        self.arr = np.array([self.val1,self.val2,self.val3])


        #self.arr = np.array([0.21912285,0.04206632,0.45376635])


    #计算距离，取得前k个，使用欧式距离
    def measure_distance(self):

        distance = {}
        for i in range(self.label.shape[0]):
            dis_1 = np.power((self.data[i,0:1]-self.arr[0]),2)
            dis_2 = np.power((self.data[i,1:2]-self.arr[1]),2)
            dis_3 = np.power((self.data[i,2:3]-self.arr[2]),2)
            distance_compute = float(np.sqrt(dis_1 + dis_2 + dis_3))
            dict_temp = {distance_compute:int(self.label[i])}
            distance.update(dict_temp)

        #根据距离筛选出前K个
        min_k =[]
        sort_dict = collections.OrderedDict(sorted(distance.items()))
        for key,value in sort_dict.items():
            min_k.append(value)

        min_k = min_k[0:10]

        #前k个中无权重投票选出最佳
        vote_1 =vote_2 = vote_3 = 0
        for val in min_k:
            if val == 1: vote_1 += 1
            if val == 2: vote_2 += 1
            if val == 3: vote_3 += 1

        keys = ["vote1","vote2","vote3"]
        values = [vote_1,vote_2,vote_3]
        vote_result = dict(zip(keys,values))
        predict = (max(vote_result.items(),key=lambda x:x[1]))[0]

        #print result
        if predict == "vote1":
            print("不喜欢的人")
        elif predict == "vote2":
            print("魅力一般的人")
        else:
            print("极具魅力的人")


if __name__ == '__main__':
    knn = KNN(10)
    knn.load_based_data('datingTestSet2.txt')
    knn.measure_distance()

