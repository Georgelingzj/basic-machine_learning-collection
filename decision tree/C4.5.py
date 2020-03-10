"""
C4.5 decision tree

degree == 2

"""

from sklearn.datasets import load_iris
import numpy as np
import math


def load_data():
    X,y = load_iris(return_X_y=True)

    y = np.reshape(y, newshape=(len(y), 1))
    return X,y

def train_test_split(test_size = 0.2):
    X,y = load_data()
    data = np.hstack((X,y))

    np.random.shuffle(data)
    num_sample = data.shape[0]
    num_chara = data.shape[1]-1

    test_X = data[:int(num_sample*test_size),0:num_chara]
    train_X = data[int(num_sample*test_size):, 0:num_chara]
    test_y = data[:int(num_sample*test_size),num_chara:num_chara+1]
    train_y = data[int(num_sample*test_size):,num_chara:num_chara+1]

    return train_X, train_y, test_X, test_y

def entropy(X,y):
    """
    :param X: feature
    :param y: label
    :return: the entropy-- H(C)
    """

    num_label = np.unique(y)
    p_xi = []

    for i in range(len(num_label)):
        num = 0
        for j in range(len(y)):
            if y[j] == num_label[i]:
               num += 1
        p_xi.append(float(num/len(y)))

    entropy_ = 0

    for k in range(len(p_xi)):
        entropy_ += p_xi[k]* math.log2(p_xi[k])

    return -1.0*entropy_

# transform numpy array to dic
# easy to compute Gain with continuous value
def sortedArr_to_Dic(list):
    length = len(list)
    dic_ = {}
    num = 1
    key_ = list[0]
    for i in range(1,length):
        if list[i] == key_:
            num += 1
        else:
            if i == length-1:
                dic_.update({key_: num})
                num = 1
                key_ = list[i]

                dic_.update({key_: num})
            else:
                dic_.update({key_: num})
                num = 1
                key_ = list[i]

    return dic_


def split_(value, list_):
    """

    :param value: decision value that split the data
    :param list_: data with its label  shape = (None,2)
    :return: array below decision value and array above decision value
    """

    for i in range(len(list_)):
        if list_[i,0] == value:
            return list_[:i,], list_[i:,]



def Entropy_inside(value,list_):
    """

    :param value: decison value that split the data
    :param list_: data with its label  shape = (None,2)
    :return: entropy below decision value and entropy above the decision value
    """

    #split data with decision value

    list_below, list_above = split_(value,list_)

    # 1 is useless
    # but for future purpose
    Entropy_below = entropy(1, list_below[:,1])
    Entropy_above = entropy(1, list_above[:,1])

    return Entropy_below, Entropy_above



def GainRadio(X,y):
    """

    :param X: feature
    :param y: label
    :return: best split label
    """
    Hc = entropy(X, y)

    num_feature = X.shape[1]
    num_sample = X.shape[0]

    # combine feature and label
    data_whole = np.hstack((X,y))

    dic_max_feature = {}

    for i in range(num_feature):

        data_withLabel = np.row_stack((data_whole[:,i],data_whole[:,num_feature]))
        data_withLabel = np.transpose(data_withLabel)
        data_withLabel.sort(axis=0)



        split_ = 0
        data = X[:, i]
        data.sort()
        # numpy list to dic
        weight_dic = sortedArr_to_Dic(data)
        # delete duplicate number
        data = np.unique(data)


        for key in weight_dic:
            value = weight_dic[key]
            split_ = float((value/num_sample))*math.log2(float((value/num_sample)))

        split_ = -1.0* split_

        Gain_inside = {}

        for j in range(1,len(data)):


            D_below = data[:j]
            D_above = data[j:]
            decision_value = D_above[0]

            Entropy_below, Entropy_above = Entropy_inside(decision_value,data_withLabel)

            # add dic
            below_ = (float((len(D_below)/num_sample)))*Entropy_below
            above_ = (float((len(D_above)/num_sample)))*Entropy_above

            Gain_in = Hc - (below_+above_)

            Gain_inside.update({Gain_in:decision_value})


        Gain_inside = sorted(Gain_inside.items(),key=lambda x:x[0],reverse=True)

        max_entro = Gain_inside[0][0]

        max_entro_radio = max_entro/split_

        max_value = Gain_inside[0][1]

        dic_max_feature.update({max_entro_radio:max_value})

    #print(dic_max_feature)
    """
    get dic with key is Gainradio
    value is split value
    """

    feature_index = 0
    dic_ = {}

    for key,value in dic_max_feature.items():
        inside = {feature_index:value}
        dic_.update({key:inside})
        feature_index += 1

    #print(dic_)

    dic_ = sorted(dic_.items(),key=lambda x:x[0],reverse=True)

    max_radio_feature = dic_[0]

    return max_radio_feature, data_whole

#
# def build_tree(data1, data2, decision_value):


def split_dataset(dic_, data_whole):


    dic_ = dic_[1]
    dic_ = dic_.popitem()

    feature_index = dic_[0]
    feature_value = dic_[1]

    # sort data by col == feature_index
    data = data_whole[np.argsort(data_whole[:, feature_index])]


    # print(data)

    # split whole data set
    for i in range(data.shape[0]):
        if data[i, feature_index] == feature_value:
            data1 = data[:i, ]
            data2 = data[i:, ]
            break

    data1 = np.delete(data1,1,axis=feature_index)
    data2 = np.delete(data2, 1, axis=feature_index)

    return data1, data2, feature_value





def run():
    train_X, train_y,_ ,_ = train_test_split()

    max_feature, data_whole = GainRadio(train_X,train_y)

    data1, data2, s = split_dataset(max_feature,data_whole)
    print(data1)

    print(data2)

    print(s)



if __name__ == '__main__':
    run()