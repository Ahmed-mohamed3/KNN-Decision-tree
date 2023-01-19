from collections import Counter
import pandas as pd
import numpy as np


# def normalization (df,data):
#     normalized_df=(df-(data.iloc[:,:-1]).mean())/(data.iloc[:,:-1]).std()
#     return normalized_df

def normalization (df):
    normalized_df=(df-(read_train_data.iloc[:,:-1]).mean())/(read_train_data.iloc[:,:-1]).std()
    return normalized_df


read_train_data = pd.read_csv (r'http://vlm1.uta.edu/~athitsos/courses/cse6363_spring2017/assignments/uci_datasets/pendigits_training.txt',delim_whitespace=True, header= None)
# read_train_data.to_csv (r'F:\sana 4\Machine learning\Assignment\Assignment 1\KNN\train_data.csv', index=None)

read_test_data = pd.read_csv (r'http://vlm1.uta.edu/~athitsos/courses/cse6363_spring2017/assignments/uci_datasets/pendigits_test.txt', delim_whitespace=True,header= None)
# read_test_data.to_csv (r'F:\sana 4\Machine learning\Assignment\Assignment 1\KNN\test_data.csv', index=None)

X_train = np.array(normalization (read_train_data.iloc[:,:-1]) )
y_train = np.array(read_train_data.iloc[:,-1 ])
X_test = np.array(normalization (read_test_data.iloc[:,:-1]))
y_test = np.array(read_test_data.iloc[:,-1 ])

k = 0
def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

# Bubble sort function to sort lit of lists according to the second element(index)
def sorts(sub_li):
    l = len(sub_li)
    for i in range(0, l):
        for j in range(0, l-i-1):
            if (sub_li[j][0] > sub_li[j + 1][0]):
                tempo = sub_li[j]
                sub_li[j]= sub_li[j + 1]
                sub_li[j + 1]= tempo
    return sub_li


def sort_by_ind(seq):
    # return [i for (v, i) in sorted((v, i) for (i, v) in enumerate(seq))]
    
    # create list of lists [[number , index],[number, index],........]
    li=[]
    for i in range(len(s)):
        li.append([s[i],i])
    li = sorts(li)

    # return list of indecies only 
    sort_index = []
    for x in li:
        sort_index.append(x[1])
    return sort_index

# create a dictionary includes the labels as a key and num.of freq as a value  {label_1 : 3, label_2: 1, ......}
def num_of_occurrences (k_neighbor_labels):
    counter = {}
    for label in k_neighbor_labels:
        if label not in counter:
            counter[label] = 0
        counter[label] += 1
    return counter

# get the most frequency label 
def mode (counter):
    k = 0
    v = 0
    for key,value in counter.items():
        if value > v:
            k = key
            v = value
    return k

# calculate the accuracy 
def accuracy(y_actual, y_pred):
    accuracy = np.sum(y_actual == y_pred) / len(y_actual)
    return accuracy
        


def model(X):
        # call predict function to  all the samples in test sample
        y_pred = [predict(x) for x in X]
        return np.array(y_pred)

def predict(x):
        #get the nearest neighbors
        # calculate the distance between x sample and all x_train samples
        distance = []
        for row in X_train:
            distance.append(euclidean_distance(x, row))
        # argsort: return array contains sorted distances by index
        # we use these indecies to search by them in y_label
        knn_indecies = sort_by_ind(distance)
        knn_indecies = knn_indecies[:k] 

        # get the labels of the K nearest neighbors
        knn_labels = []
        for index in knn_indecies:
            knn_labels.append(y_train[index])

        # do a majority role and choose the most common class label
        most_common = num_of_occurrences(knn_labels)
        most_common = mode(most_common)

        return most_common

f = open("output.txt", "a")
for i in range (1, 10):
    k = i 
    predictions = model(X_test)
    f.write(f"\n---------------------------{i}NN----------------------------\n")
    f.write(f"Accuracy for {k}_nn is:{accuracy(y_test, predictions)}\n")
    f.write(f"the number of correctly classified test instances: {np.sum(y_test == predictions)}\n" )
    f.write(f"the total number of instances in the test set: {len(y_test)}\n")
    # column_stack: convert row vector to column vector 
    # hstake: combine column vectors horizontally  
    mat = np.hstack((np.column_stack([y_test]),np.column_stack([predictions])))
    for line in mat :
        f.write(str(line[0]))
        f.write("  -  ")
        f.write(str(line[1]))
        f.write('\n')
    print (i)