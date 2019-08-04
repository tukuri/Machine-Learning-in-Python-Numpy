#Sung Hoon Choi
#CS/CNS/EE156a HW8 Problem 4

import numpy as np
from sklearn.svm import SVC   #Used for implementing SVM.

def extract_data(filename):
    data_array = []
    for line in open(filename):
        data=[]
        data_entries = line.split('  ')
        data_row = [float(data_entries[1]),float(data_entries[2]),float(data_entries[3].rstrip("\n"))]
        data_array.append(data_row)
    return data_array

def one_vs_all_label(data_array, digit):
    labelled_data_array = []
    for i in range (0, len(data_array)):
        if data_array[i,0] == digit:
            labelled_data_array.append(1)
        else:
            labelled_data_array.append(-1)
    return labelled_data_array

def calculate_binary_error(g_x, f_x):
    error_count = 0
    for i in range (0,len(g_x)):
        if(g_x[i] != f_x[i]):
            error_count = error_count + 1
    return error_count/len(g_x)

train_data_array = extract_data("features.train.txt")  #extract data
test_data_array = extract_data("features.test.txt")
train_data_array_np = np.array(train_data_array)
test_data_array_np = np.array(test_data_array)

label_0_vs_all = one_vs_all_label(train_data_array_np,0) #label +1 or -1
label_1_vs_all = one_vs_all_label(train_data_array_np,1)

clf_digit_0 = SVC(C=0.01, kernel='poly', degree=2, coef0=1.0,gamma=1.0) #kernel definition
clf_digit_1 = SVC(C=0.01, kernel='poly', degree=2, coef0=1.0,gamma=1.0)

clf_digit_0.fit(train_data_array_np[:,1:],label_0_vs_all) #fit the SVM
clf_digit_1.fit(train_data_array_np[:,1:],label_1_vs_all)

print("Number of support vectors: label 0: ", len(clf_digit_0.support_)) #Number of support vectors
print("Number of support vectors: label 1: ", len(clf_digit_1.support_))
print("Difference: ", len(clf_digit_0.support_)-len(clf_digit_1.support_))