#Sung Hoon Choi
#CS/CNS/EE156a HW8 Problem 2

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
label_2_vs_all = one_vs_all_label(train_data_array_np,2)
label_4_vs_all = one_vs_all_label(train_data_array_np,4)
label_6_vs_all = one_vs_all_label(train_data_array_np,6)
label_8_vs_all = one_vs_all_label(train_data_array_np,8)

clf_digit_0 = SVC(C=0.01, kernel='poly', degree=2, coef0=1.0,gamma=1.0) #kernel definition
clf_digit_2 = SVC(C=0.01, kernel='poly', degree=2, coef0=1.0,gamma=1.0)
clf_digit_4 = SVC(C=0.01, kernel='poly', degree=2, coef0=1.0,gamma=1.0)
clf_digit_6 = SVC(C=0.01, kernel='poly', degree=2, coef0=1.0,gamma=1.0)
clf_digit_8 = SVC(C=0.01, kernel='poly', degree=2, coef0=1.0,gamma=1.0)

clf_digit_0.fit(train_data_array_np[:,1:],label_0_vs_all) #fit the SVM
clf_digit_2.fit(train_data_array_np[:,1:],label_2_vs_all)
clf_digit_4.fit(train_data_array_np[:,1:],label_4_vs_all)
clf_digit_6.fit(train_data_array_np[:,1:],label_6_vs_all)
clf_digit_8.fit(train_data_array_np[:,1:],label_8_vs_all)

label_0_predict = clf_digit_0.predict(train_data_array_np[:,1:]) #predict the output using SVM
label_2_predict = clf_digit_2.predict(train_data_array_np[:,1:])
label_4_predict = clf_digit_4.predict(train_data_array_np[:,1:])
label_6_predict = clf_digit_6.predict(train_data_array_np[:,1:])
label_8_predict = clf_digit_8.predict(train_data_array_np[:,1:])

print("0 versus all: %2f\n" %calculate_binary_error(label_0_predict, label_0_vs_all)) #calculate Ein
print("2 versus all: %2f\n" %calculate_binary_error(label_2_predict, label_2_vs_all))
print("4 versus all: %2f\n" %calculate_binary_error(label_4_predict, label_4_vs_all))
print("6 versus all: %2f\n" %calculate_binary_error(label_6_predict, label_6_vs_all))
print("8 versus all: %2f\n" %calculate_binary_error(label_8_predict, label_8_vs_all))
