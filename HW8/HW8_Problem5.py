#Sung Hoon Choi
#CS/CNS/EE156a HW8 Problem 5

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

def one_vs_one_label(data_array, digit1, digit2): #For Problem 5 and 6.
    labelled_data_array = []
    for i in range (0, len(data_array)):
        if data_array[i,0] == digit1:
            labelled_data_array.append(1)
        elif data_array[i,0] == digit2:
            labelled_data_array.append(-1)
    return labelled_data_array

def filter_rest_of_digits(data_array, digit1, digit2): #Used for Problem 5 and 6. (one versus one)
    filtered_data_array = []
    for i in range(0, len(data_array)):
        if data_array[i,0] == digit1:
            filtered_data_array.append(data_array[i])
        elif data_array[i,0] == digit2:
            filtered_data_array.append(data_array[i])
    return np.array(filtered_data_array)

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
label_1_vs_5_train = one_vs_one_label(train_data_array_np,1,5) #label +1 or -1
label_1_vs_5_test = one_vs_one_label(test_data_array_np,1,5)

filtered_data_1_vs_5_train = filter_rest_of_digits(train_data_array_np, 1, 5)
filtered_data_1_vs_5_test = filter_rest_of_digits(test_data_array_np, 1, 5)

print("C=0.001 ------------------------------------")
clf_digit_1_and_5 = SVC(C=0.001, kernel='poly', degree=2, coef0=1.0,gamma=1.0) #kernel definition
clf_digit_1_and_5.fit(filtered_data_1_vs_5_train[:,1:],label_1_vs_5_train) #fit the SVM
label_1_vs_5_train_predict = clf_digit_1_and_5.predict(filtered_data_1_vs_5_train[:,1:])
label_1_vs_5_test_predict = clf_digit_1_and_5.predict(filtered_data_1_vs_5_test[:,1:])
print("Number of support vectors:", len(clf_digit_1_and_5.support_)) #Number of support vectors
print("Ein: %2f" %calculate_binary_error(label_1_vs_5_train_predict, label_1_vs_5_train)) #calculate Ein
print("Eout: %2f" %calculate_binary_error(label_1_vs_5_test_predict, label_1_vs_5_test))  #calculate Eout

print("C=0.01 -------------------------------------")
clf_digit_1_and_5 = SVC(C=0.01, kernel='poly', degree=2, coef0=1.0,gamma=1.0) #kernel definition
clf_digit_1_and_5.fit(filtered_data_1_vs_5_train[:,1:],label_1_vs_5_train) #fit the SVM
label_1_vs_5_train_predict = clf_digit_1_and_5.predict(filtered_data_1_vs_5_train[:,1:])
label_1_vs_5_test_predict = clf_digit_1_and_5.predict(filtered_data_1_vs_5_test[:,1:])
print("Number of support vectors:", len(clf_digit_1_and_5.support_)) #Number of support vectors
print("Ein: %2f" %calculate_binary_error(label_1_vs_5_train_predict, label_1_vs_5_train)) #calculate Ein
print("Eout: %2f" %calculate_binary_error(label_1_vs_5_test_predict, label_1_vs_5_test))  #calculate Eout

print("C=0.1 --------------------------------------")
clf_digit_1_and_5 = SVC(C=0.1, kernel='poly', degree=2, coef0=1.0,gamma=1.0) #kernel definition
clf_digit_1_and_5.fit(filtered_data_1_vs_5_train[:,1:],label_1_vs_5_train) #fit the SVM
label_1_vs_5_train_predict = clf_digit_1_and_5.predict(filtered_data_1_vs_5_train[:,1:])
label_1_vs_5_test_predict = clf_digit_1_and_5.predict(filtered_data_1_vs_5_test[:,1:])
print("Number of support vectors:", len(clf_digit_1_and_5.support_)) #Number of support vectors
print("Ein: %2f" %calculate_binary_error(label_1_vs_5_train_predict, label_1_vs_5_train)) #calculate Ein
print("Eout: %2f" %calculate_binary_error(label_1_vs_5_test_predict, label_1_vs_5_test))  #calculate Eout

print("C=1 ----------------------------------------")
clf_digit_1_and_5 = SVC(C=1, kernel='poly', degree=2, coef0=1.0,gamma=1.0) #kernel definition
clf_digit_1_and_5.fit(filtered_data_1_vs_5_train[:,1:],label_1_vs_5_train) #fit the SVM
label_1_vs_5_train_predict = clf_digit_1_and_5.predict(filtered_data_1_vs_5_train[:,1:])
label_1_vs_5_test_predict = clf_digit_1_and_5.predict(filtered_data_1_vs_5_test[:,1:])
print("Number of support vectors:", len(clf_digit_1_and_5.support_)) #Number of support vectors
print("Ein: %2f" %calculate_binary_error(label_1_vs_5_train_predict, label_1_vs_5_train)) #calculate Ein
print("Eout: %2f" %calculate_binary_error(label_1_vs_5_test_predict, label_1_vs_5_test))  #calculate Eout

print("--------------------------------------------")