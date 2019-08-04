#Sung Hoon Choi
#CS/CNS/EE156a HW8 Problem 8

import numpy as np
from sklearn.svm import SVC   #Used for implementing SVM.

def extract_data(filename):
    data_array = []
    for line in open(filename):
        data_entries = line.split('  ')
        data_row = [float(data_entries[1]),float(data_entries[2]),float(data_entries[3].rstrip("\n"))]
        data_array.append(data_row)
    return data_array

def one_vs_one_label(data_array, digit1, digit2): #For Problem 5,6,7,8.
    labelled_data_array = []
    for i in range (0, len(data_array)):
        if data_array[i,0] == digit1:
            labelled_data_array.append(1)
        elif data_array[i,0] == digit2:
            labelled_data_array.append(-1)
    return labelled_data_array

def filter_rest_of_digits(data_array, digit1, digit2): #Used for Problem 5,6,7,8.
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

def divide_into_partitions(data, folds):
    partition_size = len(data)//folds
    partitions = []
    for i in range(0, folds):
        partitions.append(data[i*partition_size:(i+1)*partition_size])
    return partitions                   #partition[0]: first partition
                                        #partition[1]: second partition
                                        # . . .

train_data_array = extract_data("features.train.txt")  #extract data
test_data_array = extract_data("features.test.txt")
train_data_array_np = np.array(train_data_array)
test_data_array_np = np.array(test_data_array)
label_1_vs_5_train = one_vs_one_label(train_data_array_np,1,5) #label +1 or -1
label_1_vs_5_test = one_vs_one_label(test_data_array_np,1,5)

filtered_data_1_vs_5_train = filter_rest_of_digits(train_data_array_np, 1, 5)
filtered_data_1_vs_5_test = filter_rest_of_digits(test_data_array_np, 1, 5)

Total_Run_Error = 0
for run in range (0,100):
    np.random.shuffle((filtered_data_1_vs_5_train))
    partitions = np.array(divide_into_partitions(filtered_data_1_vs_5_train,10))
    partitions_labels = np.array(divide_into_partitions(one_vs_one_label(filtered_data_1_vs_5_train,1,5),10))

    clf_digit_1_and_5 = SVC(C=0.001, kernel='poly', degree=2, coef0=1.0, gamma=1.0)  # kernel definition
    Each_CV_Error = 0
    for i in range (0,len(partitions)):
        cv_training_partitions = np.delete(partitions,i,0) #Leave on partition for validation.
        cv_training_labels = np.delete(partitions_labels,i,0)
        concat_cv_training_partitions = cv_training_partitions[0]
        concat_cv_training_labels = cv_training_labels[0]
        for j in range (1, len(cv_training_partitions)): #Need to reshape the partitions for processing.
            concat_cv_training_partitions = np.concatenate((concat_cv_training_partitions,cv_training_partitions[j]))
            concat_cv_training_labels = np.concatenate((concat_cv_training_labels,cv_training_labels[j]))
        clf_digit_1_and_5.fit(concat_cv_training_partitions[:,1:],concat_cv_training_labels) #Train
        predict = clf_digit_1_and_5.predict(partitions[i][:,1:]) #Validate
        Each_CV_Error = Each_CV_Error + calculate_binary_error(predict, partitions_labels[i]) #Get the error
    Each_Run_Error = Each_CV_Error/len(partitions)
    Total_Run_Error = Total_Run_Error + Each_Run_Error

print("Average Ecv for C=0.001:", Total_Run_Error/100) #Problem 8

