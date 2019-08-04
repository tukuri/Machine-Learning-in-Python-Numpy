#Sung Hoon Choi
#CS/CNS/EE156a HW8 Problem 7

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

Choice_a = 0  #Number of runs for the case when choice [a] is chosen.
Choice_b = 0  #Number of runs for the case when choice [b] is chosen.
Choice_c = 0  #Number of runs for the case when choice [c] is chosen.
Choice_d = 0  #Number of runs for the case when choice [d] is chosen.
Choice_e = 0  #Number of runs for the case when choice [e] is chosen.
for run in range (0,100):
    Total_Run_Error = 0
    np.random.shuffle((filtered_data_1_vs_5_train))
    partitions = np.array(divide_into_partitions(filtered_data_1_vs_5_train,10))
    partitions_labels = np.array(divide_into_partitions(one_vs_one_label(filtered_data_1_vs_5_train,1,5),10))
    Error = [0, 0, 0, 0, 0]
    for C_value in (0.0001, 0.001, 0.01, 0.1, 1):
        clf_digit_1_and_5 = SVC(C=C_value, kernel='poly', degree=2, coef0=1.0, gamma=1.0)  # kernel definition
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
        if C_value == 0.0001:
            Error[0] = Each_CV_Error/len(partitions)
            #print("a:", Error[0])
        elif C_value == 0.001:
            Error[1] = Each_CV_Error/len(partitions)
            #print("b:",Error[1])
        elif C_value == 0.01:
            Error[2] = Each_CV_Error/len(partitions)
            #print("c:",Error[2])
        elif C_value == 0.1:
            Error[3] = Each_CV_Error/len(partitions)
            #print("d:",Error[3])
        elif C_value == 1:
            Error[4] = Each_CV_Error/len(partitions)
            #print("e:",Error[4])

    if np.argmin(Error) == 0:       #Select the C depending on the error.
        Choice_a = Choice_a + 1
    elif np.argmin(Error) == 1:
        Choice_b = Choice_b + 1
    elif np.argmin(Error) == 2:
        Choice_c = Choice_c + 1
    elif np.argmin(Error) == 3:
        Choice_d = Choice_d + 1
    elif np.argmin(Error) == 4:
        Choice_e = Choice_e + 1
    Total_Run_Error = Total_Run_Error + Each_CV_Error

#Problem 7
print("C=0.0001 chosen:", Choice_a)
print("C=0.001 chosen:", Choice_b)
print("C=0.01 chosen:", Choice_c)
print("C=0.1 chosen:", Choice_d)
print("C=1 chosen:", Choice_e)