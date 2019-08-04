#Sung Hoon Choi
#CS/CNS/EE156a FINAL Problem 7
import numpy as np

def extract_data(filename):
    data_array = []
    for line in open(filename):
        data=[]
        data_entries = line.split('  ')
        data_row = [float(data_entries[1]),float(data_entries[2]),float(data_entries[3].rstrip("\n"))]
        data_array.append(data_row)
    return np.array(data_array)

def one_vs_all_label(data_array, digit):
    labelled_data_array = []
    for i in range (0, len(data_array)):
        if data_array[i,0] == digit:
            labelled_data_array.append(1)
        else:
            labelled_data_array.append(-1)
    return np.array(labelled_data_array)

def calculate_binary_error(g_x, f_x):
    error_count = 0
    for i in range (0,len(g_x)):
        if(g_x[i] != f_x[i]):
            error_count = error_count + 1
    return error_count/len(g_x)

def calculate_regularized_weight(x, y, Lambda):
    reg_weight_inter1 = np.dot(np.transpose(x),x)+Lambda*np.identity(3)
    reg_weight_inter2 = np.dot(np.linalg.inv(reg_weight_inter1),np.transpose((x)))
    reg_weight = np.dot(reg_weight_inter2,y)
    return reg_weight

extracted = extract_data("features.train.txt")
transformed_x = np.array(extracted[:,0:3])
transformed_x[:,0] = 1

for n in range(5,10):
    y = one_vs_all_label(extracted, n)
    weights = calculate_regularized_weight(transformed_x,y,1)
    weights = np.array([weights])
    g_x = np.squeeze(np.dot(weights,transformed_x.T))
    sign_g_x = np.sign(g_x)
    print("%d vs all: %f " %(n, calculate_binary_error(sign_g_x,y.T)))
