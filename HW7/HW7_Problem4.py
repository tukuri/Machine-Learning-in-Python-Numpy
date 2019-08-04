#Sung Hoon Choi
#CS/CNS/EE156a HW7 Problem 4
import numpy as np

IN_DTA_NUM = 35         # number of insample data points
OUT_DTA_NUM = 250       # number of outsample data points

def extract_data(filename):
    data_array = []
    for line in open(filename):
        data=[]
        data_entries = line.split('  ')
        data_row = [float(data_entries[1]),float(data_entries[2]),float(data_entries[3].rstrip("\n"))]
        data_array.append(data_row)
    return data_array           #return the data from given dta file.
                                # Format: (x1,x2,label)
                                #             ...
                                #         (x1,x2,label)

def nonlinear_trans(x,lineNum,k):
    nonlin_transformed_data = []
    for i in range(0,lineNum):
        nonlin_transformed_row = []
        nonlin_transformed_row.append(1)
        nonlin_transformed_row.append(x[i][0])
        nonlin_transformed_row.append(x[i][1])
        nonlin_transformed_row.append(x[i][0]**2)
        nonlin_transformed_row.append(x[i][1]**2)
        nonlin_transformed_row.append((x[i][0]*x[i][1]))
        nonlin_transformed_row.append(abs(x[i][0]-x[i][1]))
        nonlin_transformed_row.append(abs(x[i][0]+x[i][1]))
        nonlin_transformed_data.append(nonlin_transformed_row[0:k+1])
    return nonlin_transformed_data              # Format: (1,x1,x2,x1^2,x2^2,...)
                                                #                 ...
                                                #         (1,x1,x2,x1^2,x2^2,...)

def calculate_weight(nonlinear_x, label):
    return np.dot(np.linalg.pinv(nonlinear_x), label)

def calculate_error(weights, nonlin_input_x, y, data_num):
    error_count = 0
    for i in range(0,data_num):
        g = np.sign(np.dot(weights.T,nonlin_input_x[i]))
        if(g != y[i]):
            error_count = error_count + 1
    print("Error: ", error_count/data_num)


for k in range(3,8):
    print("k = %d ---------------" % k)
    insample_data = extract_data("in.dta")
    insample_data_np = np.array(insample_data)  # turn it into a numpy array to use numpy library functions
    y = insample_data_np[:, 2]  # extract y-labels from the input data
    transformed_data = nonlinear_trans(insample_data, IN_DTA_NUM,k)
    transformed_data = np.array(transformed_data)
    weights = calculate_weight(transformed_data[25:35,:],y[25:35])

    outsample_data = extract_data("out.dta")
    outsample_data_np = np.array(outsample_data)  # turn it into a numpy array to use numpy library functions
    y = outsample_data_np[:, 2]  # extract y-labels from the input data
    transformed_data = nonlinear_trans(outsample_data, OUT_DTA_NUM, k)
    calculate_error(weights, transformed_data, y, OUT_DTA_NUM)  # Eout - OutSample Error