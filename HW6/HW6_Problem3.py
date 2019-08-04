#Sung Hoon Choi
#CS/CNS/EE156a HW6 Problem 3
import numpy as np

IN_DTA_NUM = 35         # number of insample data points
OUT_DTA_NUM = 250       # number of outsample data points
NONLINEAR_TERMS_NUM = 8 # number of terms in the nonlinear transformed
                        # polynomial: (1,x1,x2,x1^2,x2^2, ... ,abs(x1+x2))

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

def nonlinear_trans(x,lineNum):
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
        nonlin_transformed_data.append(nonlin_transformed_row)
    return nonlin_transformed_data              # Format: (1,x1,x2,x1^2,x2^2,...)
                                                #                 ...
                                                #         (1,x1,x2,x1^2,x2^2,...)

def calculate_weight(nonlinear_x, label):
    return np.dot(np.linalg.pinv(nonlinear_x), label)

def calculate_regularized_weight(nonlinear_x, y, Lambda):
    reg_weight_inter1 = np.dot(np.transpose(nonlinear_x),nonlinear_x)+Lambda*np.identity(NONLINEAR_TERMS_NUM)
    reg_weight_inter2 = np.dot(np.linalg.inv(reg_weight_inter1),np.transpose((nonlinear_x)))
    reg_weight = np.dot(reg_weight_inter2,y)
    return reg_weight

def calculate_error(weights, nonlin_input_x, y, data_num):
    error_count = 0
    for i in range(0,data_num):
        g = np.sign(np.dot(weights.T,nonlin_input_x[i]))
        if(g != y[i]):
            error_count = error_count + 1
    print("Error: ", error_count/data_num)

# Main Code
k = -3
Lambda = 10**k

insample_data = extract_data("in.dta")
insample_data_np = np.array(insample_data)  #turn it into a numpy array to use numpy library functions
y = insample_data_np[:,2]                   #extract y-labels from the input data
transformed_data = nonlinear_trans(insample_data,IN_DTA_NUM)
weights = calculate_regularized_weight(transformed_data, y, Lambda)

calculate_error(weights,transformed_data,y,IN_DTA_NUM)  #Ein - InSample Error


outsample_data = extract_data("out.dta")
outsample_data_np = np.array(outsample_data)  #turn it into a numpy array to use numpy library functions
y = outsample_data_np[:,2]                    #extract y-labels from the input data
transformed_data = nonlinear_trans(outsample_data,OUT_DTA_NUM)
calculate_error(weights,transformed_data,y,OUT_DTA_NUM)  #Eout - OutSample Error