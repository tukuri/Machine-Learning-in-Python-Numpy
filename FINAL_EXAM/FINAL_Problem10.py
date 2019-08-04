#Sung Hoon Choi
#CS/CNS/EE156a FINAL Problem 10
import numpy as np

def extract_data(filename):
    data_array = []
    for line in open(filename):
        data=[]
        data_entries = line.split('  ')
        data_row = [float(data_entries[1]),float(data_entries[2]),float(data_entries[3].rstrip("\n"))]
        data_array.append(data_row)
    return np.array(data_array)

def one_vs_one_label(data_array, digit1, digit2):
    labelled_data_array = []
    for i in range (0, len(data_array)):
        if data_array[i,0] == digit1:
            labelled_data_array.append(1)
        elif data_array[i,0] == digit2:
            labelled_data_array.append(-1)
    return np.array(labelled_data_array)

def filter_rest_of_digits(data_array, digit1, digit2):
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

def calculate_regularized_weight(x, y, Lambda, z_param_num):
    reg_weight_inter1 = np.dot(np.transpose(x),x)+Lambda*np.identity(z_param_num)
    reg_weight_inter2 = np.dot(np.linalg.inv(reg_weight_inter1),np.transpose((x)))
    reg_weight = np.dot(reg_weight_inter2,y)
    return reg_weight

def nonlinear_trans(x,lineNum):
    nonlin_transformed_data = []
    for i in range(0,lineNum):
        nonlin_transformed_row = []
        nonlin_transformed_row.append(1)
        nonlin_transformed_row.append(x[i][1])
        nonlin_transformed_row.append(x[i][2])
        nonlin_transformed_row.append(x[i][1]*x[i][2])
        nonlin_transformed_row.append(x[i][1]**2)
        nonlin_transformed_row.append((x[i][2]**2))
        nonlin_transformed_data.append(nonlin_transformed_row)
    return np.array(nonlin_transformed_data)

extracted_in = filter_rest_of_digits(extract_data("features.train.txt"),1,5)
transformed_x_in = np.array(extracted_in[:,0:3])
transformed_x_in[:,0] = 1
nonlinear_transformed_x_in = nonlinear_trans(transformed_x_in,len(transformed_x_in))

extracted_out = filter_rest_of_digits(extract_data("features.test.txt"),1,5)
transformed_x_out = np.array(extracted_out[:,0:3])
transformed_x_out[:,0] = 1
nonlinear_transformed_x_out = nonlinear_trans(transformed_x_out,len(transformed_x_out))

n = 5
y_in = one_vs_one_label(extracted_in, 1, 5)
y_out = one_vs_one_label(extracted_out,1, 5)

for Lambda in [1,0.01]:
    print("Lambda = %.2f---------------------------" %Lambda)
    weights_nonlinear = calculate_regularized_weight(nonlinear_transformed_x_in,y_in,Lambda,6)
    weights_nonlinear = np.array([weights_nonlinear])
    g_x_nonlinear_in = np.squeeze(np.dot(weights_nonlinear,nonlinear_transformed_x_in.T))
    sign_g_x_nonlinear_in = np.sign(g_x_nonlinear_in)
    g_x_nonlinear_out = np.squeeze(np.dot(weights_nonlinear,nonlinear_transformed_x_out.T))
    sign_g_x_nonlinear_out = np.sign(g_x_nonlinear_out)
    print("Ein With Transform: %d vs all: %f " %(n, calculate_binary_error(sign_g_x_nonlinear_in,y_in.T)))
    print("Eout With Transform: %d vs all: %f " %(n, calculate_binary_error(sign_g_x_nonlinear_out,y_out.T)))
print("-----------------------------------------------")