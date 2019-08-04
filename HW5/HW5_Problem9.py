#Sung Hoon Choi
#CS/CNS/EE156a HW5 Problem 9
import math
import random
import numpy as np



def gen_target_func():  # generate a target function(f(x)) and return the corresponding vertical coordinate
    # input: none
    # output: target_function. format: [slope, y_intercept]
    rnd_x1 = np.zeros(2)
    rnd_x2 = np.zeros(2)

    for i in range(0, 2):
        rnd_x1[i] = (1 if np.random.rand(1) < 0.5 else -1) * np.random.rand(1)

    for i in range(0, 2):
        rnd_x2[i] = (1 if np.random.rand(1) < 0.5 else -1) * np.random.rand(1)

    slope_target_func = (rnd_x2[1] - rnd_x1[1]) / (rnd_x2[0] - rnd_x1[0])  # slope = (y2-y1)/(x2-x1)

    y_intercept = rnd_x2[1] - slope_target_func * rnd_x2[0]

    return [slope_target_func, y_intercept]


def Label_data(X_vector, target_f):  # return a correct label(1 or -1) by using the input vector and target equation f.
    # inputs
    # X_vector : input point's coordinate. format: [a, b]
    # target_f : target function. format: a
    # outputs
    # y : correct label for the input vector. format: a (1 or -1)

    if (X_vector[1] > target_f):  # if the input's vertical coordinate is above the target function, return 1 label
        # if the input's vertical coordinate is below the target function, return -1 label
        return 1
    else:
        return -1


def generate_random_point():  # generate random data point's coordinate
    # inputs
    # none
    # outputs
    # x: random points. format: [a,b]
    x = np.zeros(2)
    x[0] = (1 if np.random.rand(1) < 0.5 else -1) * np.random.rand(1)
    x[1] = (1 if np.random.rand(1) < 0.5 else -1) * np.random.rand(1)
    return x

def calculate_Error(N,weights,x):
    E_Sum=0
    for i in range (0,N):
        E_Sum = E_Sum + (math.log(1+math.exp(-x[i,3]*np.dot(weights.T,x[i,0:3]))))

    E_Sum = E_Sum/N
    return E_Sum

def calculate_Grad_Descent(weights,y,x):
    return ((-y * x)/ (1 + math.exp(y * np.dot(weights.T, x))))

N=100
Total_Run = 100
Total_Error = 0
weights = np.array([0,0,0])
Total_Epoch_Sum = 0  #for Problem 9

#Generate training points with their labels using the target function.
for run in range (0,Total_Run):

    target_info = gen_target_func()     #target_info = [slope_target_func, y_intercept]

    x = np.zeros([N,4])         # x - [[1, x1,x2,label(y)],
                                #      [1, x1,x2,label(y)],
                                #      ..
                                #      [1, x1,x2,label(y)]]

    w = np.zeros([3,1])         #initializing w vector
    w = np.squeeze(w)           #remove one dimension for matrix operations

    # generate N random data points with their correct labels based on the current target function f(x)
    for i in range (0, N):
        x[i,0] = 1                                     #x0 = 1
        x[i,1:3] = generate_random_point()             #random data points coordinate data
        f_x = target_info[0] * x[i,1] + target_info[1] #obtaining the target equation f
        x[i,3] = Label_data(x[i,1:3],f_x)              #using f, obtain the label(y) and append it to the array


    #Calculate the g and its weights.
    weights_prev = np.array([5,5,5])
    Final_weights = np.array([0,0,0])
    weights = np.array([0,0,0])
    epoch = 0
    while (np.linalg.norm(weights-weights_prev)>0.01):
        weights_prev = weights
        for i in random.sample(range(0,N),N):
            weights = weights - 0.01*calculate_Grad_Descent(weights,x[i,3],x[i,0:3])
        epoch = epoch+1

    Total_Epoch_Sum = Total_Epoch_Sum + epoch

    #generate data points for test.
    x_test = np.zeros([N,4])    # x - [[1, x1,x2,label(y)],
                                    #      [1, x1,x2,label(y)],
                                    #      ..
                                    #      [1, x1,x2,label(y)]]

    # generate N random data points with their correct labels based on the current target function f(x)
    for i in range (0, N):
        x_test[i,0] = 1                                     #x0 = 1
        x_test[i,1:3] = generate_random_point()             #random data points coordinate data
        f_x = target_info[0] * x_test[i,1] + target_info[1] #obtaining the target equation f
        x_test[i,3] = Label_data(x_test[i,1:3],f_x)         #using f, obtain the label(y) and append it to the array

    Total_Error = Total_Error + calculate_Error(N,weights,x_test)

#Calculate the Eout
print("Total_Error:", Total_Error/Total_Run) #Answer for Problem 8
print("Average Epoch:", Total_Epoch_Sum/Total_Run) #Answer for Problem 9