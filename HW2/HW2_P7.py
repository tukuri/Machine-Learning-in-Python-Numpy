#Sung Hoon Choi
#CS/CNS/EE156a HW2 Problem 7

import numpy as np

def gen_target_func():             #generate a target function(f(x)) and return the corresponding vertical coordinate
                                    #input: none
                                    #output: target_function. format: [slope, y_intercept]
    rnd_x1 = np.zeros(2)
    rnd_x2 = np.zeros(2)

    for i in range (0,2):
        rnd_x1[i] = (1 if np.random.rand(1) < 0.5 else -1) * np.random.rand(1)

    for i in range (0,2):
        rnd_x2[i] = (1 if np.random.rand(1) < 0.5 else -1) * np.random.rand(1)

    slope_target_func = (rnd_x2[1]-rnd_x1[1])/(rnd_x2[0]-rnd_x1[0]) # slope = (y2-y1)/(x2-x1)

    y_intercept = rnd_x2[1]-slope_target_func*rnd_x2[0]

    return [slope_target_func, y_intercept]

def Label_data(X_vector, target_f): #return a correct label(1 or -1) by using the input vector and target equation f.
                                    #inputs
                                    # X_vector : input point's coordinate. format: [a, b]
                                    # target_f : target function. format: a
                                    #outputs
                                    # y : correct label for the input vector. format: a (1 or -1)

    if(X_vector[1] > target_f):    #if the input's vertical coordinate is above the target function, return 1 label
                                   #if the input's vertical coordinate is below the target function, return -1 label
        return 1
    else:
        return -1

def generate_random_point():      #generate random data point's coordinate
                                  #inputs
                                  # none
                                  #outputs
                                  # x: random points. format: [a,b]
    x = np.zeros(2)
    x[0] = (1 if np.random.rand(1) < 0.5 else -1) * np.random.rand(1)
    x[1] = (1 if np.random.rand(1) < 0.5 else -1) * np.random.rand(1)
    return x

#Initializing the constants which will be used for training and testing
Total_iteration = 0           #Total iterations over all runs
Total_Run = 1000               #Total number of experiments
Test_Data_Num = 1000          #Total number of testing data (used for testing the hypothesis g(x))
Total_Wrong_Points = 0        #Total number of f(x) != g(x) over all runs
Total_In_Unmatched_Num = 0    #Total number of incorrect data points for training samples.
Total_Out_Unmatched_Num = 0   #Total number of incorrect data points for testing data points.
OutOfSample_Num = 1000        #number of test data points for measuring out-of-sample error, Eout.

for run in range (0,Total_Run):
    N=10                                #N=100 for problem 5 and 6. N=10 for problem 7.
    target_info = gen_target_func()     #target_info = [slope_target_func, y_intercept]

    x = np.zeros([N,4])         # x - [[1, x1,x2,label(y)],
                                #      [1, x1,x2,label(y)],
                                #      ..
                                #      [1, x1,x2,label(y)]]


    # generate N random data points with their correct labels based on the current target function f(x)
    for i in range (0, N):
        x[i,0] = 1                                     #x0 = 1
        x[i,1:3] = generate_random_point()             #random data points coordinate data
        f_x = target_info[0] * x[i,1] + target_info[1] #obtaining the target equation f
        x[i,3] = Label_data(x[i,1:3],f_x)              #using f, obtain the label(y) and append it to the array

    X = x[:,0:3]
    Y = x[:,3]

    X_pseudo_inverse = np.dot(np.linalg.inv(np.dot(X.T,X)),X.T)

    W = np.dot(X_pseudo_inverse,Y)

    g_output = np.zeros((N,1))
    for i in range (0,N):
        g_output[i] = np.sign(np.dot(W.T,X[i]))

    g_output = np.squeeze(g_output)


    insample_unmatched = 0
    for i in range (0,N):
        if g_output[i] != Y[i]:
            insample_unmatched = insample_unmatched + 1

    Total_In_Unmatched_Num = Total_In_Unmatched_Num + insample_unmatched

##perceptron.
#This part of code is for problem 7.

    for i in range (0, N):
        x[i,0] = 1                                     #x0 = 1
        x[i,1:3] = generate_random_point()             #random data points coordinate data
        f_x = target_info[0] * x[i,1] + target_info[1] #obtaining the target equation f
        x[i,3] = Label_data(x[i,1:3],f_x)              #using f, obtain the label(y) and append it to the array

    iteration = 0
    mismatch = 0
    for i in range (0, 1000):
        random_gen = (int) (np.random.rand(1)*N)  #pick random misclassified points
        g_x = np.dot(W.T, x[random_gen,0:3])      #g(x) = dot(w,x)
        if(x[random_gen,3] != np.sign(g_x)):      #if y is not equal to the sign of g(x)
            W = W + x[random_gen,3]*x[random_gen,0:3]     # w = w + y*X
            iteration = iteration + 1

            ##perceptron end
    Total_iteration = Total_iteration + iteration  # Add up the number of iterations

print('Total iterations: ', Total_iteration)
print('Average iterations: ', Total_iteration / Total_Run)