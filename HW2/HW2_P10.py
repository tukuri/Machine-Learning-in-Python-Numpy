#Sung Hoon Choi
#CS/CNS/EE156a HW2 Problem 10

import numpy as np

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
Total_iteration = 0             #Total iterations over all runs
Total_Run = 1000                #Total number of experiments
OutOfSample_Num = 1000        #number of test data points for measuring out-of-sample error, Eout.
Total_Out_Unmatched_Num = 0   #Total number of incorrect data points for testing data points.
#Training begins
for run in range (0,Total_Run):
    N=1000

    x = np.zeros([N,4])         # x - [[1, x1,x2,label(y)],
                                #      [1, x1,x2,label(y)],
                                #      ..
                                #      [1, x1,x2,label(y)]]

    # generate N random data points with their correct labels based on the current target function f(x)
    for i in range (0, N):
        x[i,0] = 1                                 #x0 = 1
        x[i,1:3] = generate_random_point()         #random data points coordinate data
        f_x = x[i,1]**2 * x[i,2]**2 - 0.6          #f=sign(x1^2+x2^2-0.6)
        x[i,3] = np.sign(f_x)                      #using f, obtain the label(y) and append it to the array

    # generate random noise by flipping 1/10 of samples.
    for i in range (0, (int)(N/10)):
        random_index = (int) (np.random.rand(1)*1000)
        x[random_index,3] = -x[random_index,3]

    # non-linear transformation
    non_linear_x = np.zeros([N,7])
    for i in range(0, N):
        non_linear_x[i,0] = x[i,0]
        non_linear_x[i,1] = x[i,1]
        non_linear_x[i,2] = x[i,2]
        non_linear_x[i,3] = x[i,1]*x[i,2]
        non_linear_x[i,4] = x[i,1]**2
        non_linear_x[i,5] = x[i,2]**2
        non_linear_x[i,6] = x[i,3]      #The label(y) attached to x array

    X = non_linear_x[:,0:6]
    Y = non_linear_x[:,6]

    non_linear_X_pseudo_inverse = np.dot(np.linalg.inv(np.dot(X.T,X)),X.T)

    W = np.dot(non_linear_X_pseudo_inverse,Y)

    g_output = np.zeros((N,1))
    for i in range (0,N):
        g_output[i] = np.sign(np.dot(W.T,X[i]))

    g_output = np.squeeze(g_output)


    ##### Testing begins ####
    test_x = np.zeros([OutOfSample_Num,4])         # test_x - [[1, test_x1,test_x2,label(test_y)],
                                                   #          [1, test_x1,test_x2,label(test_y)],
                                                   #      ..
                                                   #          [1, test_x1,test_x2,label(test_y)]]

    #generate data points to measure out of sample error, Eout.
    for i in range (0, OutOfSample_Num):
        test_x[i,0] = 1                                   #test_x0 = 1
        test_x[i,1:3] = generate_random_point()           #random data points coordinate data
        f_x = test_x[i,1]**2 * test_x[i,2]**2 - 0.6       #f=sign(x1^2+x2^2-0.6) #obtaining the target equation f
        test_x[i,3] = np.sign(f_x)                        #using f, obtain the label(y) and append it to the array


    test_non_linear_x = np.zeros([N, 7])
    for i in range(0, N):
        test_non_linear_x[i, 0] = x[i, 0]
        test_non_linear_x[i, 1] = x[i, 1]
        test_non_linear_x[i, 2] = x[i, 2]
        test_non_linear_x[i, 3] = x[i, 1] * x[i, 2]
        test_non_linear_x[i, 4] = x[i, 1] ** 2
        test_non_linear_x[i, 5] = x[i, 2] ** 2
        test_non_linear_x[i, 6] = x[i, 3]  # The label(y) attached to x array

    test_X = non_linear_x[:, 0:6]
    test_Y = non_linear_x[:, 6]

    test_g_output = np.zeros((OutOfSample_Num,1))
    for i in range (0,OutOfSample_Num):
        test_g_output[i] = np.sign(np.dot(W.T,test_X[i]))

    test_g_output = np.squeeze(test_g_output)

    outofsample_unmatched = 0
    for i in range (0,OutOfSample_Num):                         #Test the performance of g by using test data.
        if test_g_output[i] != test_Y[i]:
            outofsample_unmatched = outofsample_unmatched + 1

    Total_Out_Unmatched_Num = Total_Out_Unmatched_Num + outofsample_unmatched

print("my W: ", W)
print("Average Eout: ", Total_Out_Unmatched_Num/(OutOfSample_Num*Total_Run)) #Calculate the average of Eout
