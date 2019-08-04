#Sung Hoon Choi
#CS/CNS/EE156a HW1 Problem 7~10

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
Total_iteration = 0         #Total iterations over all runs
Total_Run = 1000            #Total number of runs
Test_Data_Num = 1000        #Total number of testing data (used for testing the hypothesis g(x))
Total_Wrong_Points = 0      #Total number of f(x) != g(x) over all runs

#Training begins
for run in range (0,Total_Run):
    N=10
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


    iteration = 0
    mismatch = 0
    for i in range (0, 5000):
        random_gen = (int) (np.random.rand(1)*N)  #pick random misclassified points
        g_x = np.dot(w.T, x[random_gen,0:3])      #g(x) = dot(w,x)
        if(x[random_gen,3] != np.sign(g_x)):      #if y is not equal to the sign of g(x)
            w = w + x[random_gen,3]*x[random_gen,0:3]     # w = w + y*X
            iteration = iteration + 1


#Testing begins
    #Generate random points to examine the error rate of g(x)
    test = np.zeros([Test_Data_Num, 4])
    for i in range(0, Test_Data_Num):
        test[i, 0] = 1  # x0 = 1
        test[i, 1:3] = generate_random_point()
        f_x = target_info[0] * test[i, 1] + target_info[1]
        test[i, 3] = Label_data(test[i, 1:3], f_x)  # y (label)

    #Examine the error using the generated test points
    wrong = 0
    for i in range (0,Test_Data_Num):
        g_x = np.dot(w.T, test[i, 0:3])  # g(x) = dot(w,x)
        if (test[i, 3] != np.sign(g_x)):
            wrong = wrong + 1            #if the hypothesis is wrong, increase the counter 'wrong'.

    Total_Wrong_Points = Total_Wrong_Points + wrong     #Add up the number of all wrong points
    Total_iteration = Total_iteration + iteration       #Add up the number of iterations


print('Total iterations: ', Total_iteration)
print('Average iterations: ', Total_iteration/Total_Run)
print('Total Wrong Points: ', Total_Wrong_Points)
print('Error Rate: ', Total_Wrong_Points/(Test_Data_Num*Total_Run))
                                                    #Test_Data_Num is the number of test data for each run
                                                    #So we have to divide the total number of wrong points by
                                                    #(Test_Data_Num*Total_Run) to obtain the accurate error rate.







