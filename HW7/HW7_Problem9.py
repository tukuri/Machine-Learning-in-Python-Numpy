#Sung Hoon Choi
#CS/CNS/EE156a HW7 Problem 9

from sklearn.svm import SVC   #Used for implementing SVM.
import numpy as np
import random

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
Total_Run = 1000             #Total number of experiment runs.
Test_Data_Num = 1000         #Total number of testing data points
Total_Support_Vector_Num = 0 #Total number of support vectors. Used to find the average number of support vectors.
SVM_better_than_PLA = 0      #Counts the number of cases when SVM did better than PLA.
N=100                         #Number of data points for training.

print("---------------------N=%d----------------------" % N)
for run in range (0,Total_Run):
    clf = SVC(kernel='linear', C=1e12)  #Set C to a very high value for the hard margin.

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

    sum_all_one_side  = 0
    for i in range (0, N):
        sum_all_one_side = sum_all_one_side + x[i,3]

    if (sum_all_one_side == N or sum_all_one_side == -N):
        continue;

    #Fit the SVM.
    clf.fit(x[:,0:3],x[:,3])
    #Get the number of support vectors.
    Total_Support_Vector_Num = Total_Support_Vector_Num + len(clf.support_)


    #Go through PLA until the hypothesis converges.
    while (1):
        misclassified_arr = []
        for i in range (0,N):
            g_x = np.dot(w.T, x[i, 0:3])  # g(x) = dot(w,x)
            if(x[i,3] != np.sign(g_x)):
                misclassified_arr.append(i)
        if(len(misclassified_arr) == 0):        #escape the loop when PLA converges.
           break
        w = w + x[random.choice(misclassified_arr), 3] * x[random.choice(misclassified_arr), 0:3]  # w = w + y*X


#Testing begins
    #Generate random points to examine the error rate of g(x)
    test = np.zeros([Test_Data_Num, 4])
    for i in range(0, Test_Data_Num):
        test[i, 0] = 1  # x0 = 1
        test[i, 1:3] = generate_random_point()
        f_x = target_info[0] * test[i, 1] + target_info[1]
        test[i, 3] = Label_data(test[i, 1:3], f_x)  # y (label)

    svm_wrong_counter = 0
    svm_predict = clf.predict(test[:,0:3])
    for i in range(0,Test_Data_Num):
        if (test[i,3] != svm_predict[i]):
            svm_wrong_counter = svm_wrong_counter + 1  #Count the data points misclassified by SVM.

    #Examine the error using the generated test points
    pla_wrong = 0
    for i in range (0,Test_Data_Num):
        g_x = np.dot(w.T, test[i, 0:3])  # g(x) = dot(w,x)
        if (test[i, 3] != np.sign(g_x)):
            pla_wrong = pla_wrong + 1            #Count the data points misclassified by PLA.

    #If SVM did better than PLA, increase the counter.
    if (svm_wrong_counter < pla_wrong):
        SVM_better_than_PLA = SVM_better_than_PLA + 1

print("svm better than pla: %d times " % SVM_better_than_PLA)
print("svn better than pla: %.2f percent" % (SVM_better_than_PLA/Total_Run))
