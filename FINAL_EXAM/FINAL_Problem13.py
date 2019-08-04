#Sung Hoon Choi
#CS/CNS/EE156a FINAL Problem 13

import numpy as np
from sklearn.svm import SVC   #Used for implementing SVM.

def generate_random_point():  # generate random data point's coordinate in [-1,1]
    x = np.zeros(2)
    x[0] = (1 if np.random.rand(1) < 0.5 else -1) * np.random.rand(1)
    x[1] = (1 if np.random.rand(1) < 0.5 else -1) * np.random.rand(1)
    return x

def label_data(x):
    f_x = np.sign(x[1]-x[0]+0.25*np.sin(np.pi*x[0]))
    return f_x

def calculate_binary_error(g_x, f_x):
    error_count = 0
    for i in range (0,len(g_x)):
        if(g_x[i] != f_x[i]):
            error_count = error_count + 1
    return error_count/len(g_x)

run_num = 10000
not_separable = 0
for run in range(0,run_num):
    x = generate_random_point() #First stack of x coordinate is built manually.
    y = label_data(x)           #First stack of y label is built manually.
    for i in range (0,99):      #Remaining 99 stacks are built by for loop.
        x_new_stack = generate_random_point()
        y_new_stack = label_data(x_new_stack)
        x = np.vstack((x, x_new_stack))
        y = np.vstack((y,y_new_stack))

    x = np.squeeze(x)
    y = np.squeeze(y)

    clf = SVC(C=1e12, kernel='rbf', degree=2, gamma=1.5)
    clf.fit(x,y)
    g_x = np.array(clf.predict(x))

    if(calculate_binary_error(g_x,y) != 0):
        not_separable = not_separable + 1
print("Total Run: %d \nNot separable data sets: %d" %(run_num,not_separable))