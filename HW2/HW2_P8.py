#Sung Hoon Choi
#CS/CNS/EE156a HW2 Problem 8

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
Total_iteration = 0         #Total iterations over all runs
Total_Run = 1             #Total number of experiments
Total_Unmatched_Num = 0

#Training begins
for run in range (0,Total_Run):
    N=1000

    x = np.zeros([N,4])         # x - [[1, x1,x2,label(y)],
                                #      [1, x1,x2,label(y)],
                                #      ..
                                #      [1, x1,x2,label(y)]]

    w = np.zeros([3,1])         #initializing w vector
    w = np.squeeze(w)           #remove one dimension for matrix operations

    # generate N random data points with their correct labels based on the current target function f(x)
    for i in range (0, N):
        x[i,0] = 1                                 #x0 = 1
        x[i,1:3] = generate_random_point()         #random data points coordinate data
        f_x = x[i,1]**2 * x[i,2]**2 - 0.6          #f=sign(x1^2+x2^2-0.6)
        x[i,3] = np.sign(f_x)                      #using f, obtain the label(y) and append it to the array


    # generate random noise by flipping 1/10 of samples.
    for i in range (0, (int)(N/10)):
        random_index = (int) (np.random.rand(1)*1000)
        #print("rand index: " , random_index)
        x[random_index,3] = -x[random_index,3]

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

    #print("Ein: ", insample_unmatched/N)                     #Ein for each test.
    Total_Unmatched_Num = Total_Unmatched_Num + insample_unmatched

print("Avergae Ein: ", Total_Unmatched_Num/(N*Total_Run))   #Calculate the average of Ein over the entire test runs.
