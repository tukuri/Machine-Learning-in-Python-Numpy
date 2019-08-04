#Sung Hoon Choi
#CS/CNS/EE156a HW2 Problem 9

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
Total_hypothesis1_unmatched = 0
Total_hypothesis2_unmatched = 0
Total_hypothesis3_unmatched = 0
Total_hypothesis4_unmatched = 0
Total_hypothesis5_unmatched = 0

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
        x[random_index ,3] = -x[random_index ,3]

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

    #hyothesis [a]
    option1_g_output = np.zeros((N,1))
    option1_W = np.array([-1, -0.05, 0.08, 0.13, 1.5, 1.5])
    for i in range (0,N):
        option1_g_output[i] = np.sign(np.dot(option1_W.T, X[i]))
    option1_g_output = np.squeeze(option1_g_output)

    #hypothesis [b]
    option2_g_output = np.zeros((N,1))
    option2_W = np.array([-1, -0.05, 0.08, 0.13, 1.5, 15])
    for i in range (0,N):
        option2_g_output[i] = np.sign(np.dot(option2_W.T, X[i]))
    option2_g_output = np.squeeze(option2_g_output)

    #hypothesis [c]
    option3_g_output = np.zeros((N,1))
    option3_W = np.array([-1, -0.05, 0.08, 0.13, 15, 1.5])
    for i in range (0,N):
        option3_g_output[i] = np.sign(np.dot(option3_W.T, X[i]))
    option3_g_output = np.squeeze(option3_g_output)

    #hypothesis [d]
    option4_g_output = np.zeros((N,1))
    option4_W = np.array([-1, -1.5, 0.08, 0.13, 0.05, 0.05])
    for i in range (0,N):
        option4_g_output[i] = np.sign(np.dot(option4_W.T, X[i]))
    option4_g_output = np.squeeze(option4_g_output)

    #hypothesis [e]
    option5_g_output = np.zeros((N,1))
    option5_W = np.array([-1, -0.05, 0.08, 1.5, 0.15, 0.15])
    for i in range (0,N):
        option5_g_output[i] = np.sign(np.dot(option5_W.T, X[i]))
    option5_g_output = np.squeeze(option5_g_output)

    #compare with hypothesis [a]
    option1_g_output = np.squeeze(option1_g_output)
    hypothesis1_unmatched = 0
    for i in range (0,N):
        if g_output[i] != option1_g_output[i]:
            hypothesis1_unmatched = hypothesis1_unmatched +1

    #compare with hypothesis [b]
    option2_g_output = np.squeeze(option2_g_output)
    hypothesis2_unmatched = 0
    for i in range (0,N):
        if g_output[i] != option2_g_output[i]:
            hypothesis2_unmatched = hypothesis2_unmatched +1

    #compare with hypothesis [c]
    option3_g_output = np.squeeze(option3_g_output)
    hypothesis3_unmatched = 0
    for i in range (0,N):
        if g_output[i] != option3_g_output[i]:
            hypothesis3_unmatched = hypothesis3_unmatched +1

    #compare with hypothesis [d]
    option4_g_output = np.squeeze(option4_g_output)
    hypothesis4_unmatched = 0
    for i in range (0,N):
        if g_output[i] != option4_g_output[i]:
            hypothesis4_unmatched = hypothesis4_unmatched +1

    #compare with hypothesis [e]
    option5_g_output = np.squeeze(option5_g_output)
    hypothesis5_unmatched = 0
    for i in range (0,N):
        if g_output[i] != option5_g_output[i]:
            hypothesis5_unmatched = hypothesis5_unmatched +1

    Total_hypothesis1_unmatched = Total_hypothesis1_unmatched + hypothesis1_unmatched
    Total_hypothesis2_unmatched = Total_hypothesis2_unmatched + hypothesis2_unmatched
    Total_hypothesis3_unmatched = Total_hypothesis3_unmatched + hypothesis3_unmatched
    Total_hypothesis4_unmatched = Total_hypothesis4_unmatched + hypothesis4_unmatched
    Total_hypothesis5_unmatched = Total_hypothesis5_unmatched + hypothesis5_unmatched

print("my W: ", W)
print("Average error with current hypothesis 1: ", Total_hypothesis1_unmatched/(N*Total_Run))
print("Average error with current hypothesis 2: ", Total_hypothesis2_unmatched/(N*Total_Run))
print("Average error with current hypothesis 3: ", Total_hypothesis3_unmatched/(N*Total_Run))
print("Average error with current hypothesis 4: ", Total_hypothesis4_unmatched/(N*Total_Run))
print("Average error with current hypothesis 5: ", Total_hypothesis5_unmatched/(N*Total_Run))
