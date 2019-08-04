#Sung Hoon Choi
#CS/CNS/EE156a FINAL Problem 17

import numpy as np
import math

def generate_random_point():  # generate random data point's coordinate in [-1,1]
    x = np.zeros(2)
    x[0] = (1 if np.random.rand(1) < 0.5 else -1) * np.random.rand(1)
    x[1] = (1 if np.random.rand(1) < 0.5 else -1) * np.random.rand(1)
    return x

def label_data(x):
    f_x = np.sign(x[1]-x[0]+0.25*np.sin(np.pi*x[0]))
    return f_x

def calculate_distance(x1,x2):
    return math.sqrt((x1[0]-x2[0])**2 + (x1[1]-x2[1])**2)

def calculate_binary_error_abs(g_x, f_x):
    error_count = 0
    for i in range (0,len(g_x)):
        if(g_x[i] != f_x[i]):
            error_count = error_count + 1
    return error_count

not_separable = 0
kernel_beats_regular = 0
K=9
NUM_DATA = 100
gamma = 1.5
total_run = 20


for run in range(0,total_run):
    ################################TRAINING BEGINS#####################################
    check_empty_cluster = 1     #Flag to check if there is an empty cluster.

    #Generate Train data.
    x = generate_random_point()          # First stack of x coordinate is built manually.
    y = label_data(x)                    # First stack of y label is built manually.
    for i in range(0, NUM_DATA-1):       # Remaining 99 stacks are built by for loop.
        x_new_stack = generate_random_point()
        y_new_stack = label_data(x_new_stack)
        x = np.vstack((x, x_new_stack))
        y = np.vstack((y, y_new_stack))

    x = np.squeeze(x)
    y = np.squeeze(y)

    for gamma in [1.5,2]: #Problem 17
        print("gamma = %.1f" %(gamma))
        #Regular Form(Lloyd method)
        centroid = np.zeros((K,2))
        for i in range(0,K):
            centroid[i] = generate_random_point()

        #Assign clusters to the data points. (Save all the cluster labels in a separate x_cluster)
        #Tracked by corresponding indices.
        dist = np.zeros((1, K))
        x_cluster = []
        for i in range (0,NUM_DATA):
            for j in range(0, K):
                dist[j] = calculate_distance(x[i],centroid[j])
                dist = np.squeeze(dist)
            x_cluster.append(np.argmin(dist))

        for value in range(0,K):
            if value in x_cluster:
                check_empty_cluster = check_empty_cluster*1
            else:
                check_empty_cluster = 0

        if(check_empty_cluster!=0):
            new_centroid = np.zeros((K, 2))
            while(1): #Run until the clusters converge.
                #Find the new mu k. (Find the new centroid coordinates)
                for i in range(0,K):
                    sum = np.zeros((K, 2))
                    num_cluster_data = 0
                    for j in range (0,NUM_DATA):
                        if(x_cluster[j] == i):
                            sum[i] = sum[i] + x[j]
                            num_cluster_data = num_cluster_data + 1
                    new_centroid[i] = sum[i]/num_cluster_data

                prev_cluster = x_cluster
                new_dist = np.zeros((1, K))
                new_x_cluster = []
                #Find the new S k. (Label the data points according to the new centroids)
                for i in range (0,NUM_DATA):
                    for j in range(0, K):
                        new_dist[j] = calculate_distance(x[i],new_centroid[j])
                        new_dist = np.squeeze(new_dist)
                    new_x_cluster.append(np.argmin(new_dist))
                x_cluster = new_x_cluster
                if(np.array_equal(prev_cluster,new_x_cluster)): #If converged.
                    break

            #Calculate phi
            phi = np.zeros((NUM_DATA,K))
            for i in range(0,K):
                for j in range(0,NUM_DATA):
                    phi[j,i] = math.exp(-gamma*(calculate_distance(x[j],new_centroid[i]))**2)

            weights = np.matmul(np.linalg.pinv(phi),y)
            h = np.sign(np.matmul(phi, weights))

            regular_in_error = 0
            for i in range(0, NUM_DATA):
                if (h[i] != y[i]):
                    regular_in_error = regular_in_error + 1
            print("E_in error: ", regular_in_error)
            ###Now we have the weights. Let's test the hypothesis by using the test data###
            ################################TEST BEGINS#####################################
            #Generate Test data
            x = generate_random_point()       # First stack of x coordinate is built manually.
            y = label_data(x)                 # First stack of y label is built manually.
            for i in range(0, NUM_DATA - 1):  # Remaining 99 stacks are built by for loop.
                x_new_stack = generate_random_point()
                y_new_stack = label_data(x_new_stack)
                x = np.vstack((x, x_new_stack))
                y = np.vstack((y, y_new_stack))


            # Assign clusters to the data points. (Save all the cluster labels in a separate x_cluster)
            # Tracked by indices.
            dist = np.zeros((1, K))
            x_cluster = []
            for i in range(0, NUM_DATA):
                for j in range(0, K):
                    dist[j] = calculate_distance(x[i], new_centroid[j])
                    dist = np.squeeze(dist)
                x_cluster.append(np.argmin(dist))

            for value in range(0, K):
                if value in x_cluster:
                    check_empty_cluster = check_empty_cluster * 1
                else:
                    check_empty_cluster = 0

            if (check_empty_cluster != 0 and not_separable == 0):
                new_centroid = np.zeros((K, 2))
                for mini_run in range(1, 50):
                    # print("x_cluster", x_cluster)
                    # Find the new mu k. (Find the new centroid coordinates)
                    for i in range(0, K):
                        sum = np.zeros((K, 2))
                        num_cluster_data = 0
                        for j in range(0, NUM_DATA):
                            if (x_cluster[j] == i):
                                sum[i] = sum[i] + x[j]
                                num_cluster_data = num_cluster_data + 1
                        new_centroid[i] = sum[i] / num_cluster_data

                    prev_cluster = x_cluster
                    new_dist = np.zeros((1, K))
                    new_x_cluster = []
                    for i in range(0, NUM_DATA):
                        for j in range(0, K):
                            new_dist[j] = calculate_distance(x[i], new_centroid[j])
                            new_dist = np.squeeze(new_dist)
                        new_x_cluster.append(np.argmin(new_dist))
                    x_cluster = new_x_cluster
                    if (np.array_equal(prev_cluster, new_x_cluster)):  # If converged.
                        break

                phi = np.zeros((NUM_DATA, K))
                for i in range(0, K):
                    for j in range(0, NUM_DATA):
                        phi[j, i] = math.exp(-gamma * (calculate_distance(x[j], new_centroid[i])) ** 2)

                h = np.sign(np.matmul(phi,weights))

                regular_out_error = 0
                for i in range(0,NUM_DATA):
                    if(h[i] != y[i]):
                        regular_out_error = regular_out_error+ 1
                print("E_out error: ", regular_out_error)
    print("--------------------------------------------------")