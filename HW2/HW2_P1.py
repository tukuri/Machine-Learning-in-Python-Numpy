#Sung Hoon Choi
#CS/CNS/EE156a HW2 Problem 1-2

import numpy as np

Exp_Num = 1000000
Flip_Times = 10
coin_data = np.zeros((1000,Flip_Times))
Total_V1 = 0
Total_Vrand = 0
Total_Vmin = 0

for experiment in range (0,Exp_Num):
    for i in range(0,1000):
        each_coin_data = np.zeros(Flip_Times)
        for j in range(0,Flip_Times):
            if np.random.rand(1) > 0.5:
                each_coin_data[j] = 1
            else:
                each_coin_data[j] = 0

        coin_data[i,0:Flip_Times] = each_coin_data

    #print("coin_data: \n",coin_data)

    #Find C1
    C1 = coin_data[0,:]
    #print("C1: ",C1)

    #Find Crand
    rand_index = (int)(np.random.rand(1)*1000)
    #print("rand_index: ",rand_index)
    Crand = coin_data[rand_index,:]
    #print("Crand: " ,Crand)

    #Find Cmin
    sum_coin_data = (np.sum(coin_data, axis=1)).T # sum_coin_data's shape = (1000,0)
    #print("sum_coin:\n", sum_coin_data)
    min_coin_sum = min(sum_coin_data)
    #print("min_coin_val: ", min_coin_sum)

    for index in range (0,1000):
        if sum_coin_data[index] == min_coin_sum:
            Cmin_coin_index = index

    Cmin = coin_data[Cmin_coin_index,:]
    #print("Cmin: ", Cmin)

    #Find V1
    V1 = np.sum(C1)/Flip_Times
    #print("V1: ", V1)

    #Find Vrand
    Vrand = np.sum(Crand)/Flip_Times
    #print("Vrand: ", Vrand)

    #Find Vmin
    Vmin = np.sum(Cmin)/Flip_Times
    #print("Vmin: ", Vmin)

    Total_V1 = Total_V1 + V1
    Total_Vrand = Total_Vrand + Vrand
    Total_Vmin = Total_Vmin + Vmin

Average_V1 = Total_V1/Exp_Num
Average_Vrand = Total_Vrand/Exp_Num
Average_Vmin = Total_Vmin/Exp_Num

print("Average V1: ", Average_V1)           #0.5046
print("Average Vrand: ", Average_Vrand)     #0.4982
print("Average Vmin: ", Average_Vmin)       #0.0371
