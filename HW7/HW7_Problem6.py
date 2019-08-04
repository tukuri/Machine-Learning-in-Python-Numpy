#Sung Hoon Choi
#CS/CNS/EE156a HW7 Problem 6

import random

DATA_NUM = 100000
total_e1 = 0
total_e2 = 0
total_e = 0
for i in range(0,DATA_NUM):
    e1 = random.uniform(0,1)
    e2 = random.uniform(0,1)
    e = min(e1,e2)
    total_e1 = total_e1 + e1
    total_e2 = total_e2 + e2
    total_e = total_e + e

print("Expected e1: ", total_e1/DATA_NUM)
print("Expected e2: ", total_e2/DATA_NUM)
print("Expected e: ", total_e/DATA_NUM)
