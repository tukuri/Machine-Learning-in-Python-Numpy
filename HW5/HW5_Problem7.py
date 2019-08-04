#Sung Hoon Choi
#CS/CNS/EE156a HW5 Problem 7
import math

def Error(u,v):
    return (u*math.exp(v)-2*v*math.exp(-u))**2

def gradient_u(u,v):
    return 2*(u*math.exp(v)-2*v*math.exp(-u))*(math.exp(v)+2*v*math.exp(-u))

def gradient_v(u,v):
    return 2*(u*math.exp(v)-2*v*math.exp(-u))*(u*math.exp(v)-2*math.exp(-u))

u=1
v=1
iteration = 0
learning_rate = 0.1
Max_iteration=50
initial_error = Error(u,v)

for i in range (0,15):
    iteration = iteration +1
    u = u - learning_rate*gradient_u(u,v)
    v = v - learning_rate*gradient_v(u,v)

print("Taken iterations: %d Error: %f" %(iteration,Error(u,v))) #Answer for problem 7
