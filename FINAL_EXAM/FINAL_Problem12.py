#Sung Hoon Choi
#CS/CNS/EE156a FINAL Problem 12

import numpy as np
from sklearn.svm import SVC   #Used for implementing SVM.

x = (((1,0),(0,1),(0,-1),(-1,0),(0,2),(0,-2),(-2,0)))
y = ((-1),(-1),(-1),(1),(1),(1),(1))

clf = SVC(C=1e12, kernel='poly', degree=2, coef0=1.0, gamma= 'auto') #kernel definition
clf.fit(x,y) #fit the SVM
print("vectors: %d" %len(clf.support_)) #number of support vectors

