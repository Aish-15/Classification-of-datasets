# -*- coding: utf-8 -*-
"""
@author: aishg
"""

import pandas as pd
import cv2
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import numpy as np
import matplotlib
from sklearn.model_selection import train_test_split
from numpy.linalg import inv
import random as rand
import matplotlib
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report




mangoC = cv2.imread("C:/Users/aishg/OneDrive/Desktop/Big data/FIDS30/mangos/57.jpg")
cantC = cv2.imread("C:/Users/aishg/OneDrive/Desktop/Big data/FIDS30/cantaloupes/16.jpg")
raspC = cv2.imread("C:/Users/aishg/OneDrive/Desktop/Big data/FIDS30/raspberries/17.jpg")


mangoC.shape

plt.imshow(mangoC[:,:,0], cmap=plt.get_cmap('Reds'))
plt.imshow(mangoC[:,:,0], cmap=plt.get_cmap('Greens'))
plt.imshow(mangoC[:,:,0], cmap=plt.get_cmap('Blues'))

plt.imshow(cantC[:,:,0], cmap=plt.get_cmap('Reds'))
plt.imshow(cantC[:,:,0], cmap=plt.get_cmap('Greens'))
plt.imshow(cantC[:,:,0], cmap=plt.get_cmap('Blues'))


plt.imshow(raspC[:,:,0], cmap=plt.get_cmap('Reds'))
plt.imshow(raspC[:,:,0], cmap=plt.get_cmap('Greens'))
plt.imshow(raspC[:,:,0], cmap=plt.get_cmap('Blues'))

mangoG = cv2.cvtColor(mangoC, cv2.COLOR_BGR2GRAY)
heightcG, widthcG = mangoG.shape
mangoG.shape
plt.imshow(mangoG, cmap=plt.get_cmap('gray'))
plt.axis('off')


cantG = cv2.cvtColor(cantC, cv2.COLOR_BGR2GRAY)
heightcG, widthcG = cantG.shape
cantG.shape
plt.imshow(cantG, cmap=plt.get_cmap('gray'))
plt.axis('off')


raspG = cv2.cvtColor(raspC, cv2.COLOR_BGR2GRAY)
heightcG, widthcG = raspG.shape
raspG.shape
plt.imshow(raspG, cmap=plt.get_cmap('gray'))
plt.axis('off')

mango = cv2.resize(mangoG, dsize=(256, 128), interpolation=cv2.INTER_CUBIC)
mango.shape

cant = cv2.resize(cantG, dsize=(256, 216), interpolation=cv2.INTER_CUBIC)
cant.shape

rasp = cv2.resize(raspG, dsize=(256, 160), interpolation=cv2.INTER_CUBIC)
rasp.shape


mango1 = cv2.normalize(mango.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)*255

cant1 = cv2.normalize(cant.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)*255

rasp1 = cv2.normalize(rasp.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)*255


heightm, widthm = mango1.shape
mango1.shape

heightc, widthc = cant1.shape
cant1.shape

heightr, widthr = rasp1.shape
rasp1.shape

plt.imshow(mango1, cmap=plt.get_cmap('gray'))
plt.axis('off')

plt.imshow(cant1, cmap=plt.get_cmap('gray'))
plt.axis('off')


plt.imshow(rasp1, cmap=plt.get_cmap('gray'))
plt.axis('off')

mm = round(((heightm-8)*(widthm-8))/64)
flatm = np.zeros((mm, 65), np.uint8)
k = 0
for i in range(0,heightm-8,8):
    for j in range(0,widthm-8,8):
        crop_tmp1 = mango1[i:i+8,j:j+8]
        flatm[k,0:64] = crop_tmp1.flatten()
        k = k + 1
fspaceM = pd.DataFrame(flatm)

fspaceM.to_csv('C:/Users/aishg/OneDrive/Desktop/Big data/assignment-1/mango.csv', index=False)


cc = round(((heightc-8)*(widthc-8))/64)
flatc = np.ones((cc, 65), np.uint8)
k = 0
for i in range(0,heightc-8,8):
    for j in range(0,widthc-8,8):
        crop_tmp2 = cant1[i:i+8,j:j+8]
        flatc[k,0:64] = crop_tmp2.flatten()
        k = k + 1
fspaceC = pd.DataFrame(flatc)


fspaceC.to_csv('C:/Users/aishg/OneDrive/Desktop/Big data/assignment-1/cant.csv', index=False)

rr = round(((heightr-8)*(widthr-8))/64)
flatr = np.ones((rr, 65), np.uint8)
flatr[:,64] = 2
k = 0
for i in range(0,heightr-8,8):
    for j in range(0,widthr-8,8):
        crop_tmp3 = rasp1[i:i+8,j:j+8]
        flatr[k,0:64] = crop_tmp3.flatten()
        k = k + 1
        
fspaceR = pd.DataFrame(flatr)  #panda object

fspaceR.to_csv('C:/Users/aishg/OneDrive/Desktop/Big data/assignment-1/rasp3.csv', index=False)

def window_blocking_overlap(image):
    windowsize_r = 8
    windowsize_c = 8
    fea_vector=[]
    for r in range(0,(image.shape[0] - windowsize_r)+1, 1):
        for c in range(0,(image.shape[1] - windowsize_c)+1, 1):
            window = image[r:r+windowsize_r,c:c+windowsize_c]
            window_array = (window.flatten())
            fea_vector.append(window_array)
    return fea_vector

np.savetxt("C:/Users/aishg/OneDrive/Desktop/Big data/assignment-1/Mango1_sliding.csv",window_blocking_overlap(mango), delimiter=',',)
np.savetxt("C:/Users/aishg/OneDrive/Desktop/Big data/assignment-1/cant1_sliding.csv",window_blocking_overlap(cant), delimiter=',')
np.savetxt("C:/Users/aishg/OneDrive/Desktop/Big data/assignment-1/rasp1_sliding.csv",window_blocking_overlap(rasp), delimiter=',')


sliding_mango1 = pd.read_csv('C:/Users/aishg/OneDrive/Desktop/Big data/assignment-1/Mango1_sliding.csv', header=None)
sliding_cant1 = pd.read_csv('C:/Users/aishg/OneDrive/Desktop/Big data/assignment-1/cant1_sliding.csv' , header=None)
sliding_rasp1 = pd.read_csv('C:/Users/aishg/OneDrive/Desktop/Big data/assignment-1/rasp1_sliding.csv' , header=None)


sliding_mango1.insert(64, "64",0)
sliding_mango1 = sliding_mango1.astype(int)
sliding_mango1.shape


sliding_cant1.insert(64, "64",1)
sliding_cant1 = sliding_cant1.astype(int)
sliding_cant1.shape

sliding_rasp1.insert(64, "64",2)
sliding_rasp1 = sliding_rasp1.astype(int)
sliding_rasp1.shape

fspaceM.describe()

import random as rand

Mango_scatter = pd.read_csv('C:/Users/aishg/OneDrive/Desktop/Big data/assignment-1/mango.csv')

x = Mango_scatter['55']
y = Mango_scatter['60']


plt.scatter(x,y, color= ('purple'))

Cant_scatter = pd.read_csv('C:/Users/aishg/OneDrive/Desktop/Big data/assignment-1/cant.csv')

x= Cant_scatter['55']
y= Cant_scatter['60']
plt.scatter(x, y, color = ('orange'))
plt.xlabel('X', labelpad=5)
plt.ylabel('Y', labelpad=5)

Rasp_scatter = pd.read_csv('C:/Users/aishg/OneDrive/Desktop/Big data/assignment-1/rasp3.csv')


plt.scatter(Rasp_scatter['55'], Rasp_scatter['60'], Rasp_scatter['57'], color= ('skyblue'))
plt.xlabel('X', labelpad=5)
plt.ylabel('Y', labelpad=5)


Mango_scatter = pd.read_csv('C:/Users/aishg/OneDrive/Desktop/Big data/assignment-1/mango.csv')
num_bins = 4

n, bins, patches = plt.hist(Mango_scatter, num_bins, facecolor='red', alpha=5)
plt.show()

Cant_scatter = pd.read_csv('C:/Users/aishg/OneDrive/Desktop/Big data/assignment-1/cant.csv')
num_bins = 4
n, bins, patches = plt.hist(Cant_scatter, num_bins, facecolor='yellow', alpha=5)
plt.show()

Rasp_scatter = pd.read_csv('C:/Users/aishg/OneDrive/Desktop/Big data/assignment-1/rasp3.csv')
num_bins = 4
n, bins, patches = plt.hist(Rasp_scatter, num_bins, facecolor='green', alpha=5)
plt.show()

framesMC = [fspaceM, fspaceC]
mgedMC = pd.concat(framesMC)

mgedMC.to_csv('C:/Users/aishg/OneDrive/Desktop/Big data/assignment-1/MC.csv', index=False)

indx = np.arange(len(mgedMC))
rndmgedMC = np.random.permutation(indx)
rndmgedMC =mgedMC.sample(frac=1).reset_index(drop=True)

rndmgedMC.to_csv('C:/Users/aishg/OneDrive/Desktop/Big data/assignment-1/PermutedMC.csv', index=False)

framesMCR = [fspaceM, fspaceC, fspaceR]
mgedMCR = pd.concat(framesMCR)
mgedMCR.to_csv('C:/Users/aishg/OneDrive/Desktop/Big data/assignment-1/MCR.csv', index=False)


framesMCR = [fspaceM, fspaceC, fspaceR]
mgedMCR = pd.concat(framesMCR)


indx = np.arange(len(mgedMCR))
rndmgedMCR = np.random.permutation(indx)
rndmgedMCR =mgedMCR.sample(frac=1).reset_index(drop=True)

rndmgedMCR.to_csv('C:/Users/aishg/OneDrive/Desktop/Big data/assignment-1/PermutedMCR.csv', index=False)


MC = pd.read_csv('C:/Users/aishg/OneDrive/Desktop/Big data/assignment-1/PermutedMC.csv')


x = MC['55']
y = MC['60']
label = MC['64']
color = ['Red', 'purple']

plt.scatter(x, y, c=label, cmap=matplotlib.colors.ListedColormap(color))

plt.xlabel("55")
plt.ylabel("60")


x = MC['44']
y = MC['46']
label = MC['64']
color = ['Red', 'purple']

ax = plt.axes(projection ="3d") 
ax.scatter3D(x, y, c=label, cmap=matplotlib.colors.ListedColormap(color))

ax.set_xlabel("x")
ax.set_ylabel("y")

MCR = pd.read_csv('C:/Users/aishg/OneDrive/Desktop/Big data/assignment-1/PermutedMCR.csv')


x = MCR['44']
y = MCR['60']
z = MCR['55']

label = MCR['64']
color = ['orange', 'maroon', 'purple']


plt.scatter(x, y, z, c=label, cmap=matplotlib.colors.ListedColormap(color))

plt.xlabel("x")
plt.ylabel("y")

MCR = pd.read_csv('C:/Users/aishg/OneDrive/Desktop/Big data/assignment-1/PermutedMCR.csv')


x = MCR['44']
y = MCR['60']
z = MCR['55']
label = MCR['64']
color = ['blue', 'red', 'green']

ax = plt.axes(projection ="3d") 

ax.scatter3D(x, y, z, c=label, cmap=matplotlib.colors.ListedColormap(color))

ax.set_xlabel('x') 
ax.set_ylabel('y') 
ax.set_zlabel('z') 



sliding_mango1 = pd.read_csv('C:/Users/aishg/OneDrive/Desktop/Big data/assignment-1/Mango1_sliding.csv', header=None)
sliding_mango1.insert(64, "64",0)
sliding_cant1 = pd.read_csv('C:/Users/aishg/OneDrive/Desktop/Big data/assignment-1/cant1_sliding.csv' , header=None)
sliding_cant1.insert(64, "64",1)
sliding_rasp1 = pd.read_csv('C:/Users/aishg/OneDrive/Desktop/Big data/assignment-1/rasp1_sliding.csv' , header=None)
sliding_rasp1.insert(64, "64",2)

framesSlideMC = [sliding_mango1, sliding_cant1]
mgedSlideMC = pd.concat(framesSlideMC)


indxs = np.arange(len(mgedSlideMC))
rndmgedSlideMC = np.random.permutation(indxs)
rndmgedSlideMC =mgedSlideMC.sample(frac=1).reset_index(drop=True)

rndmgedSlideMC.to_csv('C:/Users/aishg/OneDrive/Desktop/Big data/assignment-1/SlideingMC.csv', index=False)

framesSlideMCR = [sliding_mango1, sliding_cant1, sliding_rasp1]
mgedSlideMCR = pd.concat(framesSlideMCR)


indxsr = np.arange(len(mgedSlideMCR))
rndmgedSlideMCR = np.random.permutation(indxsr)
rndmgedSlideMCR =mgedSlideMCR.sample(frac=1).reset_index(drop=True)

rndmgedSlideMCR.to_csv('C:/Users/aishg/OneDrive/Desktop/Big data/assignment-1/SlideingMCR.csv', index=False)


# =============================================================================
# Assignment 2
# =============================================================================

nonMC = pd.read_csv("C:/Users/aishg/OneDrive/Desktop/Big data/assignment-1/PermutedMC.csv")
nonMCR = pd.read_csv("C:/Users/aishg/OneDrive/Desktop/Big data/assignment-1/PermutedMCR.csv")
slideMC = pd.read_csv("C:/Users/aishg/OneDrive/Desktop/Big data/assignment-1/SlideingMC.csv")
slideMCR = pd.read_csv("C:/Users/aishg/OneDrive/Desktop/Big data/assignment-1/SlideingMCR.csv")

nonMC_train, nonMC_test = train_test_split(nonMC, test_size=0.2, random_state = 44)
nonMC_train.to_csv('C:/Users/aishg/OneDrive/Desktop/Big data/assignment-1/nontrainMC.csv', index=False)
nonMC_test.to_csv('C:/Users/aishg/OneDrive/Desktop/Big data/assignment-1/nontestMC.csv', index=False)


nonMCR_train, nonMCR_test = train_test_split(nonMCR, test_size=0.2, random_state = 44)
nonMCR_train.to_csv('C:/Users/aishg/OneDrive/Desktop/Big data/assignment-1/nontrainMCR.csv', index=False)
nonMCR_test.to_csv('C:/Users/aishg/OneDrive/Desktop/Big data/assignment-1/nontestMCR.csv', index=False)

slideMC_train, slideMC_test = train_test_split(slideMC, test_size=0.2, random_state=44)
slideMC_train.to_csv('C:/Users/aishg/OneDrive/Desktop/Big data/assignment-1/slidetrainMC.csv', index=False)
slideMC_test.to_csv('C:/Users/aishg/OneDrive/Desktop/Big data/assignment-1/slidetestMC.csv', index=False)

slideMCR_train, slideMCR_test = train_test_split(slideMCR, test_size=0.2, random_state=44)
slideMCR_train.to_csv('C:/Users/aishg/OneDrive/Desktop/Big data/assignment-1/slidetrainMCR.csv', index=False)
slideMCR_test.to_csv('C:/Users/aishg/OneDrive/Desktop/Big data/assignment-1/slidetestMCR.csv', index=False)

plt.hist(nonMC_train['40'], bins=5, facecolor='red', alpha=5)
plt.hist(nonMC_train['55'], bins=5, facecolor='orange', alpha=5)
plt.show()
print('mean of 40:',nonMC_train['40'].mean(), 'Variance of 40:', nonMC_train['40'].var())
print('mean of 55:',nonMC_train['55'].mean(), 'Variance of 55:', nonMC_train['55'].var())


plt.hist(nonMC_test['40'], bins=5, facecolor='purple', alpha=5)
plt.hist(nonMC_test['55'], bins=5, facecolor='violet', alpha=5)
plt.show()
print('mean of 40:',nonMC_test['40'].mean(), 'Variance of 40:', nonMC_test['40'].var())
print('mean of 55:',nonMC_test['55'].mean(), 'Variance of 55:', nonMC_test['55'].var())


plt.hist(nonMCR_train['40'], bins=5, facecolor='orange', alpha=5)
plt.hist(nonMCR_train['55'], bins=5, facecolor='yellow', alpha=5)
plt.show()
print('mean of 40:',nonMCR_train['40'].mean(), 'Variance of 40:', nonMCR_train['40'].var())
print('mean of 55:',nonMCR_train['55'].mean(), 'Variance of 55:', nonMCR_train['55'].var())



plt.hist(nonMCR_test['40'], bins = 5, facecolor='purple', alpha=5)
plt.hist(nonMCR_test['55'], bins = 5, facecolor='violet', alpha=5)
plt.show()
print('mean of 40:',nonMCR_test['40'].mean(), 'Variance of 40:', nonMCR_test['40'].var())
print('mean of 55:',nonMCR_test['55'].mean(), 'Variance of 55:', nonMCR_test['55'].var())

plt.hist(slideMC_train['40'], bins = 5, facecolor='red', alpha=5)
plt.hist(slideMC_train['55'], bins = 5, facecolor='orange', alpha=5)
plt.show()
print('mean of 40:',slideMC_train['40'].mean(), 'Variance of 40:', slideMC_train['40'].var())
print('mean of 55:',slideMC_train['55'].mean(), 'Variance of 55:', slideMC_train['55'].var())

plt.hist(slideMC_test['40'], bins=5, facecolor='violet', alpha=5)
plt.hist(slideMC_test['55'], bins=5, facecolor='purple', alpha=5)
plt.show()
print('mean of 40:',slideMC_test['40'].mean(), 'Variance of 40:', slideMC_test['40'].var())
print('mean of 55:',slideMC_test['55'].mean(), 'Variance of 55:', slideMC_test['55'].var())


plt.hist(slideMCR_train['40'], bins=5, facecolor='red', alpha=5)
plt.hist(slideMCR_train['55'], bins=5, facecolor='orange', alpha=5)
plt.show()
print('mean of 40:',slideMC_train['40'].mean(), 'Variance of 40:', slideMC_train['40'].var())
print('mean of 55:',slideMC_train['55'].mean(), 'Variance of 55:', slideMC_train['55'].var())


plt.hist(slideMCR_test['40'], bins=5, facecolor='purple', alpha=5)
plt.hist(slideMCR_test['55'], bins=5, facecolor='violet', alpha=5)
plt.show()
print('mean of 40:',slideMCR_test['40'].mean(), 'Variance of 40:', slideMCR_test['40'].var())
print('mean of 55:',slideMCR_test['55'].mean(), 'Variance of 55:', slideMCR_test['55'].var())



color = ['Red', 'purple']
plt.scatter(nonMC_train['40'], nonMC_train['55'], c=nonMC_train['64'], cmap=matplotlib.colors.ListedColormap(color))


color = ['Red', 'purple']
plt.scatter(nonMC_test['40'], nonMC_test['55'], c=nonMC_test['64'], cmap=matplotlib.colors.ListedColormap(color))

color = ['Red', 'purple']
plt.scatter(nonMCR_train['40'], nonMCR_train['55'], c=nonMCR_train['64'], cmap=matplotlib.colors.ListedColormap(color))

color = ['Red', 'purple']
plt.scatter(nonMCR_test['40'], nonMCR_test['55'], c=nonMCR_test['64'], cmap=matplotlib.colors.ListedColormap(color))

color = ['Red', 'purple']
plt.scatter(slideMC_train['40'], slideMC_train['55'], c=slideMC_train['64'], cmap=matplotlib.colors.ListedColormap(color))


color = ['Red', 'purple']
plt.scatter(slideMC_test['40'], slideMC_test['55'], c=slideMC_test['64'], cmap=matplotlib.colors.ListedColormap(color))


color = ['Red', 'purple']
plt.scatter(slideMCR_train['40'], slideMCR_train['55'], c=slideMCR_train['64'], cmap=matplotlib.colors.ListedColormap(color))

color = ['Red', 'purple']
plt.scatter(slideMCR_test['40'], slideMCR_test['55'], c=slideMCR_test['64'], cmap=matplotlib.colors.ListedColormap(color))


y_nonMC = nonMC_train['64']
y_nonMC

nonMC_train.drop('64', axis=1, inplace=True)
x_nonMC= nonMC_train
x_nonMC

X1_nonMC = np.array(x_nonMC)
X2_nonMC = X1_nonMC.transpose()
XX_nonMC = np.matmul(X2_nonMC, X1_nonMC)

IX_nonMC = inv(XX_nonMC)
TX_nonMC = np.matmul(X1_nonMC, IX_nonMC)
Y1_nonMC = np.array(y_nonMC)
Y2_nonMC = Y1_nonMC.transpose()
A_nonMC = np.matmul(Y1_nonMC, TX_nonMC)
ZZ1_nonMC = np.matmul(X1_nonMC, A_nonMC)
ZZ2_nonMC = ZZ1_nonMC > ZZ1_nonMC.mean()
yhat_nonMC = ZZ2_nonMC.astype(int)
CC_nonMC = confusion_matrix(y_nonMC, yhat_nonMC)
TN_nonMC = CC_nonMC[0,0]
FP_nonMC = CC_nonMC[0,1]
FN_nonMC = CC_nonMC[1,0]
TP_nonMC = CC_nonMC[1,1]
FPFN_nonMC = FP_nonMC+FN_nonMC
TPTN_nonMC = TP_nonMC+TN_nonMC
Accuracy = 1/(1+(FPFN_nonMC/TPTN_nonMC))
print("Our_Accuracy_Score for non overlapping MC data set:",Accuracy)

Precision = 1/(1+(FP_nonMC/TP_nonMC))
print("Our_Precision_Score for non overlapping MC data set::",Precision)

Sensitivity = 1/(1+(FN_nonMC/TP_nonMC))
print("Our_Sensitivity_Score for non overlapping MC data set::",Sensitivity)

Specificity = 1/(1+(FP_nonMC/TN_nonMC))
print("Our_Specificity_Score for non overlapping MC data set::",Specificity)



y_nonMCR = nonMCR_train['64']
y_nonMCR

nonMCR_train.drop('64', axis=1, inplace=True)
x_nonMCR= nonMCR_train
x_nonMCR

X1_nonMCR = np.array(x_nonMCR)
X2_nonMCR= X1_nonMCR.transpose()
XX_nonMCR = np.matmul(X2_nonMCR, X1_nonMCR)

IX_nonMCR = inv(XX_nonMCR)
TX_nonMCR = np.matmul(X1_nonMCR, IX_nonMCR)
Y1_nonMCR = np.array(y_nonMCR)
Y2_nonMCR = Y1_nonMCR.transpose()
A_nonMCR = np.matmul(Y1_nonMCR, TX_nonMCR)
ZZ1_nonMCR = np.matmul(X1_nonMCR, A_nonMCR)
ZZ2_nonMCR = ZZ1_nonMCR > ZZ1_nonMCR.mean()
yhat_nonMCR = ZZ2_nonMCR.astype(int)
CC_nonMCR = confusion_matrix(y_nonMCR, yhat_nonMCR)
TN_nonMCR = CC_nonMCR[0,0]
FP_nonMCR= CC_nonMCR[0,1]
FN_nonMCR = CC_nonMCR[1,0]
TP_nonMCR = CC_nonMCR[1,1]
FPFN_nonMCR = FP_nonMCR+FN_nonMCR
TPTN_nonMCR = TP_nonMCR+TN_nonMCR
Accuracy = 1/(1+(FPFN_nonMCR/TPTN_nonMCR))
print("Our_Accuracy_Score for non overlapping MCR data set:",Accuracy)

Precision = 1/(1+(FP_nonMCR/TP_nonMCR))
print("Our_Precision_Score for non overlapping MCR data set::",Precision)

Sensitivity = 1/(1+(FN_nonMCR/TP_nonMCR))
print("Our_Sensitivity_Score for non overlapping MCR data set::",Sensitivity)

Specificity = 1/(1+(FP_nonMCR/TN_nonMCR))
print("Our_Specificity_Score for non overlapping MCR data set::",Specificity)



y_slideMC = slideMC_train['64']
y_slideMC

slideMC_train.drop('64', axis=1, inplace=True)
x_slideMC= slideMC_train
x_slideMC

X1_slideMC = np.array(x_slideMC)
X2_slideMC= X1_slideMC.transpose()
XX_slideMC = np.matmul(X2_slideMC, X1_slideMC)

IX_slideMC = inv(XX_slideMC)
TX_slideMC = np.matmul(X1_slideMC, IX_slideMC)
Y1_slideMC = np.array(y_slideMC)
Y2_slideMC = Y1_slideMC.transpose()
A_slideMC = np.matmul(Y1_slideMC, TX_slideMC)
ZZ1_slideMC = np.matmul(X1_slideMC, A_slideMC)
ZZ2_slideMC = ZZ1_slideMC > ZZ1_slideMC.mean()
yhat_slideMC = ZZ2_slideMC.astype(int)
CC_slideMC = confusion_matrix(y_slideMC, yhat_slideMC)
TN_slideMC = CC_slideMC[0,0]
FP_slideMC= CC_slideMC[0,1]
FN_slideMC = CC_slideMC[1,0]
TP_slideMC = CC_slideMC[1,1]
FPFN_slideMC = FP_slideMC+FN_slideMC
TPTN_slideMC = TP_slideMC+TN_slideMC
Accuracy = 1/(1+(FPFN_slideMC/TPTN_slideMC))
print("Our_Accuracy_Score for overlapping MC data set:",Accuracy)

Precision = 1/(1+(FP_slideMC/TP_slideMC))
print("Our_Precision_Score for overlapping MC data set::",Precision)

Sensitivity = 1/(1+(FN_slideMC/TP_slideMC))
print("Our_Sensitivity_Score for overlapping MC data set::",Sensitivity)

Specificity = 1/(1+(FP_slideMC/TN_slideMC))
print("Our_Specificity_Score for overlapping MC data set::",Specificity)





y_slideMCR = slideMCR_train['64']
y_slideMCR

slideMCR_train.drop('64', axis=1, inplace=True)
x_slideMCR= slideMCR_train
x_slideMCR

X1_slideMCR = np.array(x_slideMCR)
X2_slideMCR= X1_slideMCR.transpose()
XX_slideMCR = np.matmul(X2_slideMCR, X1_slideMCR)

IX_slideMCR = inv(XX_slideMCR)
TX_slideMCR = np.matmul(X1_slideMCR, IX_slideMCR)
Y1_slideMCR = np.array(y_slideMCR)
Y2_slideMCR = Y1_slideMCR.transpose()
A_slideMCR = np.matmul(Y1_slideMCR, TX_slideMCR)
ZZ1_slideMCR = np.matmul(X1_slideMCR, A_slideMCR)
ZZ2_slideMCR = ZZ1_slideMCR > ZZ1_slideMCR.mean()
yhat_slideMCR = ZZ2_slideMCR.astype(int)
CC_slideMCR = confusion_matrix(y_slideMCR, yhat_slideMCR)
TN_slideMCR = CC_slideMCR[0,0]
FP_slideMCR= CC_slideMCR[0,1]
FN_slideMCR = CC_slideMCR[1,0]
TP_slideMCR = CC_slideMCR[1,1]
FPFN_slideMCR = FP_slideMCR+FN_slideMCR
TPTN_slideMCR = TP_slideMCR+TN_slideMCR
Accuracy = 1/(1+(FPFN_slideMCR/TPTN_slideMCR))
print("Our_Accuracy_Score for overlapping MCR data set:",Accuracy)

Precision = 1/(1+(FP_slideMCR/TP_slideMCR))
print("Our_Precision_Score for overlapping MCR data set::",Precision)

Sensitivity = 1/(1+(FN_slideMCR/TP_slideMCR))
print("Our_Sensitivity_Score for overlapping MCR data set::",Sensitivity)

Specificity = 1/(1+(FP_slideMCR/TN_slideMCR))
print("Our_Specificity_Score for overlapping MCR data set::",Specificity)


from sklearn import metrics
print("BuiltIn_Accuracy:",metrics.accuracy_score(y_nonMC, yhat_nonMC))
print("BuiltIn_Precision:",metrics.precision_score(y_nonMC, yhat_nonMC))
print("BuiltIn_Sensitivity (recall):",metrics.recall_score(y_nonMC, yhat_nonMC))

print("BuiltIn_Accuracy:",metrics.accuracy_score(y_nonMCR, yhat_nonMCR))
print("BuiltIn_Sensitivity (recall):",metrics.recall_score(y_nonMCR, yhat_nonMCR, average = 'weighted'))
print('\nClassification Report of MCR dataset\n')
print(classification_report(y_nonMCR, yhat_nonMCR, target_names=['Class 1', 'Class 2', 'Class 3']))


print("BuiltIn_Accuracy:",metrics.accuracy_score(y_slideMC, yhat_slideMC))
print("BuiltIn_Precision:",metrics.precision_score(y_slideMC, yhat_slideMC))
print("BuiltIn_Sensitivity (recall):",metrics.recall_score(y_slideMC, yhat_slideMC))

print("BuiltIn_Accuracy:",metrics.accuracy_score(y_slideMCR, yhat_slideMCR))
print("BuiltIn_Sensitivity (recall):",metrics.recall_score(y_slideMCR, yhat_slideMCR, average = 'weighted'))

print('\nClassification Report of Sliding window MCR dataset\n')
print(classification_report(y_slideMCR, yhat_slideMCR, target_names=['Class 1', 'Class 2', 'Class 3']))

import pandas as pd
import numpy as np
#from numpy.linalg import inv
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
#from sklearn import tree
from sklearn.ensemble import RandomForestClassifier

slide = pd.read_csv('C:/Users/aishg/OneDrive/Desktop/Big data/assignment-1/PermutedMC.csv', header = None)



y = slide[64]
y

x = slide.drop(columns=[64])
x

x1 = np.array(x)
y1 = np.array(y)

x1_train, x1_test = train_test_split(x1, test_size=0.2, random_state = 0)

y1_train, y1_test = train_test_split(y1, test_size=0.2, random_state = 0)

rF = RandomForestClassifier(random_state=0, n_estimators=500, oob_score=True, n_jobs=-1)
model = rF.fit(x1_train,y1_train)

importance = model.feature_importances_
indices = importance.argsort()[::-1]

std = np.std([model.feature_importances_ for model in rF.estimators_], axis=0)

for f in range(x.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importance[indices[f]]))
plt.bar(range(x.shape[1]), importance[indices], color="r", yerr=std[indices], align="center")
plt.xticks(range(x.shape[1]), indices+1, rotation=90)
plt.show()

oob_error = 1 - rF.oob_score_

yhat_test = rF.predict(x1_test)
CC_test = confusion_matrix(y1_test, yhat_test)

TN = CC_test[1,1]
FP = CC_test[1,0]
FN = CC_test[0,1]
TP = CC_test[0,0]
FPFN = FP+FN
TPTN = TP+TN
Accuracy = 1/(1+(FPFN/TPTN))
print("Our_Accuracy_Score:",Accuracy)

Precision = 1/(1+(FP/TP))
print("Our_Precision_Score:",Precision)

Sensitivity = 1/(1+(FN/TP))
print("Our_Sensitivity_Score:",Sensitivity)

Specificity = 1/(1+(FP/TN))
print("Our_Specificity_Score:",Specificity)


slideMCR = pd.read_csv('C:/Users/aishg/OneDrive/Desktop/Big data/assignment-1/PermutedMCR.csv', header = None)



y = slideMCR[64]
y

x = slideMCR.drop(columns=[64])
x

x1 = np.array(x)
y1 = np.array(y)

x1_train, x1_test = train_test_split(x1, test_size=0.2, random_state = 0)

y1_train, y1_test = train_test_split(y1, test_size=0.2, random_state = 0)

rF = RandomForestClassifier(random_state=0, n_estimators=500, oob_score=True, n_jobs=-1)
model = rF.fit(x1_train,y1_train)

importance = model.feature_importances_
indices = importance.argsort()[::-1]

std = np.std([model.feature_importances_ for model in rF.estimators_], axis=0)

for f in range(x.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importance[indices[f]]))
plt.bar(range(x.shape[1]), importance[indices], color="r", yerr=std[indices], align="center")
plt.xticks(range(x.shape[1]), indices+1, rotation=90)
plt.show()

oob_error = 1 - rF.oob_score_

yhat_test = rF.predict(x1_test)
CC_test = confusion_matrix(y1_test, yhat_test)

TN = CC_test[1,1]
FP = CC_test[1,0]
FN = CC_test[0,1]
TP = CC_test[0,0]
FPFN = FP+FN
TPTN = TP+TN
Accuracy = 1/(1+(FPFN/TPTN))
print("Our_Accuracy_Score:",Accuracy)

Precision = 1/(1+(FP/TP))
print("Our_Precision_Score:",Precision)

Sensitivity = 1/(1+(FN/TP))
print("Our_Sensitivity_Score:",Sensitivity)

Specificity = 1/(1+(FP/TN))
print("Our_Specificity_Score:",Specificity)


slide = pd.read_csv('C:/Users/aishg/OneDrive/Desktop/Big data/assignment-1/SlideingMC.csv', header = None)



y = slide[64]
y

x = slide.drop(columns=[64])
x

x1 = np.array(x)
y1 = np.array(y)

x1_train, x1_test = train_test_split(x1, test_size=0.2, random_state = 0)

y1_train, y1_test = train_test_split(y1, test_size=0.2, random_state = 0)

rF = RandomForestClassifier(random_state=0, n_estimators=500, oob_score=True, n_jobs=-1)
model = rF.fit(x1_train,y1_train)

importance = model.feature_importances_
indices = importance.argsort()[::-1]

std = np.std([model.feature_importances_ for model in rF.estimators_], axis=0)

for f in range(x.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importance[indices[f]]))
plt.bar(range(x.shape[1]), importance[indices], color="r", yerr=std[indices], align="center")
plt.xticks(range(x.shape[1]), indices+1, rotation=90)
plt.show()

oob_error = 1 - rF.oob_score_

yhat_test = rF.predict(x1_test)
CC_test = confusion_matrix(y1_test, yhat_test)

TN = CC_test[1,1]
FP = CC_test[1,0]
FN = CC_test[0,1]
TP = CC_test[0,0]
FPFN = FP+FN
TPTN = TP+TN
Accuracy = 1/(1+(FPFN/TPTN))
print("Our_Accuracy_Score:",Accuracy)

Precision = 1/(1+(FP/TP))
print("Our_Precision_Score:",Precision)

Sensitivity = 1/(1+(FN/TP))
print("Our_Sensitivity_Score:",Sensitivity)

Specificity = 1/(1+(FP/TN))
print("Our_Specificity_Score:",Specificity)


slide = pd.read_csv('C:/Users/aishg/OneDrive/Desktop/Big data/assignment-1/SlideingMCR.csv', header = None)



y = slide[64]
y

x = slide.drop(columns=[64])
x

x1 = np.array(X)
y1 = np.array(y)

x1_train, x1_test = train_test_split(x1, test_size=0.2, random_state = 0)

y1_train, y1_test = train_test_split(y1, test_size=0.2, random_state = 0)

rF = RandomForestClassifier(random_state=0, n_estimators=500, oob_score=True, n_jobs=-1)
model = rF.fit(x1_train,y1_train)

importance = model.feature_importances_
indices = importance.argsort()[::-1]

std = np.std([model.feature_importances_ for model in rF.estimators_], axis=0)

for f in range(X.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importance[indices[f]]))
plt.bar(range(X.shape[1]), importance[indices], color="r", yerr=std[indices], align="center")
plt.xticks(range(X.shape[1]), indices+1, rotation=90)
plt.show()

oob_error = 1 - rF.oob_score_

yhat_test = rF.predict(x1_test)
CC_test = confusion_matrix(y1_test, yhat_test)

TN = CC_test[1,1]
FP = CC_test[1,0]
FN = CC_test[0,1]
TP = CC_test[0,0]
FPFN = FP+FN
TPTN = TP+TN
Accuracy = 1/(1+(FPFN/TPTN))
print("Our_Accuracy_Score:",Accuracy)

Precision = 1/(1+(FP/TP))
print("Our_Precision_Score:",Precision)

Sensitivity = 1/(1+(FN/TP))
print("Our_Sensitivity_Score:",Sensitivity)

Specificity = 1/(1+(FP/TN))
print("Our_Specificity_Score:",Specificity)


