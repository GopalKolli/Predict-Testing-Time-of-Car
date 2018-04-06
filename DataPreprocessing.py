import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn import tree
from PreProcessUtils import Preprocess
from sklearn.utils import shuffle
import math


uselessFeatures = ['X290', 'X293', 'X297', 'X107', 'X268', 'X11', 'X233', 'X235', 'X289', 'X347', 'X330', 'X93','X103','X114','X117','X123','X129','X139','X140','X146','X168','X18','X181','X182','X186','X206','X210','X24','X32','X340','X345','X359','X366','X40','X92']
uselessFeatures = set(uselessFeatures)

Dataset = pd.read_csv('train.csv',sep=',')
DatasetFrame = pd.DataFrame(Dataset)

#Removing features whose linear impact is zero
for f in uselessFeatures:
    del DatasetFrame[f]

TestDataset = pd.read_csv('test.csv',sep=',')
TestDatasetFrame = pd.DataFrame(TestDataset)

#Removing features whose linear impact is zero
for f in uselessFeatures:
    del TestDatasetFrame[f]

TestDatasetFrame['y'] = [100] * len(TestDatasetFrame)
TestDatasetFrame.index = range(len(DatasetFrame),len(DatasetFrame)+len(TestDatasetFrame))

ls = list(TestDatasetFrame)
print(ls)
correctorder = [ls[0]]
correctorder.append(ls[-1])
correctorder.extend(ls[1:-1])
print(correctorder)
TestDatasetFrame = TestDatasetFrame[correctorder]

print(list(TestDatasetFrame))

frames = [DatasetFrame, TestDatasetFrame]
TotalFrame = pd.concat(frames)

categorical = ['X0','X1','X2','X3','X4','X5','X6','X8']
for i in categorical:
    factorizedvars, originalvars = pd.factorize(TotalFrame[i])
    TotalFrame[i] = factorizedvars

DatasetMatrix = TotalFrame.as_matrix()

#Removing Outlier
DatasetMatrix[883,1:] = DatasetMatrix[800,1:]


Y = DatasetMatrix[:,1]
X = DatasetMatrix[:,2:]
IDs = np.asarray(DatasetMatrix[:,0])
IDs = IDs.astype(np.uint32)

print("The Y DataSet : ")
print(Y)
print(len(Y))
print("------------------------------")
print("The X DataSet : ")
print(X)
print(len(X))
print("------------------------------")


uniquevalues = 0
for i in range(0,8):
    uniquevalues += len(set(X[:,i]))
print("-----------UniqueValues : " + str(uniquevalues))


print("X Befor One Hot : ")
print(X)
print(X.shape)

enc = OneHotEncoder(sparse=False, categorical_features=[0,1,2,3,4,5,6,7])
X = enc.fit_transform(X=X)

print("X After One Hot : ")
print(X)
print(X.shape)

prep = Preprocess()

#PCA on X
#X = prep.doPCA(X,240)
prep.corrAnalysis(X,Y,10)

"""
XTrain = X[0:4209,:]
YTrain = Y[0:4209]
XTest = X[4209:,:]
IDsToPredict = IDs[4209:]

binrange = 4
YTrainBinned = np.zeros_like(YTrain)
for i in range(0,len(YTrain)):
    YTrainBinned[i] = int(YTrain[i]/binrange)

#SeleckkBest on X
XTrain, XTest = prep.selectKBestFeatures(XTrain, YTrainBinned, XTest, 125, "chi2")



kfold = 8
samplesize = int(4208/kfold)
totalrmse = 0
XTrain, YTrainBinned = shuffle(XTrain, YTrainBinned, random_state=4)
for k in range(0,kfold):
    XFortTest = np.zeros(shape=(samplesize,XTrain.shape[1]))
    XFortTest = XTrain[samplesize*k:samplesize*(k+1),:]

    YBinnedForTest = np.zeros(shape=(samplesize))
    YBinnedForTest = YTrainBinned[samplesize*k:samplesize*(k+1)]

    XForTrain = np.zeros(shape=(len(XTrain)-len(XFortTest),XTrain.shape[1]))
    XForTrain[:samplesize*k,:] = XTrain[:samplesize*k,:]
    XForTrain[samplesize*k:,:] = XTrain[samplesize*(k+1):,:]

    YBinnedForTrain = np.zeros(shape=(len(XTrain)-len(XFortTest)))
    YBinnedForTrain[:samplesize*k] = YTrainBinned[:samplesize*k]
    YBinnedForTrain[samplesize*k:] = YTrainBinned[samplesize*(k+1):]

    totalrmse += prep.DecisionTreeModel(48,10,XForTrain,YBinnedForTrain,XFortTest,YBinnedForTest)
    #totalrmse += prep.RandomForestClassModel(40, 'gini', 120, XForTrain, YBinnedForTrain, XFortTest, YBinnedForTest)
print("The mean RMSE over all folds : " + str(totalrmse/kfold))

"""
"""
YOut = (YPred*binrange)+5
OutPutFrame = pd.DataFrame()
OutPutFrame['id'] = IDsToPredict
OutPutFrame['y'] = YOut
OutPutFrame.to_csv('Sub.csv',index=False)
"""

"""
poly = PolynomialFeatures(degree=1)
X_ = poly.fit_transform(X)

lg = LinearRegression(fit_intercept=True,normalize=True,copy_X=True, n_jobs=2)
lg.fit(X_[0:4209,:], Y[0:4209])

YPred = lg.predict(X_[4209:,:])

print(YPred)
"""
