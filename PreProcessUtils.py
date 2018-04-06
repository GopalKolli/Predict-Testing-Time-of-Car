from sklearn.decomposition import PCA
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import mutual_info_classif
from sklearn import tree
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import math
import matplotlib.pyplot as plt

class Preprocess:
    def doPCA(self, X, number_of_components):
        print("Before PCA Shape : " + str(X.shape))
        sklearn_pca = PCA(n_components=number_of_components)
        X = sklearn_pca.fit_transform(X)
        print("After PCA Shape : " + str(X.shape))
        return X

    def selectKBestFeatures(self,XTrain,YTrainBinned,XTest,k_val,method):
        # SeleckkBest on X
        DummyXTrain = XTrain
        DummyYTrainBinned = YTrainBinned
        DummyXTrain = DummyXTrain.astype(np.float32)
        DummyYTrainBinned = DummyYTrainBinned.astype(np.float32)

        if method == "chi2":
            selector = SelectKBest(chi2, k=k_val)
        elif method == "f_clas":
            selector = SelectKBest(f_classif, k=k_val)
        elif method == "mic":
            selector = SelectKBest(mutual_info_classif, k=k_val)

        DummyXTrain = selector.fit_transform(DummyXTrain, DummyYTrainBinned)
        selectedfeaturesindicesmask = selector.get_support()
        XTrain = XTrain[:, selectedfeaturesindicesmask]
        XTest = XTest[:, selectedfeaturesindicesmask]
        return XTrain, XTest

    def corrAnalysis(self,X,Y,number):
        print(X.shape)
        newY = np.zeros(shape=(len(Y),1))
        newY[:,0] = Y[:]
        print(newY.shape)
        Mat = np.concatenate((X, newY), axis=1)
        DF = pd.DataFrame(data=Mat)
        corrMat = DF.corr(method='pearson')
        print(corrMat.shape)
        plt.plot(corrMat)
        plt.plot(corrMat)
        plt.show()

    def DecisionTreeModel(self, maxtreedepth, minleaves, XTrain, YTrain, XTest, YOriginal):
        clf = tree.DecisionTreeClassifier(max_depth=maxtreedepth, min_samples_leaf=minleaves)
        clf = clf.fit(XTrain, YTrain)

        YPred = clf.predict(XTest)

        print("Original Values : ")
        print(YOriginal)
        print("predicted probabilities : ")
        print(YPred)

        diff = np.subtract(YOriginal, YPred)
        diffsquare = np.multiply(diff, diff)
        diffsqsum = sum(diffsquare)
        diffsqsummean = diffsqsum / len(YPred)
        rmse = math.sqrt(diffsqsummean)
        print("RMSE : " + str(rmse))
        return rmse

    def RandomForestClassModel(self, numberoftrees, cirteria, maxdepth, XTrain, YTrain, XTest, YOriginal):
        clf = RandomForestClassifier(n_estimators= numberoftrees, criterion= cirteria, max_depth= maxdepth)
        clf = clf.fit(XTrain, YTrain)

        YPred = clf.predict(XTest)

        print("Original Values : ")
        print(YOriginal)
        print("predicted probabilities : ")
        print(YPred)

        diff = np.subtract(YOriginal, YPred)
        diffsquare = np.multiply(diff, diff)
        diffsqsum = sum(diffsquare)
        diffsqsummean = diffsqsum / len(YPred)
        rmse = math.sqrt(diffsqsummean)
        print("RMSE : " + str(rmse))
        return rmse