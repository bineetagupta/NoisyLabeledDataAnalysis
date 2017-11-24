import numpy as np
import random


class crtTrainTestCls():
    def crtTrainTest(self,trainFlNm, testFlNm):
        #Opening training file which is flipped by rho_pos and rho_neg.
        trainFile = open(trainFlNm, "r")
        #Opening testing file which is not flipped.
        testFile = open(testFlNm, "r")
        
        #populating training trainX(features) and trainY(labels)
        
        train_lines = trainFile.read().split("\n") # "\r\n" if needed
#         random.shuffle(train_lines,random.random)
        trainX = [];
        trainY = [];
        
        for line in train_lines:
            if line != "": # add other needed checks to skip titles
                cols = line.split(",")
                trainX.append([float(cols[0]),float(cols[1])]);
                trainY.append(int(cols[2]));
        
         
        #populating training testX(features) and testY(labels)       
        test_lines = testFile.read().split("\n") # "\r\n" if needed 
#         random.shuffle(test_lines,random.random)   
        testX = [];
        testY = [];
        
        for line in test_lines:
            if line != "": # add other needed checks to skip titles
                cols = line.split(",")
                testX.append([float(cols[0]),float(cols[1])]);
                testY.append(int(cols[2]));
                      
        trainX = np.asarray(trainX, dtype=np.float64)
        trainY = np.asarray(trainY, dtype=np.int64)
        testX = np.asarray(testX, dtype=np.float64)
        testY = np.asarray(testY, dtype=np.int64)
        
        return trainX,trainY,testX,testY