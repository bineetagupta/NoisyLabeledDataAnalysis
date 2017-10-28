import numpy as np
def my_custom_loss_func(ground_truth, predictions):
    diff = np.abs(ground_truth - predictions).max()
    return np.log(1 + diff)

#http://scikit-learn.org/stable/modules/svm.html
trainFile = open("/Users/jaydeep/jaydeep_workstation/ASU/Fall2017/CSE569_FSL/project/trainData.csv", "r")
testFile = open("/Users/jaydeep/jaydeep_workstation/ASU/Fall2017/CSE569_FSL/project/testData.csv", "r")


train_lines = trainFile.read().split("\n") # "\r\n" if needed
trainX = [];
trainY = [];

for line in train_lines:
    if line != "": # add other needed checks to skip titles
        cols = line.split(",")
        trainX.append([float(cols[0]),float(cols[1])]);
        trainY.append(int(cols[2]));
        
test_lines = testFile.read().split("\n") # "\r\n" if needed       
testX = [];
testY = [];

for line in test_lines:
    if line != "": # add other needed checks to skip titles
        cols = line.split(",")
        testX.append([float(cols[0]),float(cols[1])]);
        testY.append(int(cols[2]));

from sklearn import svm
from sklearn.metrics import make_scorer
loss  = make_scorer(my_custom_loss_func, greater_is_better=False)
score = make_scorer(my_custom_loss_func, greater_is_better=True)
linear_svc = svm.LinearSVC();
linear_svc.fit(trainX, trainY);
l = loss(linear_svc,trainX, trainY);
print(l);
predictions = linear_svc.predict(testX);
from sklearn.metrics import confusion_matrix
print(confusion_matrix(testY, predictions));
from sklearn.metrics import precision_score,recall_score,f1_score,accuracy_score
print(precision_score(testY, predictions, average='micro'));
print(recall_score(testY, predictions, average='micro'));
print(f1_score(testY, predictions, average='micro'));
print(accuracy_score(testY, predictions));