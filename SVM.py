from sklearn import svm
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import hinge_loss
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score


def custom_loss_func(ground_truth, predictions, rho_pos, rho_neg): 
    ground_truth = np.array(ground_truth);
    predictions = np.array(predictions)
    
    neg_indices = [i for i,x in enumerate(ground_truth) if x == -1]
    neg_truth_val = ground_truth[neg_indices]
    neg_pred_val = predictions[neg_indices]
    
    pos_indices = [i for i,x in enumerate(ground_truth) if x == 1]
    pos_truth_val = ground_truth[pos_indices]
    pos_pred_val = predictions[pos_indices]
    
    neg_hingeLoss = hinge_loss(neg_truth_val, neg_pred_val)
    pos_hingeLoss = hinge_loss(pos_truth_val, pos_pred_val)
    
    cust_loss = (1-rho_neg)*pos_hingeLoss - rho_pos*neg_hingeLoss
    return cust_loss

#http://scikit-learn.org/stable/modules/svm.html
rho_pos = 0.1
rho_neg = 0.1
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


linear_svc = svm.LinearSVC();
linear_svc.fit(trainX, trainY);

# trainingset accuracy
train_predictions = linear_svc.predict(trainX);
train_hinge_loss = hinge_loss(trainY, train_predictions);
trian_cust_acc = custom_loss_func(trainY, train_predictions, rho_pos, rho_neg);
print("Hinge Train Loss:- ", train_hinge_loss);
print("Custom Train Loss:- ", trian_cust_acc);

# testingset accuracy
test_predictions = linear_svc.predict(testX);
test_hinge_loss = hinge_loss(testY, test_predictions);
test_cust_acc = custom_loss_func(testY, test_predictions, rho_pos, rho_neg);
print("Hinge Test Loss:- ", test_hinge_loss);
print("Custom Test Loss:- ", test_cust_acc);

# Evaluation matrix
print("Confusion matrix on test data:- ");
print(confusion_matrix(testY, test_predictions));
print("Precision Test Score:- ",precision_score(testY, test_predictions, average='micro'));
print("Recall Test Score:- ",recall_score(testY, test_predictions, average='micro'));
print("F1 Test Score:- ",f1_score(testY, test_predictions, average='micro'));
print("Accuracy Test Score:- ",accuracy_score(testY, test_predictions));