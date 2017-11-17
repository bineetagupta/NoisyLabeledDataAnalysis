from sklearn import svm
import numpy as np
from sklearn.metrics import hinge_loss,zero_one_loss
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score


def custom_loss_func(ground_truth, prediction, rho_pos, rho_neg): 

    hingeLoss_1 = hinge_loss([ground_truth], [prediction])
    hingeLoss_2 = hinge_loss([ground_truth], [-prediction])
    if(prediction == 1):
        cust_loss = ((1-rho_neg)*hingeLoss_1 - rho_pos*hingeLoss_2)/(1- rho_pos - rho_neg)
    else:
        cust_loss = ((1-rho_pos)*hingeLoss_1 - rho_neg*hingeLoss_2)/(1- rho_pos - rho_neg)
    
    return cust_loss


def calSVM(rho_pos_val, rho_neg_val, trainFileNm, testFileNm):
    #http://scikit-learn.org/stable/modules/svm.html
    rho_pos = rho_pos_val
    rho_neg = rho_neg_val
    trainFile = open(trainFileNm, "r")
    testFile = open(testFileNm, "r")
    
    
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
    
    
    # testingset accuracy
    test_predictions = linear_svc.predict(testX);
    test_hinge_loss = hinge_loss(testY, test_predictions);
    test_cust_acc = 0;
    for i in range(0, len(testY)):
        test_cust_acc += custom_loss_func(testY[i], test_predictions[i], rho_pos, rho_neg);
        
    mean_test_cust_acc = np.mean(test_cust_acc)
    
#     print("Hinge Test Loss:- ", test_hinge_loss)
#     print("Custom Test Loss:- ", mean_test_cust_acc)
    
    # Evaluation matrix
#     print("Confusion matrix on test data:- ")
#     print(confusion_matrix(testY, test_predictions))
    precision_val = precision_score(testY, test_predictions, average='micro')
    recall_val = recall_score(testY, test_predictions, average='micro')
    f1_val = f1_score(testY, test_predictions, average='micro')
    acc_val = accuracy_score(testY, test_predictions)
    
    return test_hinge_loss, mean_test_cust_acc, precision_val, recall_val, f1_val, acc_val