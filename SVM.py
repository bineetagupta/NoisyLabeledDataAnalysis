from sklearn import svm
import numpy as np
from sklearn.metrics import hinge_loss,zero_one_loss, log_loss
from sklearn.metrics.scorer import make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score

class noisyLblSVM(object):

    def __init__(self):
        self._rho_pos = 0
        self._rho_neg = 0
        
    @property
    def rho_pos(self):
        print("getter of rho_pos called")
        return self._rho_pos

    @rho_pos.setter
    def rho_pos(self, value):
        print("setter of rho_pos called")
        self._rho_pos = value

    @property
    def rho_neg(self):
        print("getter of rho_neg called")
        return self._rho_neg

    @rho_neg.setter
    def rho_neg(self, value):
        print("setter of rho_neg called")
        self._rho_neg = value

    def custom_loss_func(self,ground_truth, prediction): 
#         print('custom', '-------------------')  
        #˜l(t, y) = ( (1 − ρ(−y))l(t, y) − ρ(y)l(t,−y) ) / ( 1 - ρ(−y) - ρ(y))
        cust_loss = 0
#         labels = np.array([-1, 1])
        #rho_pos is ρ(y) and rho_neg is ρ(−y)  
        for i in range(0, len(ground_truth)):
            #Here l(t, y) is considered as SKLearn inbuilt hinge loss
#             print([ground_truth[i]])
#             print([prediction[i]])
            hingeLoss_1 = zero_one_loss([ground_truth[i]], [prediction[i]]) #l(t, y)
            hingeLoss_2 = zero_one_loss([ground_truth[i]], [-prediction[i]]) #l(t,−y)
             
            if(prediction[i] == 1):
#                 print([ground_truth[i]], [prediction[i]], ((1-self._rho_neg)*hingeLoss_1 - self._rho_pos*hingeLoss_2)/(1- self._rho_pos - self._rho_neg))
                cust_loss += ((1-self._rho_neg)*hingeLoss_1 - self._rho_pos*hingeLoss_2)/(1- self._rho_pos - self._rho_neg)
            else:
#                 print([ground_truth[i]], [prediction[i]], ((1-self._rho_pos)*hingeLoss_1 - self._rho_neg*hingeLoss_2)/(1- self._rho_pos - self._rho_neg))
                cust_loss += ((1-self._rho_pos)*hingeLoss_1 - self._rho_neg*hingeLoss_2)/(1- self._rho_pos - self._rho_neg)
            
#             print(cust_loss)
        return cust_loss
    
    def org_loss_func(self,ground_truth, prediction):
#         print('original', len(ground_truth))
        #print('------------------')
        org_loss = 0
        for i in range(0, len(ground_truth)):
            #print([ground_truth[i]], [prediction[i]], zero_one_loss([ground_truth[i]], [prediction[i]]))
            #print('org_loss', org_loss)
            org_loss += zero_one_loss([ground_truth[i]], [prediction[i]])
            
        return org_loss
        
    
    def calSVM(self,rho_pos_val, rho_neg_val, trainFileNm, testFileNm):
        #http://scikit-learn.org/stable/modules/svm.html
#         labels = np.array([-1, 1])
        #print(zero_one_loss([1], [1]))
        #print(zero_one_loss([1], [-1]))
        #print(zero_one_loss([-1], [1]))
        #print(zero_one_loss([-1], [-1]))
        #assigning rho(+y) and rho(-y) 
        self._rho_pos = rho_pos_val
        self._rho_neg = rho_neg_val
        #Opening training file which is flipped by rho_pos and rho_neg.
        trainFile = open(trainFileNm, "r")
        #Opening testing file which is not flipped.
        testFile = open(testFileNm, "r")      
        
        #populating training trainX(features) and trainY(labels)
        train_lines = trainFile.read().split("\n") # "\r\n" if needed
        trainX = [];
        trainY = [];
        
        for line in train_lines:
            if line != "": # add other needed checks to skip titles
                cols = line.split(",")
                trainX.append([float(cols[0]),float(cols[1])]);
                trainY.append(int(cols[2]));
         
         
        #populating training testX(features) and testY(labels)       
        test_lines = testFile.read().split("\n") # "\r\n" if needed       
        testX = [];
        testY = [];
        
        for line in test_lines:
            if line != "": # add other needed checks to skip titles
                cols = line.split(",")
                testX.append([float(cols[0]),float(cols[1])]);
                testY.append(int(cols[2]));
        
        
        org_scorer = make_scorer(self.org_loss_func, greater_is_better=False)
        
        #Using SKLearn inbuilt svm library
        svc_model_org = GridSearchCV(svm.SVC(kernel='rbf', gamma=0.1),
                       scoring=org_scorer,
                       cv=5,
                       param_grid={"C": [1e0, 1e1]})
#         svc_model_org = svm.SVC()
        #train the model with flipped training data
        svc_model_org = svc_model_org.fit(trainX, trainY);
        
        print('training score',  svc_model_org.score(trainX,trainY))
        #Predict the test data labels by using the same model.
#         test_predictions_org = svc_model_org.predict(testX);
        
        #Calculating custom loss of each of the test data
        #test_cust_acc_org = self.org_loss_func(testY, test_predictions_org)
#         test_cust_acc_org = svc_model_org.score(testX,testY)
        test_cust_acc_org = svc_model_org.score(testX,testY)
        print('testing score', test_cust_acc_org)
      
        
        #Calculating mean of the total custom loss     
        mean_test_cust_acc_org = test_cust_acc_org / len(testY)
        
#         print(accuracy_score(testY, test_predictions_org, normalize=False))
        
        cust_scorer = make_scorer(self.custom_loss_func, greater_is_better=False)
        svc_model_cust = GridSearchCV(svm.SVC(kernel='rbf', gamma=0.1),
               scoring=cust_scorer,
               cv=5,
               param_grid={"C": [1e0, 1e1]})
        
        
        #train the model with flipped training data
        svc_model_cust = svc_model_cust.fit(trainX, trainY); 
        print('training score',  svc_model_cust.score(trainX,trainY))       
        
        #Predict the test data labels by using the same model.
        test_predictions_cust = svc_model_cust.predict(testX);
        
        #Calculating custom loss of each of the test data
        #test_cust_acc_cust = self.custom_loss_func(testY, test_predictions_cust)
        test_cust_acc_cust = svc_model_cust.score(testX,testY)
#         test_cust_acc_cust = self.org_loss_func(testY,test_predictions_cust)
        print('testing score',  test_cust_acc_cust)
        
        #Calculating mean of the total custom loss     
        mean_test_cust_acc_cust = test_cust_acc_cust / len(testY)
        
#         print(accuracy_score(testY, test_predictions_cust, normalize=False))
        
        return mean_test_cust_acc_org, mean_test_cust_acc_cust