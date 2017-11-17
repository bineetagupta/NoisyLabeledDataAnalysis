import csv
from SVM import calSVM
from DeepLearning import calNN

svmOpFile = open('svmOpFile.csv', 'w')

nrs = [[0.1,0.2],[0.6,0.9],[0.5,0.6],[0.8,0.3],[0.8,0.7]];

path = '/Users/jaydeep/jaydeep_workstation/ASU/Fall2017/CSE569_FSL/project/';          
datasets_num = 4;
nr_num = 5;
with svmOpFile:
    writer = csv.writer(svmOpFile)
    val = ['Data', 'NR_pos', 'NP_neg', 'Hinge Loss', 'Custom Accuracy Val', 'Precision Val', 'Recall Val', 'F1 Val', 'Accuracy Val']
    writer.writerows([val])
    for i in range(1, datasets_num+1):
        test_filename = path + 'testData_'+str(i)+'.csv'
        for j in range(1, nr_num+1):
            train_filename = path + 'DS'+str(i)+'_NR'+str(j)+'_flippedTrainData.csv'
            rho = nrs[j-1];
            test_hinge_loss, mean_test_cust_acc, precision_val, recall_val, f1_val, acc_val = calSVM(rho[0],rho[1],train_filename,test_filename);
            val = [str(i), str(rho[0]), str(rho[1]), test_hinge_loss, mean_test_cust_acc, precision_val, recall_val, f1_val, acc_val]
            writer.writerows([val])
            
nnOpFile = open('nnOpFile.csv', 'w')
with nnOpFile:
    writer = csv.writer(nnOpFile)
    val = ['Data', 'NR_pos', 'NP_neg', 'Entropy Loss', 'Custom Accuracy Val', 'Precision Val', 'Recall Val', 'F1 Val', 'Accuracy Val']
    writer.writerows([val])
    for i in range(1, datasets_num+1):
        test_filename = path + 'testData_'+str(i)+'.csv'
        for j in range(1, nr_num+1):
            train_filename = path + 'DS'+str(i)+'_NR'+str(j)+'_flippedTrainData.csv'
            rho = nrs[j-1];
            emtropy_loss, mean_test_cust_acc, precision_val, recall_val, f1_val, acc_val = calNN(rho[0],rho[1],train_filename,test_filename);
            val = [str(i), str(rho[0]), str(rho[1]), str(emtropy_loss), mean_test_cust_acc, precision_val, recall_val, f1_val, acc_val]
            writer.writerows([val])
    