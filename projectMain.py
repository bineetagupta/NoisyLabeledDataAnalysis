import csv
from SVM import noisyLblSVM
from DeepLearning import calNN

svmOpFile = open('svmOpFile.csv', 'w')

nrs = [[0.4,0.4]];

path = '/Users/jaydeep/jaydeep_workstation/ASU/Fall2017/CSE569_FSL/project/';          
datasets_num = 1;
nr_num = 1;
with svmOpFile:
    writer = csv.writer(svmOpFile)
    val = ['DataSet', 'NR_pos', 'NP_neg', 'Original Loss', 'Custom Loss']
    writer.writerows([val])
    for i in range(1, datasets_num+1):
        test_filename = path + 'testData_'+str(i)+'.csv'
        print(test_filename)
        for j in range(1, nr_num+1):
            train_filename = path + 'DS'+str(i)+'_NR'+str(j)+'_flippedTrainData.csv'
            print(train_filename)
            rho = nrs[j-1];
            cls = noisyLblSVM()
            mean_test_cust_acc_org, mean_test_cust_acc_cust = cls.calSVM(rho[0],rho[1],train_filename,test_filename);
            val = [str(i), str(rho[0]), str(rho[1]), mean_test_cust_acc_org, mean_test_cust_acc_cust]
            print(val)
            writer.writerows([val])
            
# nnOpFile = open('nnOpFile.csv', 'w')
# with nnOpFile:
#     writer = csv.writer(nnOpFile)
#     val = ['DataSet', 'NR_pos', 'NP_neg', 'Entropy Loss', 'Custom Loss Val', 'Precision Val', 'Recall Val', 'F1 Val', 'Accuracy Val']
#     writer.writerows([val])
#     for i in range(1, datasets_num+1):
#         test_filename = path + 'testData_'+str(i)+'.csv'
#         for j in range(1, nr_num+1):
#             train_filename = path + 'DS'+str(i)+'_NR'+str(j)+'_flippedTrainData.csv'
#             rho = nrs[j-1];
#             emtropy_loss, mean_test_cust_acc, precision_val, recall_val, f1_val, acc_val = calNN(rho[0],rho[1],train_filename,test_filename);
#             val = [str(i), str(rho[0]), str(rho[1]), str(emtropy_loss), mean_test_cust_acc, precision_val, recall_val, f1_val, acc_val]
#             writer.writerows([val])
#     