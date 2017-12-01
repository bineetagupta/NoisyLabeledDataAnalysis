import csv
from asu_fsl.CreateTrainTest import crtTrainTestCls
from asu_fsl.perceptron_fsl_org import noisyLblPerceptronOrg
from asu_fsl.perceptron_fsl_cust import noisyLblPerceptronCust

perceptronOpFile = open('perceptronOpFile.csv', 'w')

nrs = [[0.1,0.2],[0.2,0.4],[0.4,0.4],[0.6,0.8],[0.7,0.7]];

path = '/Users/jaydeep/jaydeep_workstation/ASU/Fall2017/CSE569_FSL/project/';          
datasets_num = 5;
nr_num = 1;
with perceptronOpFile:
    writer = csv.writer(perceptronOpFile)
    val = ['DataSet', 'NR_pos', 'NR_neg', 'Accuracy-OriginalLoss', 'Accuracy-CustomLoss']
    writer.writerows([val])
    for i in range(1, datasets_num+1):
        test_filename = path + 'testData_'+str(i)+'.csv'
        print(test_filename)
        for j in range(1, nr_num+1):
            train_fl_nm = 'DS'+str(i)+'_NR'+str(j)+'_flippedTrainData.csv'
            train_filename = path + train_fl_nm
            print(train_filename)
            rho = nrs[j-1];
            crtTrainTestClsObj = crtTrainTestCls()
            trainX,trainY,testX,testY = crtTrainTestClsObj.crtTrainTest(train_filename,test_filename)
            clsOrg = noisyLblPerceptronOrg()
            mean_test_cust_acc_org = clsOrg.calcPreceptronOrg(trainX, testX, trainY, testY, train_fl_nm)
            clsCust = noisyLblPerceptronCust()
            mean_test_cust_acc_cust = clsCust.calcPreceptronCust(trainX, testX, trainY, testY, train_fl_nm,rho[0],rho[1])
            val = [str(i), str(rho[0]), str(rho[1]), mean_test_cust_acc_org, mean_test_cust_acc_cust]
            print(val)
            writer.writerows([val])


