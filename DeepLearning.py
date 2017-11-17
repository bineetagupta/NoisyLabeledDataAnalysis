import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score

def loss_0_1(y_actual,y_pred):
    if(y_actual == y_pred):
        return 0
    else:
        return 1

def custom_loss_func(ground_truth, predictions, rho_pos, rho_neg): 
       
    if(predictions == 1):
        cust_loss = ((1-rho_neg)*loss_0_1(ground_truth,predictions) - rho_pos*loss_0_1(ground_truth,-predictions))/(1- rho_pos - rho_neg)
    else:
        cust_loss = ((1-rho_pos)*loss_0_1(ground_truth,predictions) - rho_neg*loss_0_1(ground_truth,-predictions))/(1- rho_pos - rho_neg)
        
    return cust_loss

def calNN(rho_pos_val, rho_neg_val, trainFileNm, testFileNm):

    rho_pos = rho_pos_val
    rho_neg = rho_neg_val
    train_file = open(trainFileNm, "r")
    test_file = open(testFileNm, "r")
    
    #import test data 
    train_data=pd.read_csv(train_file, names=['f1','f2','f3'])
    test_data=pd.read_csv(test_file, names=['f1','f2','f3'])
    
    #shuffle the train data
    train_data=train_data.iloc[np.random.permutation(len(train_data))]
    
    #map data label into arrays, #specifically for deep learning
    lbl_0=np.asarray([1,0])
    lbl_1=np.asarray([0,1])
    
    
    #training data
    x_train=train_data[['f1','f2']]
    y_train_actual=train_data['f3']
    y_train = y_train_actual.map({1: lbl_0, -1: lbl_1})
    #test data
    x_test=test_data[['f1','f2']]
    y_test_actual=test_data['f3']
    y_test = y_test_actual.map({1: lbl_0, -1: lbl_1})
    
    #placeholders and variables. input has 2 features and output has 2 classes
    x=tf.placeholder(tf.float32,shape=[None,2])
    y_=tf.placeholder(tf.float32,shape=[None, 2])
    
    pkeep = tf.placeholder(tf.float32)
    
    #weights and biases and initially they are assigned randomly
    W1 = tf.Variable(tf.truncated_normal([2, 4], stddev=0.1))
    b1 = tf.Variable(tf.zeros([4]))
    W2 = tf.Variable(tf.truncated_normal([4, 4], stddev=0.1))
    b2 = tf.Variable(tf.zeros([4]))
    W3 = tf.Variable(tf.truncated_normal([4, 2], stddev=0.1))
    b3 = tf.Variable(tf.zeros([2]))
    
    # model 2-layer neural network.
    xx = x
    Y1 = tf.nn.sigmoid(tf.matmul(xx, W1) + b1)
    Y1d = tf.nn.dropout(Y1, pkeep)
    Y2 = tf.nn.sigmoid(tf.matmul(Y1d, W2) + b2)
    Y2d = tf.nn.dropout(Y2, pkeep)
    Ylogits = tf.matmul(Y2d, W3) + b3
    y = tf.nn.softmax(Ylogits)#softmax function for multiclass classification
    
    #loss function (Here we have to create new loss function) 
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits = Ylogits, labels = y_)
    cross_entropy = tf.reduce_mean(cross_entropy)*100
    
    #optimiser -
    optimizer = tf.train.AdamOptimizer(0.003).minimize(cross_entropy)
    
    #calculating accuracy of our model
    prediction = tf.argmax(y,1)
    correct_label = tf.argmax(y_,1)
    is_correct = tf.equal(prediction, correct_label)
    accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
    
    #session parameters
    sess = tf.InteractiveSession()
    #initialising variables
    init = tf.global_variables_initializer()
    sess.run(init)
    
    epoch = 1000
    for step in range(epoch):
        feed_dict_train_data = {x: x_train, y_:[t for t in y_train.as_matrix()], pkeep:1}# we kept all the synopsys as we have very less data 
        # train
        sess.run(optimizer, feed_dict=feed_dict_train_data)
    
    
    #finding accuracy on testing set
    feed_dict_test = {x: x_test,
                      y_: [t for t in y_test.as_matrix()], pkeep:1} 
    
    acc = sess.run(accuracy, feed_dict=feed_dict_test)
    
#     print("Test Accuracy:- "+str(acc))
    
    
    y_test_mat = y_test.as_matrix()
    test_predicted = []
    for i in range(0, len(x_test)):
        test_res = sess.run(prediction, feed_dict={x: [x_test.as_matrix()[i]], pkeep:1})
        test_res[test_res == 1] = -1
        test_res[test_res == 0] = 1
        test_predicted.append(test_res[0])
    
    
    test_cust_acc = 0;
    for i in range(0, len(y_test_actual)):
        test_cust_acc += custom_loss_func(y_test_actual[i], test_predicted[i], rho_pos, rho_neg);
        
    mean_test_cust_acc = np.mean(test_cust_acc)
    # print("Custom Test Loss:- ", mean_test_cust_acc)
    
    # Evaluation matrix
    # print("Confusion matrix on test data:- ")
    # print(confusion_matrix(y_test_actual, test_predicted))
    precision_val = precision_score(y_test_actual, test_predicted, average='micro')
    recall_val = recall_score(y_test_actual, test_predicted, average='micro')
    f1_val = f1_score(y_test_actual, test_predicted, average='micro')
    acc_val = accuracy_score(y_test_actual, test_predicted)
    
    return acc, mean_test_cust_acc, precision_val, recall_val, f1_val, acc_val
