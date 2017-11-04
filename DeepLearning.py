import tensorflow as tf
import numpy as np
import pandas as pd


def custom_loss_func(ground_truth, predictions, rho_pos, rho_neg): 
    ground_truth = np.array(ground_truth);
    predictions = np.array(predictions)
    
    neg_indices = [i for i,x in enumerate(ground_truth) if x == -1]
    neg_truth_val = ground_truth[neg_indices]
    neg_pred_val = predictions[neg_indices]
    
    pos_indices = [i for i,x in enumerate(ground_truth) if x == 1]
    pos_truth_val = ground_truth[pos_indices]
    pos_pred_val = predictions[pos_indices]
    
    neg_Loss = ((1-np.count_nonzero((neg_truth_val == neg_pred_val)))/neg_pred_val.shape[0])
    pos_Loss = ((1-np.count_nonzero((pos_truth_val == pos_pred_val)))/pos_pred_val.shape[0])
    
    cust_loss = (1-rho_neg)*pos_Loss - rho_pos*neg_Loss
    return cust_loss


rho_pos = 0.1
rho_neg = 0.1
#train and test files
train_file = '/Users/jaydeep/jaydeep_workstation/ASU/Fall2017/CSE569_FSL/project/trainData.csv'
test_file = '/Users/jaydeep/jaydeep_workstation/ASU/Fall2017/CSE569_FSL/project/testData.csv'

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

y_train_mat = y_train.as_matrix()
train_predicted = []
for i in range(0, len(x_train)):
    train_res = sess.run(prediction, feed_dict={x: [x_train.as_matrix()[i]], pkeep:1})
    train_res[train_res == 1] = -1
    train_res[train_res == 0] = 1
    train_predicted.append(train_res[0])



train_ground_truth = np.array(y_train_actual.as_matrix())
train_predictions = np.array(train_predicted)
print("Training Custom Loss:- ", custom_loss_func(train_ground_truth, train_predictions, rho_pos, rho_neg))

#finding accuracy on testing set
feed_dict_test = {x: x_test,
                  y_: [t for t in y_test.as_matrix()], pkeep:1} 

acc = sess.run(accuracy, feed_dict=feed_dict_test)

print("Test Accuracy:- "+str(acc))


y_test_mat = y_test.as_matrix()
test_predicted = []
for i in range(0, len(x_test)):
    test_res = sess.run(prediction, feed_dict={x: [x_test.as_matrix()[i]], pkeep:1})
    test_res[test_res == 1] = -1
    test_res[test_res == 0] = 1
    test_predicted.append(test_res[0])
    
test_ground_truth = np.array(y_test_actual.as_matrix())
test_predictions = np.array(test_predicted)
print("Testing Custom Loss:- ", custom_loss_func(test_ground_truth, test_predictions, rho_pos, rho_neg))