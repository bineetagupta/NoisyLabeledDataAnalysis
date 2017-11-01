% The C parameter tells the SVM optimization how much you want to avoid misclassifying each training 
% example. For large values of C, the optimization will choose a smaller-margin hyperplane if that 
% hyperplane does a better job of getting all the training points classified correctly. Conversely, 
% a very small value of C will cause the optimizer to look for a larger-margin separating hyperplane,
% even if that hyperplane misclassifies more points. For very tiny values of C, you should get 
% misclassified examples, often even if your training data is linearly separable.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Separable Data
%Nonseparable Data. SVM can use a soft margin, meaning a hyperplane that separates many, but not all data points.
% There are two standard formulations of soft margins. Both involve adding slack variables ?j and a penalty parameter C.
% The L1-norm , L2-norm
%Nonlinear Transformation with Kernels
% Some binary classification problems do not have a simple hyperplane as a useful separating criterion.
% Polynomials, Radial basis function (Gaussian), Multilayer perceptron or sigmoid
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Training an SVM Classifier
% SVMModel = fitcsvm(X,Y,'KernelFunction','rbf','Standardize',true,'ClassNames',{'negClass','posClass'});
%Classifying New Data with an SVM Classifier
% [label,score] = predict(SVMModel,newX);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear;
rng(1);  % For reproducibility
train_data = csvread('NR1_trainData.csv');
X_train=train_data(:,1:end-1);
Y_train=train_data(:,end);

test_data = csvread('testData.csv');
X_test=test_data(:,1:end-1);
Y_test=test_data(:,end);

CVSVMModel = fitcsvm(X_train,Y_train);
%https://www.mathworks.com/help/stats/classificationsvm.resubloss.html
%https://www.mathworks.com/help/stats/classificationkernel.loss.html
linearloss = @(C,S,W,Cost)sum(-W.*sum(S.*C,2))/sum(W);
train_loss = loss(CVSVMModel,X_train,Y_train,'LossFun',linearloss);
test_loss = loss(CVSVMModel,X_test,Y_test,'LossFun',linearloss);

[label,scores] = predict(CVSVMModel,X_test);

table(Y_test,label,scores(:,2),'VariableNames',{'TrueLabel','PredictedLabel','Score'})

[C,order] = confusionmat(Y_test,label);


function l = customLoss(model,rho_pos,rho_neg)
L = resubLoss(model,'LossFun','logit');
l = ((1-rho_neg)*L - rho_pos*L)/(1-(rho_pos+rho_neg));
end
