data = csvread('trainData.csv');  %Select the entire training data

idx_0 = data(:,3) == 0;
lbl_0 = data(idx_0,:); %All the training data with label 0
idx_1 = data(:,3) == 1;
lbl_1 = data(idx_1,:); %All the training data with label 1

%rho(+1) = rho(?1) = 0.1. (noise rates) Select manually
train_size = size(lbl_0,1); %700
flip_train_size = train_size * 0.1; %70
idx = randsample(train_size,flip_train_size); %select random 70 rows from 0 labled training data
lbl_0(idx,3) = ~lbl_0(idx,3); %flipped the label of the selected data in above step

train_size = size(lbl_1,1); %700
flip_train_size = train_size * 0.1; %70
idx = randsample(train_size,flip_train_size); %select random 70 rows from 1 labled training data
lbl_1(idx,3) = ~lbl_1(idx,3); %flipped the label of the selected data in above step

flipped_data = [lbl_0;lbl_1]; %combined the flipped data
csvwrite('NR1_trainData.csv',flipped_data)
clear;


data = csvread('trainData.csv');  %Select the entire training data
idx_0 = data(:,3) == 0;
lbl_0 = data(idx_0,:); %All the training data with label 0
idx_1 = data(:,3) == 1;
lbl_1 = data(idx_1,:); %All the training data with label 1

%rho(+1) = rho(?1) = 0.2.(noise rates) Select manually
train_size = size(lbl_0,1); %700
flip_train_size = train_size * 0.2; %140
idx = randsample(train_size,flip_train_size); %select random 140 rows from 0 labled training data
lbl_0(idx,3) = ~lbl_0(idx,3); %flipped the label of the selected data in above step

train_size = size(lbl_1,1); %700
flip_train_size = train_size * 0.2; %140
idx = randsample(train_size,flip_train_size); %select random 140 rows from 1 labled training data
lbl_1(idx,3) = ~lbl_1(idx,3); %flipped the label of the selected data in above step

flipped_data = [lbl_0;lbl_1]; %combined the flipped data
csvwrite('NR2_trainData.csv',flipped_data)
clear;

data = csvread('trainData.csv');  %Select the entire training data
idx_0 = data(:,3) == 0;
lbl_0 = data(idx_0,:); %All the training data with label 0
idx_1 = data(:,3) == 1;
lbl_1 = data(idx_1,:); %All the training data with label 1

%rho(+1) = rho(?1) = 0.3.(noise rates) Select manually
train_size = size(lbl_0,1); %700
flip_train_size = train_size * 0.3; %210
idx = randsample(train_size,flip_train_size); %select random210 rows from 0 labled training data
lbl_0(idx,3) = ~lbl_0(idx,3); %flipped the label of the selected data in above step

train_size = size(lbl_1,1); %700
flip_train_size = train_size * 0.3; %210
idx = randsample(train_size,flip_train_size); %select random 210 rows from 1 labled training data
lbl_1(idx,3) = ~lbl_1(idx,3); %flipped the label of the selected data in above step

flipped_data = [lbl_0;lbl_1]; %combined the flipped data
csvwrite('NR3_trainData.csv',flipped_data)
clear;

data = csvread('trainData.csv');  %Select the entire training data
idx_0 = data(:,3) == 0;
lbl_0 = data(idx_0,:); %All the training data with label 0
idx_1 = data(:,3) == 1;
lbl_1 = data(idx_1,:); %All the training data with label 1
%rho(+1) = rho(?1) = 0.4.(noise rates) Select manually
train_size = size(lbl_0,1); %700
flip_train_size = train_size * 0.4; %280
idx = randsample(train_size,flip_train_size); %select random280 rows from 0 labled training data
lbl_0(idx,3) = ~lbl_0(idx,3); %flipped the label of the selected data in above step

train_size = size(lbl_1,1); %700
flip_train_size = train_size * 0.3; %280
idx = randsample(train_size,flip_train_size); %select random 280 rows from 1 labled training data
lbl_1(idx,3) = ~lbl_1(idx,3); %flipped the label of the selected data in above step

flipped_data = [lbl_0;lbl_1]; %combined the flipped data
csvwrite('NR4_trainData.csv',flipped_data)
clear;

data = csvread('trainData.csv');  %Select the entire training data
idx_0 = data(:,3) == 0;
lbl_0 = data(idx_0,:); %All the training data with label 0
idx_1 = data(:,3) == 1;
lbl_1 = data(idx_1,:); %All the training data with label 1
%rho(+1) = rho(?1) = 0.5.(noise rates) Select manually
train_size = size(lbl_0,1); %700
flip_train_size = train_size * 0.5; %350
idx = randsample(train_size,flip_train_size); %select random 350 rows from 0 labled training data
lbl_0(idx,3) = ~lbl_0(idx,3); %flipped the label of the selected data in above step

train_size = size(lbl_1,1); %700
flip_train_size = train_size * 0.5; %350
idx = randsample(train_size,flip_train_size); %select random 350 rows from 1 labled training data
lbl_1(idx,3) = ~lbl_1(idx,3); %flipped the label of the selected data in above step

flipped_data = [lbl_0;lbl_1]; %combined the flipped data
csvwrite('NR5_trainData.csv',flipped_data)
clear;
