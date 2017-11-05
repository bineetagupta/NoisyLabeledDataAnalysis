data = csvread('trainData.csv');  %Select the entire training data

idx_0 = data(:,3) == 1;
lbl_0 = data(idx_0,:); %All the training data with label 0
idx_1 = data(:,3) == -1;
lbl_1 = data(idx_1,:); %All the training data with label 1

%rho(+1) = rho(-1) = 0.1. (noise rates) Select manually
train_size = size(lbl_0,1); %700
flip_train_size = train_size * 0.1; %70
idx = randsample(train_size,flip_train_size); %select random 70 rows from 0 labled training data
lbl_0(idx,3) = -1; %flipped the label of the selected data in above step

train_size = size(lbl_1,1); %700
flip_train_size = train_size * 0.1; %70
idx = randsample(train_size,flip_train_size); %select random 70 rows from 1 labled training data
lbl_1(idx,3) = 1; %flipped the label of the selected data in above step

flipped_data = [lbl_0;lbl_1]; %combined the flipped data
csvwrite('NR1_trainData.csv',flipped_data)

hold on;
ind1 = flipped_data(:,3) == 1;
cluster1 = flipped_data(ind1,:);
ind2 = flipped_data(:,3) == -1;
cluster2 = flipped_data(ind2,:);
h1 = scatter(cluster1(:,1),cluster1(:,2),10,'r.');
h2 = scatter(cluster2(:,1),cluster2(:,2),10,'g.');
legend([h1 h2],'Cluster 1','Cluster 2','Location','NW')



