datasets = {'trainData_1.csv','trainData_2.csv','trainData_3.csv','trainData_4.csv'};

dataset_ind = 1;
for dataset = datasets
   NRs = {[0.1,0.2],[0.6,0.9],[0.5,0.6],[0.8,0.3],[0.8,0.7]};
   nrt_ind = 1;
   for nr = NRs
       rho = nr{1};
       rho_pos = rho(1);
       rho_neg = rho(2);
       data = csvread(dataset{1});  %Select the entire training data

       idx_0 = data(:,3) == 1;
       lbl_0 = data(idx_0,:); %All the training data with label 0
       idx_1 = data(:,3) == -1;
       lbl_1 = data(idx_1,:); %All the training data with label 1
       
       %rho(+1) = rho(-1) = rho_pos. (noise rates) Select manually
       train_size = size(lbl_0,1); %700
       flip_train_size = train_size * rho_pos; %70 (0.1)
       idx = randsample(train_size,flip_train_size); %select random 70 rows from 0 labled training data
       lbl_0(idx,3) = -1; %flipped the label of the selected data in above step
       
       train_size = size(lbl_1,1); %700
       flip_train_size = train_size * rho_neg; %70 (0.1)
       idx = randsample(train_size,flip_train_size); %select random 70 rows from 1 labled training data
       lbl_1(idx,3) = 1; %flipped the label of the selected data in above step
       
       flipped_data = [lbl_0;lbl_1]; %combined the flipped data
       csv_filename = strcat('DS',int2str(dataset_ind),'_NR',int2str(nrt_ind),'_flippedTrainData.csv');
       csvwrite(csv_filename,flipped_data);
       h = figure();
       hold on;
       ind1 = flipped_data(:,3) == 1;
       cluster1 = flipped_data(ind1,:);
       ind2 = flipped_data(:,3) == -1;
       cluster2 = flipped_data(ind2,:);
       h1 = scatter(cluster1(:,1),cluster1(:,2),10,'r.');
       h2 = scatter(cluster2(:,1),cluster2(:,2),10,'g.');
       legend([h1 h2],'Cluster 1','Cluster 2','Location','NW');
       png_filename = strcat('DS',int2str(dataset_ind),'_NR',int2str(nrt_ind),'_flippedTrainData.png');
       saveas(h,png_filename);
       nrt_ind = nrt_ind + 1;
   end;
   dataset_ind = dataset_ind + 1;
end;

clear;