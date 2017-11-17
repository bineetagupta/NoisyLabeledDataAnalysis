% https://www.mathworks.com/help/stats/gmdistribution.cluster.html
MU1 = [9 9];
SIGMA1 = [2 0; 0 1];
MU2 = [5 6];
SIGMA2 = [1 0; 0 1];
rng(1); % For reproducibility
X = [mvnrnd(MU1,SIGMA1,1000);mvnrnd(MU2,SIGMA2,1000)];

h = figure();
scatter(X(:,1),X(:,2),10,'.')
hold on


obj = fitgmdist(X,2);
idx = cluster(obj,X);
cluster1 = X(idx == 1,:);
cluster2 = X(idx == 2,:);
h1 = scatter(cluster1(:,1),cluster1(:,2),10,'r.');
h2 = scatter(cluster2(:,1),cluster2(:,2),10,'g.');
legend([h1 h2],'Cluster 1','Cluster 2','Location','NW')
saveas(h,'Data_1.png');

cluster1 = [cluster1 zeros(size(cluster1,1),1)];
cluster1(:,3) = 1 ;

cluster2 = [cluster2 zeros(size(cluster2,1),1)];
cluster2(:,3) = -1 ;


%70-30 data split
idx = randperm(size(cluster1,1));
train_cluster1 = cluster1(idx(1:700),:);
test_cluster1 = cluster1(idx(701:end),:);
train_cluster2 = cluster2(idx(1:700),:);
test_cluster2 = cluster2(idx(701:end),:);

train_data = [train_cluster1;train_cluster2];
test_data = [test_cluster1;test_cluster2];

csvwrite('trainData_1.csv',train_data)
csvwrite('testData_1.csv',test_data)
