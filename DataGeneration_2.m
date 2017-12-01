function DataGeneration_2()
    % https://www.mathworks.com/help/stats/gmdistribution.cluster.html
    MU1 = [0 1];
    SIGMA1 = [2 0; 0 0.5];
    MU2 = [-1 -2];
    SIGMA2 = [1 0;0 1];
    rng(1); % For reproducibility
    X = [mvnrnd(MU1,SIGMA1,5000);mvnrnd(MU2,SIGMA2,5000)];

    h = figure();
    scatter(X(:,1),X(:,2),10,'.')
    hold on

    obj = fitgmdist(X,2);
    idx = cluster(obj,X);
    cluster1 = X(idx == 1,:);
    cluster2 = X(idx == 2,:);
    h1 = scatter(cluster1(:,1),cluster1(:,2),10,'r.');
    h2 = scatter(cluster2(:,1),cluster2(:,2),10,'g.');
    legend([h1, h2],'Cluster 1','Cluster 2','Location','NW')
    saveas(h,'Data_2.png');

    cluster1 = [cluster1 zeros(size(cluster1,1),1)];
    cluster1(:,3) = 1 ;

    cluster2 = [cluster2 zeros(size(cluster2,1),1)];
    cluster2(:,3) = 0 ;


    %70-30 data split
    idx = randperm(size(cluster1,1));
    train_cluster1 = cluster1(idx(1:3000),:);
    test_cluster1 = cluster1(idx(3001:end),:);
    train_cluster2 = cluster2(idx(1:3000),:);
    test_cluster2 = cluster2(idx(3001:end),:);

    train_data = [train_cluster1;train_cluster2];
    test_data = [test_cluster1;test_cluster2];

    csvwrite('trainData_2.csv',train_data)
    csvwrite('testData_2.csv',test_data)

end
