function DataGeneration_1()

    load benchmarks.mat banana

    x_train = banana.x(banana.train(42,:),:);
    t_train = banana.t(banana.train(42,:));
    x_test  = banana.x(banana.test(42,:),:);
    t_test  = banana.t(banana.test(42,:));

    train_data = [x_test,t_test];
    test_data = [x_train,t_train];
    train_data(train_data(:,3)==-1,3) = 0;
    test_data((test_data(:,3)==-1),3) = 0;
    csvwrite('trainData_1.csv',train_data);
    csvwrite('testData_1.csv',test_data);
    
    
    idx_0 = train_data(:,3) == 1;
    lbl_0 = train_data(idx_0,:); %All the training data with label 1
    idx_1 = train_data(:,3) == 0;
    lbl_1 = train_data(idx_1,:); %All the training data with label 0
    
    h = figure();
    h1 = scatter(lbl_0(:,1),lbl_0(:,2),10,'r.');hold on;
    h2 = scatter(lbl_1(:,1),lbl_1(:,2),10,'g.');hold on
    legend([h1,h2],'Cluster 1','Cluster 2','Location','NW')
    saveas(h,'Data_1.png');

end