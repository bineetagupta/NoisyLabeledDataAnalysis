rng(1); % For reproducibility
r = sqrt(rand(1400,1)); % Radius
t = 2*pi*rand(1400,1);  % Angle
data1 = [r.*cos(t), r.*sin(t)]; % Points

r2 = sqrt(3*rand(1400,1)+1); % Radius
t2 = 2*pi*rand(1400,1);      % Angle
data2 = [r2.*cos(t2), r2.*sin(t2)]; % points

h = figure();
plot(data1(:,1),data1(:,2),'r.','MarkerSize',15)
hold on
plot(data2(:,1),data2(:,2),'b.','MarkerSize',15)
%ezpolar(@(x)1);ezpolar(@(x)2);
axis equal
hold off
saveas(h,'Data_4.png');


data1 = [data1 zeros(size(data1,1),1)];
data1(:,3) = 1 ;

data2 = [data2 zeros(size(data2,1),1)];
data2(:,3) = -1 ;


%70-30 data split
idx = randperm(size(data1,1));
train_cluster1 = data1(idx(1:700),:);
test_cluster1 = data1(idx(701:end),:);
train_cluster2 = data2(idx(1:700),:);
test_cluster2 = data2(idx(701:end),:);

train_data = [train_cluster1;train_cluster2];
test_data = [test_cluster1;test_cluster2];

csvwrite('trainData_4.csv',train_data)
csvwrite('testData_4.csv',test_data)
