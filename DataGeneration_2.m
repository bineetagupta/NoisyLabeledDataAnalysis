C1 = [1 2];   % center of the circle
C2 = [-1 4];
R1 = [8 10];  % range of radii
R2 = [8 10];
A1 = [1 3]*pi/2; % [rad] range of allowed angles
A2 = [-1 1]*pi/2;

nPoints = 1400;

urand = @(nPoints,limits)(limits(1) + rand(nPoints,1)*diff(limits));
randomCircle = @(n,r,a)(pol2cart(urand(n,a),urand(n,r)));

[P1x,P1y] = randomCircle(nPoints,R1,A1);
P1x = P1x + C1(1);
P1y = P1y + C1(2);

[P2x,P2y] = randomCircle(nPoints,R2,A2);
P2x = P2x + C2(1);
P2y = P2y + C2(2);

figure
plot(P1x,P1y,'or'); hold on;
plot(P2x,P2y,'sb'); hold on;
axis square

cluster1 = [P1x,P1y];
cluster2 = [P2x,P2y];

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

csvwrite('trainData_2.csv',train_data)
csvwrite('testData_2.csv',test_data)