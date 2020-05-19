
format()

nclass = 2;
neu = 5;

k=0;

filename = sprintf('set_train_rnn%d',k);    
train1 = importdata(filename);
filename = sprintf('set_val_rnn%d',k);    
val = importdata(filename);
filename = sprintf('set_test_rnn%d',k);    
test = importdata(filename);

[n1 n2]=size(train1);
[n3 n4a]=size(val);
[n5 n6]=size(test);

train4 = [train1 val test];

n4 = n2+n4a;

[n7 n8] = size(train4);
X = train4(1:end-1,:);
T = train4(end,:);
T2=transformtarget(T,nclass);
T2 = T2';
T = T2;
net = layrecnet(1:2,neu);
net.trainFcn = 'trainbr';
%net.trainFcn = 'trainbfg';
net.trainParam.epochs = 5;
%net.trainParam.lr = 0.0001;
net.trainParam.showWindow = false;
net.trainParam.showWindow=0;
net.performParam.regularization = 0.5;
net.divideFcn = 'divideind'; 
net.performFcn = 'mse'; 
net.divideParam.trainInd = 1:n2;
net.divideParam.valInd   = n2+1:n4;
net.divideParam.testInd  = n4+1:size(train4,2);

[net,tr] = train(net,X,T);
Y = net(X);
newy = train4(end,:);
newx = [T(:,1:n4)'*net.LW{2,1};Y(:,n4+1:end)'*net.LW{2,1}];
newx = newx';
newdata = [newx; newy];

filename = sprintf('data_svm%d.txt',k);
dlmwrite(filename,newdata);
filename = sprintf('train_rnn_%d',k);
save (filename, 'net') ;
