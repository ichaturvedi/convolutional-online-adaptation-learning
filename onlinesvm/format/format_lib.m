dataO = importdata('data_svm0.txt');

data = dataO;
count = 1000 ;
train_cnt = 12000-count;
data = data(:,1:end-train_cnt);
for i=1:size(data,2)
if data(end,i)==1
data(end,i)=-1;
end
if data(end,i)==2
data(end,i)=1;
end
end

data2 = data(1:end-1,:);
data3 = data(end,:);
data4 = [data3' data2'];
data4 = [data4(1,:);data4];
dlmwrite('train',data4);


data = dataO;
data = data(:,end-train_cnt+1:end);
for i=1:size(data,2)
if data(end,i)==1
data(end,i)=-1;
end
if data(end,i)==2
data(end,i)=1;
end
end
data2 = data(1:end-1,:);
data3 = data(end,:);
data4 = [data3' data2'];
data4 = [data4(1,:);data4];
dlmwrite('test',data4);
exit
