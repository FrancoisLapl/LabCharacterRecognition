[DataSet] = load('Char_UpperLower52.test.arff');

Target = transpose(DataSet(:,109));

Data = transpose(DataSet(:,1:108));

setdemorandstream(491218382);

net = fitnet(80);
net.trainParam.min_grad = 1.00e-07;
net.trainParam.max_fail = 1000;
net.trainParam.epochs = 1000;

[net,tr] = train(net, Data, Target,'useGPU','yes');
nntraintool
