NumberOfClasses = 52;

% Loading the datasets
[TestSet] = load('Char_UpperLower52.test.arff');
[DataSet] = load('Char_UpperLower52.train.arff');
[ValidationSet] = load('Char_UpperLower52.val.arff');

%Processing the Test input datasets
TempTestTarget = transpose(ValidationSet(:,109));
TestData = transpose(ValidationSet(:,1:108));
[TestRow,TestCol] = size(TempTestTarget);
TestTarget = zeros(NumberOfClasses, TestCol, 'double');

for n = 1:TestCol
    class = TempTestTarget(1, n);
    TestTarget(class, n) = 1;
end

%Processing the Test input datasets
TempTrainTarget = transpose(DataSet(:,109));
TrainData = transpose(DataSet(:,1:108));
[TrainRow,TrainCol] = size(TempTrainTarget);
TrainTarget = zeros(NumberOfClasses, TrainCol, 'double');

for i = 1:TrainCol
    class = TempTrainTarget(1, i);
    TrainTarget(class, i) = 1;
end

%Processing the Test input datasets
TempValidationTarget = transpose(ValidationSet(:,109));
ValidationData = transpose(ValidationSet(:,1:108));
[ValidationRow, ValidationCol] = size(TempValidationTarget);
ValidationTarget = zeros(NumberOfClasses, ValidationCol, 'double');

for o = 1:ValidationCol
    class = TempValidationTarget(1, o);
    ValidationData(class, o) = 1;
end

% Setting up the neural networks
setdemorandstream(491218382);

net = feedforwardnet(80);
net.trainParam.min_grad = 1.00e-07;
net.trainParam.max_fail = 1000;
net.trainParam.epochs = 1000;
net.divideParam.trainRatio = 98/100;
net.divideParam.valRatio = 1/100;
net.divideParam.testRatio = 1/100;

% Training the neural networks
[net,tr] = train(net, TestData, TestTarget,'useGPU','yes');

% validate the network on the test, training and validation dataset
% Test dataset validation
% TestOutputs = net(TestTarget);
% TestErrors = gsubtract(targets, TestOutputs);
% TestPerformance = perform(net,targets,outputs)
% 
% plotconfusion(TestOutputs, TestErrors )

% Training dataset validation
TrainingOutputs = net(TrainData);
TrainingErrors = gsubtract(TrainTarget, TrainingOutputs);
TrainingPerformance = perform(net, TrainTarget, TrainingOutputs);
plotconfusion(TrainingOutputs, TrainingErrors );

% Validation dataset validation
% ValidationOutputs = net(inputs);
% ValidationErrors = gsubtract(targets,outputs);
% ValidationPerformance = perform(net,targets,outputs)

