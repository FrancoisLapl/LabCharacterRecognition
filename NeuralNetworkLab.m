NUMBER_OF_CLASSES = 52;

% Loading the datasets
[TestSet] = load('Char_UpperLower52.test.arff');
[TrainSet] = load('Char_UpperLower52.train.arff');
[ValidationSet] = load('Char_UpperLower52.val.arff');

%Processing the Test input datasets
[testTarget, testData] = parseData(TestSet, NUMBER_OF_CLASSES);

%Processing the Training input datasets
[trainTarget, trainData] = parseData(TrainSet, NUMBER_OF_CLASSES);

%Processing the Validation input datasets
[validationTarget, validationData] = parseData(ValidationSet, NUMBER_OF_CLASSES);

% Setting up the neural networks
setdemorandstream(491218382);

network = feedforwardnet(80);
% network.trainParam.min_grad = 1.00e-07;
% network.trainParam.max_fail = 1000;
network.trainParam.epochs = 1000;
% network.divideParam.trainRatio = 98/100;
% network.divideParam.valRatio = 1/100;
% network.divideParam.testRatio = 1/100;

% Training the neural networks
[trainedNetwork, trainingRecord] = train(network, testData, testTarget,'useGPU','yes');

% Training dataset validation
trainingOutputs = trainedNetwork(testData);
[processedOutput] = processNetworkOutput(trainingOutputs);
PrintConfusionMatrix(testTarget, processedOutput);

%functions
function [target,dataset] = parseData(rawDataSet, numberOfClasses)
    [~,rawColCount] = size(rawDataSet);
    tempTarget = transpose(rawDataSet(:,rawColCount));
    dataset = transpose(rawDataSet(:,1:rawColCount-1));
    [~,TestColCount] = size(tempTarget);
    target = zeros(numberOfClasses, TestColCount, 'double');

    for n = 1:TestColCount
        class = tempTarget(1, n);
        target(class, n) = 1;
    end
end

function [processedOutput] = processNetworkOutput(aiResult)
    [rowCount,colCount] = size(aiResult);
    %Create a zeroed matrix
    processedOutput =  zeros(rowCount, colCount, 'double');
    
    for i = 1:colCount
        %For each column of the aiResultDataset find the most confident
        %value(biggest).
        biggestValue = aiResult(1,i);
        biggestValueIndex = 1;
        
        for j = 2:rowCount
            if aiResult(j,i) > biggestValue
                biggestValue = aiResult(j,i);
                biggestValueIndex = j;
            end
        end
        processedOutput(biggestValueIndex, i) = 1;
    end
    
end

function [] = PrintConfusionMatrix(trainTarget, processedResultMatrix)
    trainingErrors = gsubtract(trainTarget, processedResultMatrix);
    trainingPerformance = perform(trainedNetwork, TrainTarget, trainingOutputs);
    plotconfusion(processedResultMatrix, trainingErrors );
end
