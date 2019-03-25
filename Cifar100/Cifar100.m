%% Transfer Learning on Cifar100 using GoogLeNet.

% Select the location of the dataset
outputFolder = fullfile('E:\Lakehead\SEM 2\Neural networks\Project\Cifar100\cifar-100-matlab\CIFAR-100'); % define output folder


%%
trainFolder = fullfile(outputFolder, 'TRAIN');
testFolder =  fullfile(outputFolder, 'TEST');
trainingSet = imageDatastore(fullfile(trainFolder), 'LabelSource', 'foldernames','IncludeSubfolders',true);
TestSet = imageDatastore(fullfile(testFolder), 'LabelSource', 'foldernames','IncludeSubfolders',true);

%%
trainingSet.ReadFcn = @(filename)readAndPreprocessImageForGoogle(filename); %redefine read function to process images while read
TestSet.ReadFcn = @(filename)readAndPreprocessImageForGoogle(filename); %redefine read function to process images while read
%%
lrate=20;
miniBatchSize = 30;
net = googlenet;
lgraph = layerGraph(net);
lgraph = removeLayers(lgraph, {'loss3-classifier','prob','output'});%discard output layers
numClasses = numel(categories(trainingSet.Labels));%Set the fully connected layer to the same size as the number of classes in the new data sat. 
newLayers = [
    fullyConnectedLayer(numClasses,'Name','fc','WeightLearnRateFactor',lrate,'BiasLearnRateFactor', lrate)%set the learning rate of new layers
    softmaxLayer('Name','softmax')
    classificationLayer('Name','classoutput')];

lgraph = addLayers(lgraph,newLayers);
lgraph = connectLayers(lgraph,'pool5-drop_7x7_s1','fc'); %add the new output layers to the pretrained CNN
figure('Units','normalized','Position',[0.3 0.3 0.4 0.4]);
plot(lgraph)
ylim([0,10])
%%
%training options
options = trainingOptions('sgdm',...
    'MiniBatchSize',miniBatchSize,... %set mini batch size
      'LearnRateSchedule','piecewise',...
      'LearnRateDropFactor',0.1,... 
      'LearnRateDropPeriod',3,... 
      'MaxEpochs',12,...
      'InitialLearnRate',1e-3,...
    'ValidationFrequency',3, ...
    'Verbose',false, ...
    'Plots','training-progress',...
    'ExecutionEnvironment','auto');
%% Train the network using the training data.
ans = gpuDevice(1);
clear ans;
net = trainNetwork(trainingSet,lgraph,options);
save GoogLeNetCifar100 net
clear lgraph lrate miniBatchSize newLayers numClasses outputFolder testFolder trainFolder options
%%
predictedLabels = classify(net,TestSet);
fin_accuracy = mean(predictedLabels == TestSet.Labels);
fprintf('accuracy of ImageNet Pretrained GoogLeNet: %s \n', fin_accuracy);