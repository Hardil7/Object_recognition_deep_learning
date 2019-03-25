% %%
% 
% % Enter the location of Dataset
% outputFolder = fullfile('C:\Users\Student\Desktop\Neural\Cifar10'); % define output folder
% 
% 
% %% Load DataSet
% trainFolder = fullfile(outputFolder, 'cifar10Train');
% testFolder =  fullfile(outputFolder, 'cifar10Test');
% trainingSet = imageDatastore(fullfile(trainFolder), 'LabelSource', 'foldernames','IncludeSubfolders',true);
% validationSet = imageDatastore(fullfile(testFolder), 'LabelSource', 'foldernames','IncludeSubfolders',true);
% 
% %% Preprocess Images for GoogLeNet
% trainingSet.ReadFcn = @(filename)readAndPreprocessImageForGoogle(filename); %redefine read function to process images while read
% validationSet.ReadFcn = @(filename)readAndPreprocessImageForGoogle(filename); %redefine read function to process images while read
% %% Transfer the last three layers for learning.
 lrate=20;
 miniBatchSize = 120;
% net = googlenet;
% lgraph = layerGraph(net);
% lgraph = removeLayers(lgraph, {'loss3-classifier','prob','output'});%discard output layers
% %numClasses = numel(categories(trainingSet.Labels));%Set the fully connected layer to the same size as the number of classes in the new data sat. 
newLayers = [
    fullyConnectedLayer(10,'Name','fc','WeightLearnRateFactor',lrate,'BiasLearnRateFactor', lrate)%set the learning rate of new layers
    softmaxLayer('Name','softmax')
    classificationLayer('Name','classoutput')];

lgraph = addLayers(lgraph,newLayers);
lgraph = connectLayers(lgraph,'pool5-drop_7x7_s1','fc'); %add the new output layers to the pretrained CNN
figure('Units','normalized','Position',[0.3 0.3 0.4 0.4]);
plot(lgraph)
ylim([0,10])
% %%
% %training options
% options = trainingOptions('sgdm',...
%     'MiniBatchSize',miniBatchSize,... %set mini batch size
%       'LearnRateSchedule','piecewise',...
%       'LearnRateDropFactor',0.1,... 
%       'LearnRateDropPeriod',2,... 
%       'MaxEpochs',6,...
%       'InitialLearnRate',1e-3,...
%     'ValidationFrequency',3, ...
%     'Verbose',false, ...
%     'Plots','training-progress',...
%     'ExecutionEnvironment','auto');
% %% Train the network using the training data.
% 
% net = trainNetwork(trainingSet,lgraph,options);
% save GoogLeNetCifar10 net
% clear lgraph lrate miniBatchSize newLayers numClasses outputFolder testFolder trainFolder options
% %% Predict Output
% predictedLabels = classify(net,validationSet);
% fin_accuracy = mean(predictedLabels == validationSet.Labels);
% fprintf('Accuracy of ImageNet Pretrained GoogLeNet: %s \n', fin_accuracy);