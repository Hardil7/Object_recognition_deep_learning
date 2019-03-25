%% Download the dataset

% Location of the compressed data set
url = 'E:\Lakehead\SEM 2\Neural networks\Project\256_ObjectCategories.tar';

% Store the output in a temporary folder
outputFolder = fullfile('E:\Lakehead\SEM 2\Neural networks\Project\caltech256'); % define output folder

if ~exist(outputFolder, 'dir')
    disp('Extracting Caltech256 data set...');
    untar(url, outputFolder);
end

disp('12 steps to output')
%% create imagedatastore
imagefolders = fullfile(outputFolder, '256_ObjectCategories');
imds = imageDatastore(fullfile(imagefolders), 'LabelSource', 'foldernames','IncludeSubfolders',true);
clear url outputFolder imagefolders;
%% Split each label 
% Using 30 images for training and rest for testing
[trainingSet, testingSet] = splitEachLabel(imds, 30);
disp('1. preprocessing training image for resnet101');
trainingSet.ReadFcn = @(filename)readAndPreprocessImageForGoogle(filename); %redefine read function to process images while read
disp('2. preprocessing testing image for resnet101');
testingSet.ReadFcn = @(filename)readAndPreprocessImageForGoogle(filename); %redefine read function to process images while read
clear imds
%% Load resnet
disp('3. Loading Pretrained Resnet');
 net = resnet101;
 %% Get features from resnet
 disp('4. Loading Resnet train features');
 gpuDevice(1)
 resnet_features_train = activations(net,trainingSet,'fc1000','MiniBatchSize',50);
 disp('Loading Resnet test features');
 disp('5. Loading Resnet test features');
 resnet_features_test = activations(net,testingSet,'fc1000','MiniBatchSize',50);
 resnet_features_train = reshape(resnet_features_train,[1*1*1000,size(resnet_features_train,4)])' ;
 resnet_features_test = reshape(resnet_features_test,[1*1*1000,size(resnet_features_test,4)])';
 
%% Load inceptionv3
disp('6. preprocessing training image for inceptionv3');
trainingSet.ReadFcn = @(filename)readAndPreprocessImage(filename); %redefine read function to process images while read
disp('7. preprocessing testing image for inceptionv3');
testingSet.ReadFcn = @(filename)readAndPreprocessImage(filename); %redefine read function to process images while read
net = inceptionv3;
%% Get deep features from inceptionv3

gpuDevice(1)
disp('8. Loading inceptionv3 train features');
inceptionv3_features_train = activations(net,trainingSet,'avg_pool','MiniBatchSize',50);
disp('9. Loading inceptionv3 test features');
inceptionv3_features_test = activations(net,testingSet,'avg_pool','MiniBatchSize',50);
inceptionv3_features_train = reshape(inceptionv3_features_train,[1*1*2048,size(inceptionv3_features_train,4)])' ;
inceptionv3_features_test = reshape(inceptionv3_features_test,[1*1*2048,size(inceptionv3_features_test,4)])';

%% Merge Resnet and inceptionv3 features
disp('10. Combining the features from inceptionv3 and resnet');
new_F_train = horzcat(inceptionv3_features_train, resnet_features_train);
new_F_test = horzcat(inceptionv3_features_test, resnet_features_test);

%% Converting the Labels.
train_labels = grp2idx(trainingSet.Labels);
test_labels = grp2idx(testingSet.Labels);

%% Concating labels and dataset together.
disp('11. creating training and testing dataset for elm');
training = horzcat(train_labels,new_F_train);
testing = horzcat( test_labels,new_F_test);
%% Using the Deep Features in ELM classifier.
C = 2^-12;
disp('12. Classification using ELM');
[TrainingTime, TestingAccuracy,Training,Testing] = ELM(training, testing, 1, 10000,'sig',C);
