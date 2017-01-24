clear ; close all; clc

%%Setup the parameters 
input_layer_size  = 561;  %no.of features
hidden_layer_size = 20;   
num_labels = 6;          % 6 labels  (output)
                         

%% Load Training Data
fprintf('Loading Data ...\n')
X=textread('X_train.txt');
y=textread('y.txt');
m = size(X, 1); %no.of training samples

%%  Random initialisation
fprintf('\nInitializing Neural Network Parameters ...\n')


Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
Theta2 = randInitializeWeights(hidden_layer_size, num_labels);

% Unroll parameters
initial_nn_params = [Theta1(:) ; Theta2(:)];

%% feedforward

p = predict(Theta1, Theta2, X)

% fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);

%% cost
lambda=1;
costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, X, y, lambda);
                               
                               
                               %% fmincg . outut is final cost and final theta
% Now, costFunction is a function that takes in only one argument (the
% neural network parameters)

options = optimset('MaxIter',300);
iter=300;

[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);


% Obtain Theta1 and Theta2 back from nn_params
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

p = predict(Theta1, Theta2, X);
fprintf('Training data size = %d*%d\n',size(X,1),size(X,2));
fprintf('Hidden units=%d  Iterations=%d\n',hidden_layer_size,iter);
fprintf('\nTraining Set Accuracy: %f\n', mean(double(p == y)) * 100);
testdata(Theta1,Theta2);