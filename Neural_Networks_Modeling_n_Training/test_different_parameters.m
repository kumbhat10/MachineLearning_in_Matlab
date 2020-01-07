
input_layer_size  = 400;  % 20x20 Input Images of Digits
hidden_layer_size = 25;   % 25 hidden units
num_labels = 10;          % 10 labels, from 1 to 10
load('ex4data1.mat');
initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);
% displayData(initial_Theta1(:, 2:end));
initial_Theta1 = Theta1 ;
initial_Theta2 = Theta2 ;
% Unroll parameters
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];

%% parameters to be changed
iterations = 500 ;
lambda = 1;
alpha  = 0.1;
%% train neural network
options = optimset('MaxIter', iterations);
costFunction = @(p) nnCostFunction(p, input_layer_size,     hidden_layer_size,  num_labels, X, y, lambda);

[nn_params,X_collect_fmincg, cost] = fmincg(costFunction, initial_nn_params, options);
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
    hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
    num_labels, (hidden_layer_size + 1));
%% train using gradient descent method - it converges at some local minima
alpha  = 0.1;

[Theta1_GD, Theta2_GD, J, X_collect_GD] = ...
    Gradient_DescentNN(X, y,initial_Theta1,initial_Theta2, lambda, alpha, iterations);

%% Visualize and compare results from fmincg method and Gradient descent method
fprintf('\nVisualizing Neural Network... \n')
displayData(Theta1(:, 2:end));
index = iterations;
plotThetaCollected(X_collect,index)
plotThetaCollected(X_collect_GD,index)
title(['Lambda = ',num2str(lambda),'  iterations = ',num2str(iterations)])
pred = predict(Theta1, Theta2, X);
pred_GD = predict(Theta1_GD, Theta2_GD, X);

fprintf('\nWith fmincg Training Set Accuracy: %0.3f\n', mean(double(pred == y)) * 100);
fprintf('\nWith Gradient Descent :- Training Set Accuracy: %0.3f\n', mean(double(pred_GD == y)) * 100);

fprintf(['Lambda = ',num2str(lambda),'  iterations = ',num2str(iterations),' Training Set Accuracy: %0.3f\n '], mean(double(pred == y)) * 100);
%% Check the cost decrease and prediction accuracy progress with iterations
predict_accuracy = predict_Accuracy(X_collect_fmincg,y,X);
plot(predict_accuracy,'DisplayName',['Lambda = ',num2str(lambda),'  iterations = ',num2str(iterations)])

function predict_accuracy = predict_Accuracy(X_collect,y, X)
predict_accuracy = zeros(size(X_collect,2),1);
input_layer_size  = 400;  % 20x20 Input Images of Digits
hidden_layer_size = 25;   % 25 hidden units
num_labels = 10;

for i = 1:size(X_collect,2)
    nn_params_collected = X_collect(:,i);
    
    Theta1 = reshape(nn_params_collected(1:hidden_layer_size * (input_layer_size + 1)), ...
        hidden_layer_size, (input_layer_size + 1));
    
    Theta2 = reshape(nn_params_collected((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
        num_labels, (hidden_layer_size + 1));
    
    pred = predict(Theta1, Theta2, X);
    predict_accuracy(i,1) = mean(double(pred == y)) * 100 ;
    % fprintf('\nTraining Set Accuracy: %0.3f\n', mean(double(pred == y)) * 100);
end
end