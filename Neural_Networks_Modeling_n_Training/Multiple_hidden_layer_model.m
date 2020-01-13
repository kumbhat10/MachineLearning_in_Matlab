% Ni = number of units on i layer
N1  = 400;  % 20x20 Input Images of Digits
N2 = 25;   % 25 hidden units
N3 = 25;   % 25 hidden units

N4 = 10;   % num_labels 10 labels, from 1 to 10
load('ex4data1.mat');
%%
ini_Theta1 = randInitializeWeights (N1, N2);
ini_Theta2 = randInitializeWeights (N2, N3);
ini_Theta3 = randInitializeWeights (N3, N4);

initial_nn_params = [ini_Theta1(:) ; ini_Theta2(:) ; ini_Theta3(:)];

%% parameters to be changed
iterations = 2000 ;
lambda = 1;
alpha  = 0.1;

[J ,grad] = nnCostFunction_MultLayer(initial_nn_params, N1, N2, N3, N4, X, y, lambda) ;
%% train neural network
options                            = optimset('MaxIter', iterations);
costFunction                       = @(p) nnCostFunction_MultLayer(p, N1, N2, N3, N4, X, y, lambda);
[nn_params, X_collect_fmincg, cost] = fmincg(costFunction, initial_nn_params, options);

Theta1 = reshape(nn_params(1: N2*(N1+1)), N2, (N1+1) );
Theta2 = reshape(nn_params(1+N2*(N1+1): N2*(N1+1)+ N3*(N2+1) ), N3, N2+1 );
Theta3 = reshape(nn_params(1+N2*(N1+1)+ N3*(N2+1):end), N4, N3+1);

%% train using gradient descent method - it converges at some local minima
alpha  = 0.1;

[Theta1_GD, Theta2_GD, J, X_collect_GD] = ...
    Gradient_DescentNN(X, y,ini_Theta1,ini_Theta2, lambda, alpha, iterations);

%% Visualize and compare results from fmincg method and Gradient descent method
fprintf('\nVisualizing Neural Network... \n')
displayData(Theta1(:, 2:end));
index = iterations;
plotThetaCollected(X_collect_fmincg,index)

% plotThetaCollected(X_collect_GD,index)
title(['Lambda = ',num2str(lambda),'  iterations = ',num2str(iterations)])
% pred = predict(Theta1, Theta2, X);
pred = predict_MultiLayer(Theta1, Theta2, Theta3, X);
% pred_GD = predict(Theta1_GD, Theta2_GD, X);

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