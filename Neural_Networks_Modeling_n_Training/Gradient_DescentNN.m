function [Theta1, Theta2, J, X_collect] = Gradient_DescentNN(X, y,initial_Theta1,initial_Theta2, lambda, alpha, iterations)
input_layer_size  = 400;  % 20x20 Input Images of Digits size(initial_Theta1,2)-1
hidden_layer_size = 25;   % 25 hidden units size(initial_Theta1,1)
num_labels = 10;

nn_params = [initial_Theta1(:) ; initial_Theta2(:)];

costFunc = @(p) nnCostFunction(p, input_layer_size, hidden_layer_size, ...
    num_labels, X, y, lambda);

S='Iteration ';
%% loop for gradient descent
J = zeros(iterations,1);
X_collect = zeros(size(nn_params,1),iterations);
for iter = 1:iterations
    [J(iter) ,grad] =  costFunc(nn_params);
    nn_params = nn_params - alpha*grad;
    X_collect (:,iter) = nn_params ;
    fprintf('%s %4i | Cost: %4.6f\n', S, iter, J(iter) );
    
end
%% unroll theta1 and theta 2
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
    hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
    num_labels, (hidden_layer_size + 1));

end