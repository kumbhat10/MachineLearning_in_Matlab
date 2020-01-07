function [J ,grad] = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 

Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));
m = size(X, 1);         
%% cost and gradient calculation
a1    = [ones(m,1) X];
z2    = a1 * Theta1'  ;
a2    = [ones(m,1) sigmoid(z2) ] ;    %% a2 = g(Z2) = g(X * theta1')
z3    = a2 * Theta2' ; 
a3    = sigmoid( z3 ) ;               %% a3 = h_theta = g ( a2 * theta2') = g(Z3)

y_kmatrix = repmat(1:num_labels,m,1);
y_kmatrix = double(y_kmatrix==y);

J = -sum(y_kmatrix.*log(a3) +(1-y_kmatrix).*log(1-a3),'all')/m ...
 + lambda/2/m* (sum(Theta1(:,2:end).^2,'all') + sum(Theta2(:,2:end).^2,'all'));

d3 = a3 - y_kmatrix;   %% error function 
d2 = d3 * Theta2(:,2:end) .* sigmoidGradient(z2);
   
Theta1_grad = (d2' * a1)/m  + [zeros(size(Theta1,1),1) lambda/m*Theta1(:,2:end) ];  %% not regularize first column for bias j=0
Theta2_grad = (d3' * a2)/m  + [zeros(size(Theta2,1),1) lambda/m*Theta2(:,2:end) ];  %% not regularize first column for bias j=0  

%% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
