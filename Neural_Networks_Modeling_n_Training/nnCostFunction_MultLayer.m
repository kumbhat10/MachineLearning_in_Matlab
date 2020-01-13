function [J ,grad] = nnCostFunction_MultLayer(nn_params, N1, N2, N3, N4, X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices.

Theta1 = reshape(nn_params(1: N2*(N1+1)), N2, (N1+1) );
Theta2 = reshape(nn_params(1+N2*(N1+1): N2*(N1+1)+ N3*(N2+1) ), N3, N2+1 );
Theta3 = reshape(nn_params(1+N2*(N1+1)+ N3*(N2+1):end), N4, N3+1);

m = size(X, 1);
%% cost and gradient calculation
a1    = [ones(m,1) X];
z2    = a1 * Theta1' ;
a2    = [ones(m,1) sigmoid(z2)] ;    %% a2 = g(Z2) = g(X * theta1')
z3    = a2 * Theta2' ;
a3    = [ones(m,1) sigmoid(z3)] ;    %% a3 = g(Z3) = g(a2 * theta2') 
z4    = a3 * Theta3' ;
a4    = sigmoid(z4)  ;

y_kmatrix = repmat(1:N4,m,1);
y_kmatrix = double(y_kmatrix==y);

J = -sum(y_kmatrix.*log(a4) +(1-y_kmatrix).*log(1-a4),'all')/m ...
    + lambda/2/m* (sum(Theta1(:,2:end).^2,'all') + sum(Theta2(:,2:end).^2,'all') + sum(Theta3(:,2:end).^2,'all'));

d4 = a4 - y_kmatrix;  %% error function 
d3 = d4 * Theta3(:,2:end) .* sigmoidGradient(z3);

% d3temp = a3 - d3; 
d2 = d3 * Theta2(:,2:end) .* sigmoidGradient(z2);

Theta1_grad = (d2' * a1)/m  + [zeros(size(Theta1,1),1) lambda/m*Theta1(:,2:end) ];  %% not regularize first column for bias j=0
Theta2_grad = (d3' * a2)/m  + [zeros(size(Theta2,1),1) lambda/m*Theta2(:,2:end) ];  %% not regularize first column for bias j=0
Theta3_grad = (d4' * a3)/m  + [zeros(size(Theta3,1),1) lambda/m*Theta3(:,2:end) ];  %% not regularize first column for bias j=0

%% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:) ; Theta3_grad(:)];

end
