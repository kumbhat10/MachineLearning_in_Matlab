function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the
%   cost of using theta as the parameter for linear regression to fit the
%   data points in X and y. Returns the cost in J and the gradient in grad

m = length(y); % number of training examples

J = sum((y - X*theta).^2) / 2/m + lambda/2/m * theta(2:end)'*theta(2:end);
grad = X'*(X*theta - y ) /m  +  [0;lambda/m*theta(2:end)];

grad = grad(:);

end
