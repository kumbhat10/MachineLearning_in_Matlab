function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);
theta_hist   = zeros(num_iters, size(X,2));

for iter = 1:num_iters 
    J_history(iter) = computeCostMulti(X, y, theta);
    theta = theta - (alpha/m * sum(( X*theta - y) .* X))';
    theta_hist (iter,:) = theta' ;
    
end

end
