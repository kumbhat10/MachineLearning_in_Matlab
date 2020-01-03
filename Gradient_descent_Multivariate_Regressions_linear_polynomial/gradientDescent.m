function [theta, J_history] = gradientDescent(X, y, theta, alpha, iterations)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(iterations, 1);
theta_hist   = zeros(iterations, size(X,2));

for iter = 1:iterations
%     disp(iter)
    J_history(iter) = computeCost(X, y, theta);
    theta = theta - (alpha/m * sum(( X*theta - y) .* X))';
    theta_hist (iter,:) = theta' ;
    
end

end
