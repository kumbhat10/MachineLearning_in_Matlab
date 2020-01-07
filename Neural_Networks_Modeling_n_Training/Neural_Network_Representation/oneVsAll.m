function [all_theta] = oneVsAll(X, y, num_labels, lambda)
% Some useful variables
m = size(X, 1);
n = size(X, 2);

all_theta     = zeros(num_labels, n + 1);
X             = [ones(m, 1) X];
initial_theta = zeros(n + 1, 1);

for iter = 1:num_labels
    options = optimset('GradObj', 'on', 'MaxIter', 50);
    all_theta(iter,:)= ...
        fmincg(@(t)(lrCostFunction(t, X, double(y == iter), lambda)), initial_theta, options )';
end

end
