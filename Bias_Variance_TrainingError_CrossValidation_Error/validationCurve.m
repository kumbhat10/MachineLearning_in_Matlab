function [lambda_vec, error_train, error_val] = validationCurve(X, y, Xval, yval)
%VALIDATIONCURVE Generate the train and validation errors needed to
%   [lambda_vec, error_train, error_val] = ...
%       VALIDATIONCURVE(X, y, Xval, yval) returns the train
%       and validation errors (in error_train, error_val)
%       for different values of lambda.  Given are the training set (X,
%       y) and validation set (Xval, yval).
lambda_vec = [0 0.001 0.003 0.01 0.03 0.1 0.3 1 3 10]';

error_train = zeros(length(lambda_vec), 1);
error_val = zeros(length(lambda_vec), 1);

for i = 1:length(lambda_vec)
    [theta]          = trainLinearReg( X, y, lambda_vec(i));
    error_train(i,1) = linearRegCostFunction( X   , y   , theta, 0);
    error_val(i,1)   = linearRegCostFunction( Xval, yval, theta, 0);  
end
