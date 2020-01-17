function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and
%   sigma. You should complete this function to return the optimal C and
%   sigma based on a cross-validation set.
%
values = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30 ];
iter = 1;
result = zeros(length(values)^2,3);
% You need to return the following variables correctly.

for iC = 1:length(values)
    for isigma = 1:length(values)
%         disp(iter)
        C = values(iC);
        sigma = values(isigma);
        model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
        %predictions done on validation set
        predictions = svmPredict(model, Xval); %   mean(double(predictions ~= yval))
        result(iter,:) = [C sigma mean(double(predictions ~= yval))];
        iter = iter +1;
    end
end
[~,indexMinError] = min(result(:,3));
C     = result(indexMinError,1);
sigma = result(indexMinError,2);
% disp(length(X))
% disp(length(Xval))
end
