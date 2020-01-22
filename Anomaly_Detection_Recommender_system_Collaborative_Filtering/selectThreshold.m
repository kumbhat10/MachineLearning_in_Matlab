function [bestEpsilon ,bestF1] = selectThreshold(yval, pval)
%SELECTTHRESHOLD Find the best threshold (epsilon) to use for selecting
%outliers
%   [bestEpsilon bestF1] = SELECTTHRESHOLD(yval, pval) finds the best
%   threshold to use for selecting outliers based on the results from a
%   validation set (pval) and the ground truth (yval).
%

bestEpsilon = 0;
bestF1 = 0;
% F1 = 0;

stepsize = (max(pval) - min(pval)) / 1000;
for epsilon = min(pval):stepsize:max(pval)
    predictions = (pval < epsilon);
    
    tp = sum(yval==1 & predictions==1 ); % true positive
    fp = sum(yval==0 & predictions==1 ); % false positive
    fn = sum(yval==1 & predictions==0 ); % false negative
    
    precision = tp / (tp+ fp);    
    recall = tp / (tp + fn) ;
    F1 = 2* precision * recall / (precision + recall ) ;
    
    if F1 > bestF1
       bestF1 = F1;
       bestEpsilon = epsilon;
    end
end

end
