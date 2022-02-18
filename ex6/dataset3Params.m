function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 0.01;
sigma = 0.01;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%
C_test = [0.01;0.03;0.1;0.3;1;3;10;30];
Sigma_test = [0.01;0.03;0.1;0.3;1;3;10;30];
m = size(C_test, 1);
Value = zeros(m^2,1);

for i = 1:m
  for j = 1:m
    C = C_test(i);
    sigma = Sigma_test(j);
    model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma)); 
    predictions = svmPredict(model, Xval);
    Value((i-1)*m+j) = mean(double(predictions ~= yval));
  end
end
% 这里的整数判定很奇怪

[minValue, minIndex] = min(Value);
index1 = floor(minIndex / m);
index2 = minIndex - (index1) * m;
C = C_test(index1);
sigma = Sigma_test(index2);

% =========================================================================

end
