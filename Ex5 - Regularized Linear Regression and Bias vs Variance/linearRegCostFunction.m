function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%               You should set J to the cost and grad to the gradient.
%
hypothesis = X*theta;
J = (1/(2*m))*sum((hypothesis - y).^2) + lambda/(2*m)*sum(theta(2:end).^2);
grad = (1/m)*((hypothesis - y)'*X)' + (lambda/m).*theta.*[0; ones(length(theta)-1, 1)];
% =========================================================================

grad = grad(:);
% For Debugging
% size(X')
% size((hypothesis - y))
% size(X'*(hypothesis - y))
end
