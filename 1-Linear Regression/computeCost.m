function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples
J = 0;

% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.

h = X * theta; % predictions of hypothesis
J = sum((h - y).^2)/(2*m);

end
