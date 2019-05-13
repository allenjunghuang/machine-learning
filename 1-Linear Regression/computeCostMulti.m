function J = computeCostMulti(X, y, theta)
%gradientDescentMulti Compute cost for linear regression with multiple variables
%   J = gradientDescentMulti(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples
J = 0;

% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.

pred = X * theta; % predictions of hypothesis
J = sum((pred - y).^2)/(2*m); %mean square error


end
