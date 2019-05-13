function [theta, J] = gradientDescentMulti(X, y, theta, alpha, iterations)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, iterations) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values

m = length(y); % number of training examples
J = zeros(iterations, 1);

for i = 1:iterations

    % Instructions: Perform a single gradient step on the parameter vector theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCostMulti) and gradient here.
    
    theta = theta - alpha * (1/m) * ((X*theta - y)'*X)';
    J(i) = computeCostMulti(X, y, theta); % save the cost J in every iteration

end

end
