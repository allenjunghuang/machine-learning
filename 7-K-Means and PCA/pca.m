function [U, S] = pca(X)
%PCA Run principal component analysis on the dataset X
%   [U, S] = pca(X) computes eigenvectors of the covariance matrix of X
%   Returns the eigenvectors U, the eigenvalues (on diagonal) in S
%

[m, n] = size(X);

U = zeros(n); % eigenvectors
S = zeros(n); % eigenvalues

% Instructions: You should first compute the covariance matrix. Then, you
%               should use the SVD function to compute the eigenvectors
%               and eigenvalues of the covariance matrix. 
%
% Note: When computing the covariance matrix, remember to divide by m (the
%       number of examples).

C = (1/m)*X'*X;
[U, S, V] = svd(C);


end
