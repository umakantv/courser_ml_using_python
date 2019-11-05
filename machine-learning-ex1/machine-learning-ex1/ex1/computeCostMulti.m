function J = computeCostMulti(X, y, theta)
%COMPUTECOSTMULTI Compute cost for linear regression with multiple variables
%   J = COMPUTECOSTMULTI(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.


h_theta_x = X * theta;
% I now have a row vector that has h_theta_x for all the rows
% Now I'll subtract the row wise y
sum_error = sum(sum(((h_theta_x - y).^2)));
J = (1/(2*m)) * sum_error;




% =========================================================================

end
