function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%
X = [ones(m, 1) X];
q = sigmoid(Theta1 * X');
q = [ones(1, m); q];
q = sigmoid(Theta2 * q);
[val, index] = max(q);
p = index';


% Take the row-wise element i from "row" and make p's ith row element
% equal to j = row-wise element from "col"
% p(row(:, 1), 1) = col(:, 1);


% Now replace the 10's from p to 0's 
% p = mod(p, 10);

% =========================================================================


end
