
% test manually
load('ex3data1.mat');
lambda = 0.1;
num_labels = 10;
m = size(X, 1);
n = size(X, 2);
all_theta = zeros(num_labels, n+1);
X = [ones(m, 1) X];
y_f = (y==i);
initial_theta = all_theta(1, :)';

% ptions = optimset('GradObj', 'on', 'MaxIter', 50);

% [all_theta] = oneVsAll(X, y, num_labels, lambda)


=====================================================

% test oneVsAll
load('ex3data1.mat');
lambda = 0.1;
num_labels = 10;
[all_theta] = oneVsAll(X, y, num_labels, lambda);
WORKED!!

% test Neural Network Predict

load('ex3data1.mat');
m = size(X, 1);
load('ex3weights.mat');
m = size(X, 1);
num_labels = size(Theta2, 1);
p = zeros(size(X, 1), 1);
X = [ones(m, 1) X];
q = sigmoid(Theta1 * X');
q = [ones(1, m); q];
q = sigmoid(Theta2 * q);
[val, index] = max(q);
p = index';
!!WORKED!!
% p = mod(p, 10);


% Take the row-wise element i from "row" and make p's ith row element
% equal to j = row-wise element from "col"
% p(row(:, 1), 1) = col(:, 1);


% Now replace the 10's from p to 0's 
k = find(p==10);
p(k) = 0;
