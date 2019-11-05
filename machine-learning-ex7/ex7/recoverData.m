function X_rec = recoverData(Z, U, K)
%RECOVERDATA Recovers an approximation of the original data when using the 
%projected data
%   X_rec = RECOVERDATA(Z, U, K) recovers an approximation the 
%   original data that has been reduced to K dimensions. It returns the
%   approximate reconstruction in X_rec.
%

% You need to return the following variables correctly.
X_rec = zeros(size(Z, 1), size(U, 1));
% Z = [m x K]
% U = [n x n] out of which fist K columns are to be used
% in reconstructing X_rec so U_reduce = [n x K]
% x_i = [1 x K] is obtained by U * z_i
% where U = [n x K] and z_i = [K x 1]
% X_rec = [m x n]

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the approximation of the data by projecting back
%               onto the original space using the top K eigenvectors in U.
%
%               For the i-th example Z(i,:), the (approximate)
%               recovered data for dimension j is given as follows:
%                    v = Z(i, :)';
%                    recovered_j = v' * U(j, 1:K)';
%
%               Notice that U(j, 1:K) is a row vector.
%               
X_rec = Z * U(:, 1:K)';


% ==========================)===================================

end
