function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta.
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCostMulti) and gradient here.
    %




    % Vectorized implmentation
    predictions = X * theta;                % Hypothesis function
    diff = predictions - y;                 % Predicted - actual
    diffs = repmat(diff, 1, size(X,2));     % Copy of diffs
    delta = (1 / m) * sum(X.* diffs);       % AKA "cost function" - each
                                            % X(i,j) needs to be multiplied by
                                            % (h(x) - y). We can achieve that
                                            % with element-wise matrix multiplication.

    theta = (theta' - (alpha * delta))';    % Reduced theta values






    % ============================================================

    % Save the cost J in every iteration
    J_history(iter) = computeCostMulti(X, y, theta);

end

end
