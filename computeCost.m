function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y);      % Number of training examples

% You need to return the following variables correctly
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.

function prediction = h(theta, example)
  prediction = theta' * example;
end;

function J = cost(theta, X, y)
  J = zeros(m, 1);

  for i = [1:m],
    J(i) = (h(theta, X(i, :)') - y(i)).^2;
  end;
end;

J = (1 / (2 * m)) * sum(cost(theta, X, y));


% =========================================================================

end
