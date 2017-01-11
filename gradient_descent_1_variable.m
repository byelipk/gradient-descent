setenv('GNUTERM','qt');
graphics_toolkit('gnuplot');

% Our linear regression model


% H(theta, example)
%
%   Hypothesis function for predicting values for linear regression
%
%   Given values for theta and 1 example feature, H(theta, example) predicts
%   an output value y.
function prediction = h(theta, example)
  prediction = theta(1) + theta(2) * example;
end;

% COST(X, y, theta)
%
%   Squared error cost function for linear regression
%
%   Returns a scalar
%
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y.
%
%   Our cost function algorithm (for a single variable) is the sum from
%   i to m of the squared difference between the value predicted by our
%   hypothesis function and the actual value.
%
%   When we perform gradient descent with a single variable we use two
%   cost functions: one for theta0 and one for theta1. This can be refactored
%   by adding prepending a column of ones to our feature set X.
function cost = cost0(theta, X, y, m)
  J = zeros(m, 1);                  % An mx1 column vector.

  for i = [1:m],
    J(i) = (h(theta, X(i)) - y(i)); % Difference between predicted and actual
  end;
  
  cost = (1/m) * sum(J);            % Compute the cost function J
end;

function cost = cost1(theta, X, y, m)
  J = zeros(m, 1);
  for i = [1:m],
    J(i) = (h(theta, X(i)) - y(i)) * X(i);
  end;

  cost = (1/m) * sum(J);
end;


% GRADIENT_DESCENT(theta, alpha, X, y)
%
%   Minimize our cost function J
%
%   The gradient descent algorithm (for a single variable) reduces the value
%   the the cost function J until it finds the local minimum.
%
%   There's a bit of calculus that needs to happen to implement the gradient
%   descent algorithm, so we have to rewrite our cost function.
function T = gradient_descent(theta, alpha, X, y)
  m = size(X, 1);
  T = theta;

  % Simultaneous updates of theta
  T(1) = theta(1) - alpha * cost0(theta, X, y, m);
  T(2) = theta(2) - alpha * cost1(theta, X, y, m);
end;

% A sanity check dataset (3x2 matrix).
data = [
  1 1;
  2 2;
  3 3;
];

X = data(:, 1);  % Features
y = data(:, 2);  % Actual values

% Initialize fitting parameters
theta    = zeros(2, 1);
theta(2) = .5;

alpha    = 0.05; % The learning rate (always a positive number)
iters    = 100;  % Number of steps of gradient descent to take

% For debugging our gradient descent algorithm we can plot out the values for
% theta over time.
for i = [1:iters],
  theta = gradient_descent(theta, alpha, X, y);

  % Plot the value for theta(2)
  plot(i, theta(2), 'b'); hold on;
end;

xlabel('Iterations');
ylabel('Cost Function J');

printf('After %d iterations theta0 = %f and theta1 = %f\n',
  iters, theta(1), theta(2));
