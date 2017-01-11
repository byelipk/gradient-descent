% A cost function is used to measure the accuracy of our hypothesis function.

data = load('ex1data1.txt');
X = data(:, 1); % Features
y = data(:, 2); % Targets

m = size(X, 1); % Number of training examples
n = size(X, 2); % Number of features

X = [ones(m, 1), data(:,1)];  % Add a column of ones to x
theta = zeros(2, 1);          % Initialize fitting parameters

function prediction = h(theta, example)
  prediction = theta(1) + theta(2) * example;
end;

function cost = cost(theta, X, y, m)
  J = zeros(m, 1);
  for i = [1:m],
    J(i) = (h(theta, X(i)) - y(i)) * X(i);
  end;

  cost = (1/m) * sum(J);
end;

cost(theta, X, y, m)
