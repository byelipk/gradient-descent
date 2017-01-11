% Non vectorized hypothesis function
%
% Sum the product of theta(j) and feature(j) of example(i).
% Return an mx1 column vector.
%
% h(x) = SUM ( theta_j * x_j )
function predictions = h_non_vectorized(theta, X)
  m = size(X, 1);
  n = size(X, 2);
  predictions = zeros(m, 1);
  for i = [1:m],
    for j = [1:n],
      predictions(i) = predictions(i) + theta(j) * X(i, j);
    end;
  end;
end;

function predictions = h_vectorized_1(theta, X)
  s = size(X);
  m = s(1);
  n = s(2);
  predictions = zeros(m, 1);
  for i = [1:m],
    % The training example must be in the form of a column vector, so
    % we're taking the transpose of the ith training example.
    x = X(i, :)';
    predictions(i) = theta' * x;
  end;
end;

% Vectorized hypothesis function
%
% Perform matrix multiplication
function predictions = h_vectorized_2(theta, X)
  predictions = X * theta;
end;
