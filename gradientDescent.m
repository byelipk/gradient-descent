function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
  %GRADIENTDESCENT Performs gradient descent to learn theta
  %   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by
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
      %       of the cost function (computeCost) and gradient here.
      %

      % Iterative implementation (Much slower!)
      % T = theta;
      % delta0 = zeros(m, 1);
      % delta1 = zeros(m, 1);
      %
      % for i = [1:m],
      %   delta0(i) = ((theta' * X(i, :)') - y(i));
      % end;
      %
      % for i = [1:m],
      %   delta1(i) = ((theta' * X(i, :)') - y(i)) * X(i, 2)';
      % end;
      %
      % delta0 = (1/m) * sum(delta0);
      % delta1 = (1/m) * sum(delta1);
      %
      % T(1) = theta(1) - alpha * delta0;
      % T(2) = theta(2) - alpha * delta1;
      %
      % theta = T;



      % Vectorized implmentation
      predictions = X * theta;                % Hypothesis function
      diff = predictions - y;                 % Predicted - actual
      diffs = repmat(diff, 1, size(X,2));     % Copy of diffs
      delta = (1 / m) * sum(X.* diffs);       % AKA "cost function" - each
                                              % X(i,j) needs to be multiplied by
                                              % (h(x) - y). We can achieve that
                                              % with matrix multiplication.

      theta = (theta' - (alpha * delta))';    % Reduced theta values

      % ============================================================

      % Save the cost J in every iteration
      J_history(iter) = computeCost(X, y, theta);
  end

end
