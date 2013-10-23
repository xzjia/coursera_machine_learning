clear ; close all; clc

fprintf('Loading data ...\n');

%% Load Data
data = load('ex1data2.txt');
X = data(:, 1:2);
y = data(:, 3);
m = length(y);

fprintf('Normalizing Features ...\n');

[X mu sigma] = featureNormalize(X);

% Add intercept term to X
X = [ones(m, 1) X];

fprintf('Running gradient descent ...\n');

% Choose some alpha value
alpha = 0.3;
num_iters = 100;

% Init Theta and Run Gradient Descent 
theta = zeros(3, 1);
%for i = 1:length(alpha)
[theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters);
% Plot the convergence graph
figure;

plot(1:numel(J_history), J_history, '-b', 'LineWidth', 2);
xlabel('Number of iterations');
ylabel('Cost J');
fprintf('Theta computed from gradient descent: \n');
fprintf(' %f \n', theta);
fprintf('\n');


testCase = [1 1650 3];
price = testCase * theta; % You should change this

fprintf(['Predicted price of a 1650 sq-ft, 3 br house ' ...
         '(using gradient descent):\n $%f\n'], price);

