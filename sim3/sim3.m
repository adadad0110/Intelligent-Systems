clc; clear; close;

% 參數
inputsz = 2;
hiddensz =20;
outputsz = 1;
learningrate = 0.5;
alpha = 0.99;
iterations = 200000;
train_n = 300;
test_n = 100;
total_n = train_n + test_n;

% 資料生成
x_all = -0.8 + 1.5 * rand(total_n, inputsz);  % [-0.8, 0.7]
x = x_all(:,1);
y = x_all(:,2);
y_all = 5 * sin(pi * (x.^2)) .* sin(2 * pi * y) + 1;

% 分割 train/test
x_train = x_all(1:train_n, :);
x_test  = x_all(train_n+1:end, :);
y_train = y_all(1:train_n, :);
y_test  = y_all(train_n+1:end, :);

% Normalization
min_y = min(y_all); max_y = max(y_all);
y_train = ((y_train - min_y) / (max_y - min_y)) * 0.6 + 0.2;
y_test  = ((y_test  - min_y) / (max_y - min_y)) * 0.6 + 0.2;

% 初始化
W_input_hidden = -0.3 + 0.6 * rand(inputsz, hiddensz);
W_hidden_output = -0.3 + 0.6 * rand(hiddensz, outputsz);
bias_hidden = -0.3 + 0.6 * rand(1, hiddensz);
bias_output = -0.3 + 0.6 * rand(1, outputsz);

dW_input_hidden = zeros(size(W_input_hidden));
dW_hidden_output = zeros(size(W_hidden_output));
db_hidden = zeros(size(bias_hidden));
db_output = zeros(size(bias_output));

mse_history = zeros(1, iterations);

activation = @(x) tanh(x);
activation_d = @(x) 1 - tanh(x).^2;

% 訓練
for epoch = 1:iterations
    v_hidden = x_train * W_input_hidden + bias_hidden;
    y_hidden = activation(v_hidden);

    v_output = y_hidden * W_hidden_output + bias_output;
    y_output = activation(v_output);

    error = y_train - y_output;
    mse_history(epoch) = mean(0.5 * error.^2);

    delta_output = error .* activation_d(v_output);
    delta_hidden = (delta_output * W_hidden_output') .* activation_d(v_hidden);

    grad_W_output = y_hidden' * delta_output / train_n;
    grad_W_hidden = x_train' * delta_hidden / train_n;
    grad_b_output = sum(delta_output, 1) / train_n;
    grad_b_hidden = sum(delta_hidden, 1) / train_n;

    W_hidden_output = W_hidden_output + learningrate * grad_W_output + alpha * dW_hidden_output;
    W_input_hidden  = W_input_hidden  + learningrate * grad_W_hidden + alpha * dW_input_hidden;
    bias_output     = bias_output     + learningrate * grad_b_output + alpha * db_output;
    bias_hidden     = bias_hidden     + learningrate * grad_b_hidden + alpha * db_hidden;

    dW_hidden_output = learningrate * grad_W_output;
    dW_input_hidden  = learningrate * grad_W_hidden;
    db_output = learningrate * grad_b_output;
    db_hidden = learningrate * grad_b_hidden;

    if mod(epoch, 1000) == 0
        fprintf('Epoch %d, Error: %.10f\n', epoch, mse_history(epoch));
    end
end

% 測試
v_hidden_test = x_test * W_input_hidden + bias_hidden;
y_hidden_test = activation(v_hidden_test);
v_output_test = y_hidden_test * W_hidden_output + bias_output;
y_output_test = activation(v_output_test);

mse_test = mean(0.5 * (y_test - y_output_test).^2);
fprintf('E_train: %.10f, E_test: %.10f\n', mse_history(end), mse_test);

% 畫圖
figure;
subplot(2,2,1);
plot(1:iterations, mse_history);
title('Converge');

[x1_grid, x2_grid] = meshgrid(linspace(-0.8, 0.7, 100), linspace(-0.8, 0.7, 100));
y_desired = 5 * sin(pi * (x1_grid.^2)) .* sin(2 * pi * x2_grid) + 1;
subplot(2,2,2);
mesh(x1_grid, x2_grid, y_desired);
title('Desired Output');

% Train output
y_train_pred = zeros(size(x1_grid));
for i = 1:size(x1_grid, 2)
    x_pair = [x1_grid(:,i), x2_grid(:,i)];
    v_hidden = x_pair * W_input_hidden + bias_hidden;
    y_hidden = activation(v_hidden);
    v_output = y_hidden * W_hidden_output + bias_output;
    y_out = activation(v_output);
    y_train_pred(:,i) = ((y_out - 0.2) / 0.6) * (max_y - min_y) + min_y;
end
subplot(2,2,3);
mesh(x1_grid, x2_grid, y_train_pred);
title('Train Output');

% Test output
y_test_pred = zeros(size(x1_grid));
for i = 1:size(x1_grid, 2)
    x_pair = [x1_grid(:,i), x2_grid(:,i)];
    v_hidden = x_pair * W_input_hidden + bias_hidden;
    y_hidden = activation(v_hidden);
    v_output = y_hidden * W_hidden_output + bias_output;
    y_out = activation(v_output);
    y_test_pred(:,i) = ((y_out - 0.2) / 0.6) * (max_y - min_y) + min_y;
end
subplot(2,2,4);
mesh(x1_grid, x2_grid, y_test_pred);
title('Test Output');
