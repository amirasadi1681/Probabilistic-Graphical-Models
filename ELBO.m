% Load libraries
clc;
clear;
close all;

% Load the image
data = imread('PGM.jpg');
img = double(data); % Converting the image from uint8 to double
img_mean = mean(img(:)); % Finding the mean value of pixels
img_binary = 2*(img > img_mean) - 1; % Mapping each pixel to either -1 or +1 (Binary Classification)
[M, N] = size(img_binary); % Extracting the dimensions of the image

% Mean-field parameters
sigma  = 1; % Noise power spectral density
y = img_binary + sigma*randn(M, N); % Adding the noise: y_i ~ N(x_i; sigma^2);
J = 1; % Coupling strength (w_ij)
rate = 0.5; % Update smoothing rate
max_iter = 30; % Maximum Iteration
ELBO = zeros(1, max_iter); % Memorizing ELBO values at each step for later plotting
Hx_mean = zeros(1, max_iter); % Memorizing Entropy values at each step for later plotting

% Plotting the noisy image
figure;
imshow(y, []);
title('Observed noisy image');
imwrite(y, 'PGM_class_plus_noise.png');

% Mean-Field Variational Inference
logodds = log(mvnpdf(y(:), 1, sigma^2)) - log(mvnpdf(y(:), -1, sigma^2));
logodds = reshape(logodds, [M, N]);

% Initializing parameters
p1 = sigmoid(logodds);
mu = 2 * p1 - 1; % mu_init

a = mu + 0.5 * logodds;
qxp1 = sigmoid(2 * a); % q_i(x_i=+1)
qxm1 = sigmoid(-2 * a); % q_i(x_i=-1)

logp1 = reshape(log(mvnpdf(y(:), 1, sigma^2)), [M, N]);
logm1 = reshape(log(mvnpdf(y(:), -1, sigma^2)), [M, N]);

for i = 1:max_iter
    muNew = mu;
    for ix = 1:N
        for iy = 1:M
            pos = iy + M*(ix-1);
            neighborhood = pos + [-1, 1, -M, M];
            boundary_idx = [iy~=1, iy~=M, ix~=1, ix~=N];
            neighborhood = neighborhood(boundary_idx);
            [xx, yy] = ind2sub([M, N], pos);
            [nx, ny] = ind2sub([M, N], neighborhood);
            
            Sbar = J * sum(mu(sub2ind([M, N], nx, ny)));
            muNew(xx, yy) = (1-rate) * muNew(xx, yy) + rate * tanh(Sbar + 0.5 * logodds(xx, yy));
            ELBO(i) = ELBO(i) + 0.5 * (Sbar * muNew(xx, yy));
        end
    end
    mu = muNew;
    
    a = mu + 0.5 * logodds;
    qxp1 = sigmoid(2 * a); % q_i(x_i=+1)
    qxm1 = sigmoid(-2 * a); % q_i(x_i=-1)
    Hx = -qxm1 .* log(qxm1 + 1e-10) - qxp1 .* log(qxp1 + 1e-10); % Entropy
    
    ELBO(i) = ELBO(i) + sum(qxp1(:) .* logp1(:) + qxm1(:) .* logm1(:)) + sum(Hx(:));
    Hx_mean(i) = mean(Hx(:));
end

% Plotting the denoised image
figure;
imshow(mu, []);
title(sprintf('After %d mean-field iterations', max_iter));
imwrite(mu, 'PGM_ising_denoised.png');

% Plotting the ELBO objective function value of Ising model
figure;
plot(1:max_iter, ELBO, 'b', 'LineWidth', 2);
title('Variational Inference for Ising Model');
xlabel('Iterations');
ylabel('ELBO objective');
legend('ELBO', 'Location', 'NorthEast');
saveas(gcf, 'PGM_VarInfer_ELBO.png');

% Plotting the Average Entropy
figure;
plot(1:max_iter, Hx_mean, 'b', 'LineWidth', 2);
title('Variational Inference for Ising Model');
xlabel('Iterations');
ylabel('Average Entropy');
legend('Avg Entropy', 'Location', 'NorthEast');
saveas(gcf, 'PGM_AvgEnt.png');

function y = sigmoid(x)
    y = 1 ./ (1 + exp(-x));
end

function p = mvnpdf(x, mu, sigma2)
    p = (1 / sqrt(2 * pi * sigma2)) * exp(-0.5 * (x - mu).^2 / sigma2);
end


