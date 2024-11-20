% Parameters
T = 1;               
mu = 0.1;             
sigma = 0.2;          % Volatility coefficient
S0 = 1;               % Initial value of the process
N_values = [20, 50, 100,1000]; % Different time step counts for approximations
N_exact = 1000;        % Fixed time step count for exact solution
N_max = max([N_values, N_exact]); % Maximum N for Brownian motion simulation

% Simulate Brownian motion at high resolution
dt_max = T / N_max;                % Time step size for maximum N
dW_max = sqrt(dt_max) * randn(1, N_max); % Increments of Wiener process
W_max = [0, cumsum(dW_max)];       % Wiener process

% Exact solution at N_exact time steps
idx_exact = round(linspace(1, N_max+1, N_exact+1));
t_exact = linspace(0, T, N_exact+1);      % Time vector for exact solution
W_exact = W_max(idx_exact);               % Brownian motion at exact time steps
S_exact = S0 * exp((mu - 0.5 * sigma^2) * t_exact + sigma * W_exact);


% Preallocate error storage
absolute_errors_em = zeros(length(N_values), 1);
absolute_errors_milstein = zeros(length(N_values), 1);
rmse_errors_em = zeros(length(N_values), 1);
rmse_errors_milstein = zeros(length(N_values), 1);


% Loop through different N values for approximations
for idx = 1:length(N_values)
    N = N_values(idx);
    dt = T / N;                   % Time step size for approximation
    t = linspace(0, T, N+1);      % Time vector for approximation
    
    % Get indices for approximation time steps
    idx_approx = round(linspace(1, N_max+1, N+1));
    W_approx = W_max(idx_approx);              % Brownian motion at approximation steps
    dW = diff(W_approx);                       % Increments for approximation
    
    % Preallocate arrays
    S_em = zeros(1, N+1);         % Euler-Maruyama approximation
    S_milstein = zeros(1, N+1);   % Milstein approximation
    
    % Initial condition
    S_em(1) = S0;
    S_milstein(1) = S0;
    
    % Compute Euler-Maruyama and Milstein approximations
    for i = 1:N
        % Euler-Maruyama approximation
        S_em(i+1) = S_em(i) + mu * S_em(i) * dt + sigma * S_em(i) * dW(i);
        
        % Milstein approximation
        S_milstein(i+1) = S_milstein(i) + mu * S_milstein(i) * dt + sigma * S_milstein(i) * dW(i) ...
                          + 0.5 * sigma^2 * S_milstein(i) * (dW(i)^2 - dt);
    end
    
    % Interpolate exact solution to match approximation time steps
    S_exact_interp = interp1(t_exact, S_exact, t, 'linear');
   
 % Compute maximum absolute error
    absolute_errors_em(idx) = max(abs(S_em - S_exact_interp));
    absolute_errors_milstein(idx) = max(abs(S_milstein - S_exact_interp));

   % Compute RMSE
    rmse_errors_em(idx) = sqrt(mean((S_em - S_exact_interp).^2));
    rmse_errors_milstein(idx) = sqrt(mean((S_milstein - S_exact_interp).^2));

    % Print errors for this N
    fprintf('N = %d: Max Error (EM) = %.5f, Max Error (Milstein) = %.5f\n', ...
            N, absolute_errors_em(idx), absolute_errors_milstein(idx));
    fprintf('N = %d: RMSE (EM) = %.5f, RMSE (Milstein) = %.5f\n', ...
            N, rmse_errors_em(idx), rmse_errors_milstein(idx));



    % Plotting results
    figure;
    plot(t_exact, S_exact, 'b-', 'LineWidth', 0.7); hold on;
    plot(t, S_em, 'r--', 'LineWidth', 1.25);
    plot(t, S_milstein, 'g-.', 'LineWidth', 1.25);
    legend('Exact Solution (N=1000)', 'Euler-Maruyama', 'Milstein', 'Location', 'Best');
    xlabel('Time');
    ylabel('S(t)');
    title(sprintf('Geometric Brownian Motion Approximations (N = %d)', N));
    grid on;
end


% Plot errors against N values
figure;
loglog(N_values, absolute_errors_em, 'r-o', 'LineWidth', 1.5); hold on;
loglog(N_values, absolute_errors_milstein, 'g-s', 'LineWidth', 1.5);
loglog(N_values, rmse_errors_em, 'm-^', 'LineWidth', 1.5);
loglog(N_values, rmse_errors_milstein, 'c-v', 'LineWidth', 1.5);
legend('Max Error (Euler-Maruyama)', 'Max Error (Milstein)', ...
       'RMSE (Euler-Maruyama)', 'RMSE (Milstein)', 'Location', 'Best');
xlabel('Number of Time Steps (N) (log scale)');
ylabel('Error (log scale)');
title('Error vs Time Step Count');
grid on;
