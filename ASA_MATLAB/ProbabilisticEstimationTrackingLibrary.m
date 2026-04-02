classdef ProbabilisticEstimationTrackingLibrary
    % Class for Probabilistic Estimation Tracking Library functions and methods
    methods

    %% 1) Approximations And Estimation Methods
        % Measurements Simulator with Gaussian Noise 
            function noisy_signal = simulateMeasurementsGaussianNoise(obj, x_real, Htilde, Rk, m, nz, tspan, tspan_meas)
                    % simMeasurementsGaussianNoise: Simulates noisy measurements by 
                    % mapping the true state to measurements using Htilde and adding Gaussian noise.
                    %
                    % This function generates noisy measurements from a true state by first mapping
                    % the state through the measurement matrix Htilde and then adding Gaussian noise 
                    % based on the provided covariance matrix Rk and mean m.
                    %
                    % Inputs:
                    %   x_real       - The true state (matrix with dimensions [num_samples, num_dims])
                    %                  Each column represents one dimension (e.g., x, y, z).
                    %                  NOTE: Must include all corresponding state elements history!
                    %   Htilde       - Measurement matrix mapping true state to measured state (num_meas_dims x num_state_dims)
                    %   Rk           - Noise covariance matrix (num_meas_dims x num_meas_dims), describing the 
                    %                  variance and correlation of the noise in measurements.
                    %   m            - Mean of the noise (1 x num_meas_dims or num_meas_dims x 1 vector)
                    %   nz           - Measured state dimensions (how many variables are we measuring?)
                    %   tspan        - Time span of the original signal (1D array), assumed to have
                    %                  more frequent measurements (e.g., every 1 second).
                    %   tspan_meas   - Time span of the measurements (1D array), representing
                    %                  the times at which noise-corrupted measurements are taken 
                    %                  (e.g., every 30 seconds: t0:dt:tf).
                    %
                    % Outputs:
                    %   noisy_signal - The signal corrupted with Gaussian noise at the given 
                    %                  measurement times. Has the same number of rows as 
                    %                  tspan_meas and the same number of columns as the measured signal.
                    %
                    % Example:
                    %   % Simulate noisy 2D position measurements
                    %   t0 = 0;
                    %   tf = 600;                  % Total time in seconds
                    %   tspan = t0:1:tf;           % Time span with 1 second intervals
                    %   tspan_meas = t0:30:tf;     % Measurements taken every 30 seconds
                    %   x_real = [100 * sin(0.01 * tspan); 50 * cos(0.01 * tspan)]'; % True state
                    %   Htilde = [1 0; 0 1];       % Measurement matrix mapping state to x, y position
                    %   Rk = diag([4 4]);          % Noise covariance matrix for x and y
                    %   m = [0 0];                 % Zero mean for the noise
                    %   noisy_signal = simulateMeasurementswithGaussianNoise(x_real, Htilde, Rk, m, tspan, tspan_meas);
                    %
                    % Notes:
                    % - Ensure that the dimensions of Rk, Htilde, and m match the dimensions of the measurements.
                    % - The Cholesky decomposition is used to create correlated Gaussian noise.
                    
                    % Number of measurements and dimensions
                    num_meas = length(tspan_meas);         
                
                    % Initialize noisy signal array (same size as the number of measurements and dimensions)
                    noisy_signal = zeros(num_meas, nz);
                
                    % Find the indices in tspan corresponding to the measurement times
                    [~, meas_indices] = ismember(tspan_meas, tspan);
                
                    % Cholesky decomposition of the noise covariance matrix (lower triangular matrix L)
                    L = chol(Rk, 'lower');
                    
                    % Loop through each time step to generate noise and corrupt the true signal
                    for i = 1:num_meas
                        % Map the true state to measurement space using Htilde
                        mapped_state = Htilde * x_real(meas_indices(i), :)';
                        
                        % Generate Gaussian noise using randn and Cholesky decomposition
                        noise = L * randn(nz, 1) + m(:);  % Generates noise vector for the measurements
                        
                        % Corrupt the mapped state with Gaussian noise at the measurement times
                        noisy_signal(i, :) = (mapped_state + noise)';
                    end
            end

        %% 1.1) Least Squares Based Estimation Methods
            % LSQ Unweighted  
                function [a] = leastSquaresCoefficients(obj, H, z)
                    % Computes the least squares solution to the system H * a = z.
                    % The goal is to find the coefficients 'a' that minimize the error 
                    % in approximating z using the measurement mapping matrix H.
                    %
                    % Input:
                    %   H - H Matrix mapping the observations or math model to the approximation
                    %   z - Vector of observed values (e.g., X, Y, or Z position)
                    %
                    % Output:
                    %   a - Least squares solution, i.e., the vector of coefficients
                
                    % Solve the normal equation using the pseudo-inverse approach
                    a = (H' * H) \ (H' * z);
                end
            
            % Recursive Least Squares
                function [xhat_updated, P_updated, xf_hat, Pf_hat] = RLS_recursiveLeastSquares(obj, x0hat, P0, z_hist, tspan_meas, Rk, Hk_handle, Phi_handle, nx)
                    % RLS_recursiveLeastSquares: Performs the Recursive Least Squares (RLS) algorithm to update
                    % state estimates and covariances based on new measurements in a linear discrete-time system.
                    %
                    % This function implements the RLS algorithm for a generic linear system, allowing for custom
                    % state transition matrices and measurement matrices. It updates the state estimate and covariance
                    % recursively as new measurements become available.
                    %
                    % Inputs:
                    %   x0hat       - Initial state estimate at time t0 (nx x 1 vector)
                    %   P0          - Initial covariance estimate at time t0 (nx x nx matrix)
                    %   z_hist      - Measurement history matrix (num_meas x nz), where each row is the measurement vector at time t_k
                    %   tspan_meas  - Vector of measurement times (1 x num_meas)
                    %   Rk          - Measurement noise covariance matrix (nz x nz), assumed constant or can be a function handle
                    %   Hk_handle   - Function handle to compute the measurement matrix Htil at time t_k (function of t_k)
                    %                 Usage: H_k = Hk_handle(t_k). This could also be a
                    %                 constant value. 
                    %   Phi_handle  - Function handle to compute the state transition matrix Phi_k from t_k-1 to t_k
                    %                 Usage: Phi_k = Phi_handle(t_k, t_k_minus_1)
                    %   nx          - Dimension of the state vector
                    %
                    % Outputs:
                    %   xhat_updated - Updated state estimates over all measurements (nx x num_meas)
                    %   P_updated    - Updated covariance matrices over all measurements (nx x nx x num_meas)
                    %   xf_hat       - Final state estimate at the last measurement time (nx x 1 vector)
                    %   Pf_hat       - Final covariance matrix at the last measurement time (nx x nx matrix)
                    %
                    % Example usage:
                    %   [xhat_updated, P_updated, xf_hat, Pf_hat] = obj.RLS_recursiveLeastSquares(x0hat, P0, z_hist, tspan_meas, Rk, Hk_handle, Phi_handle, nx);
                    %
                    % Notes:
                    %   - The system is assumed to be linear and can have time-varying dynamics and measurement matrices.
                    %   - Measurement noise can be time-varying if Rk is provided as a function handle.
                    %   - The function handles Hk_handle and Phi_handle allow for flexibility in specifying system dynamics.
                    
                        % Number of measurements
                        num_meas = size(z_hist, 1);  % Number of measurements
                        nz = size(z_hist, 2);        % Dimension of each measurement
                    
                        % Initialize arrays for storing updated state estimates and covariances
                        xhat_updated = zeros(nx, num_meas);       % State estimates
                        P_updated = zeros(nx, nx, num_meas);      % Covariance matrices
                    
                        % Set initial state estimate and covariance
                        xhat_updated(:, 1) = x0hat;               % Initial state estimate at t0
                        P_updated(:, :, 1) = P0;                  % Initial covariance at t0
                    
                        % Loop through each measurement starting from the second one
                        for k = 2:num_meas
                            % Time indices
                            t_k = tspan_meas(k);
                            t_k_minus_1 = tspan_meas(k - 1);
                    
                            % State Transition Matrix Phi_k from t_k_minus_1 to t_k
                            Phi_k = Phi_handle(t_k, t_k_minus_1);  % Function handle to compute Phi_k
                    
                            % Step 1: Prediction using STM
                            xk_bar = Phi_k * xhat_updated(:, k - 1);                       % Predicted state estimate
                            Pk_bar = Phi_k * P_updated(:, :, k - 1) * Phi_k';              % Predicted covariance estimate
                    
                            % Measurement matrix H_k at time t_k
                            H_k = Hk_handle(t_k);  % Function handle to compute H_k
                    
                            % Measurement noise covariance Rk_k at time t_k
                            if isa(Rk, 'function_handle')
                                Rk_k = Rk(t_k);    % If Rk is a function handle, compute Rk_k at time t_k
                            else
                                Rk_k = Rk;         % Else, use the constant Rk provided
                            end
                    
                            % Step 2: Kalman Gain
                            % Kalman Gain computation
                            S_k = H_k * Pk_bar * H_k' + Rk_k;      % Innovation covariance
                            Kk = Pk_bar * H_k' / S_k;              % Kalman Gain
                    
                            % Measurement at time t_k
                            z_k = z_hist(k, :)';                   % Measurement vector at time t_k
                    
                            % Step 3: Update the state and covariance
                            xk_hat = xk_bar + Kk * (z_k - H_k * xk_bar);                   % Updated state estimate
                            Pk_hat = (eye(nx) - Kk * H_k) * Pk_bar;                        % Updated covariance estimate
                    
                            % Store the updated estimates
                            xhat_updated(:, k) = xk_hat;
                            P_updated(:, :, k) = Pk_hat;
                        end
                    
                        % Final state estimate and covariance
                        xf_hat = xhat_updated(:, end);            % Final state estimate at the last measurement time
                        Pf_hat = P_updated(:, :, end);            % Final covariance matrix at the last measurement time
                end
            
            % Chebyshev Polynomial Approximation Function
                function f_approx = chebyshev_approximation(obj, t, a, t1, t2)
                    % Computes the Chebyshev polynomial approximation f(t) using the 
                    % provided coefficients a and time vector t (Julian Dates).
                    %
                    % Input:
                    %   t   - Time vector (Julian Dates)
                    %   a   - Coefficients vector (a_i values)
                    %   t1  - Start time (initial Julian Date)
                    %   t2  - End time (final Julian Date)
                    %
                    % Output:
                    %   f_approx - Approximated function values for each time in t
                    
                    m = length(t);          % Number of time steps
                    f_approx = zeros(m, 1); % Initialize the output vector for f(t)
                    n = length(a);          % The order of the approximation is determined by the length of the coefficient vector 'a'
                    
                    for i = 1:m % for each timestep
                        % Calculate tau for the current time step
                        tau = 2 * (t(i) - t1) / (t2 - t1) - 1;   
                        
                        % Initialize the Chebyshev polynomials for the current time step T0
                        % and T1 are fixed
                        T = zeros(1, n); 
                        T(1) = 1;       % T_0(τ) = 1
                        if n > 1
                            T(2) = tau; % T_1(τ) = τ
                        end
                        
                        % Generate the remaining Chebyshev polynomials using the recurrence relation
                        for j = 3:n
                            T(j) = 2 * tau * T(j-1) - T(j-2);
                        end
                        
                        % Perform the summation to calculate f(t) for the current time
                        f_approx(i) = a' * T';  % Perform dot product between a and T
                
                    end
                end
            
            % Chebyshev Polynomial H Matrix 
                function [H] = chebyshev_H(obj, n, t)
                    
                    m = length(t);          % m : number of timesteps 
                    H = zeros(m,n);                                            
                    t1 = t(1);              % t1 : initial time 
                    t2 = t(end);            % t2 : final time 
                    
                    for i = 1:m % for each timestep produce a row of Tis and stack them 
                        % Calculate tau for this timestep
                        tau = 2*(t(i)-t1)/(t2-t1) - 1;   
                        
                        % Initialize time vector for this timestep
                        T = zeros(1,n); 
                        T(1) = 1;       % preset T_0       
                        if n > 1
                            T(2) = tau;     % preset T_1
                        end
                        
                        % Generate the remaining Chebyshev polynomials based on the degree
                        % required
                        for j = 3:n
                            T(j) = 2 * tau * T(j-1) - T(j-2); % Recursive formula for Chebyshev polynomials
                        end
                    
                        % Add the computed row of polynomials to the H matrix
                        H(i,:) = T;
                    end
                end
            
            % LUMVE - Linear Unbiased Minimum Variance Estimator 
                function [x0hat, P0] = LUMVE_estimator(obj, t0, tspan_meas, z_hist, Hk_handle, Rk, Phi_handle)
                    % LUMVE_estimator estimates the initial state x0 using the Linear Unbiased Minimum Variance Estimator (LUMVE)
                    % for a linear discrete-time system with measurements collected over time.
                    %
                    % This function computes the best estimate of the initial state x0 given a series of measurements z_k, the
                    % measurement model H_k, the measurement noise covariance R_k, and the system dynamics specified by the
                    % state transition matrix Phi_k. It combines all the measurements to form a linear system and solves for x0
                    % using the LUMVE method.
                    %
                    % Inputs:
                    %   t0          - Initial time (scalar)
                    %   tspan_meas  - Vector of measurement times (1 x num_meas), where num_meas is the number of measurements
                    %   z_hist      - Measurements matrix (num_meas x nz), where each row z_hist(k,:) is the measurement z_k at time t_k
                    %                 nz is the dimension of each measurement vector z_k
                    %   Hk_handle   - Function handle to compute the measurement matrix H_k at time t_k (function of t_k)
                    %                 Usage: H_k = Hk_handle(t_k)
                    %   Rk          - Measurement noise covariance matrix (nz x nz), assumed to be the same for all measurements
                    %                 Alternatively, Rk can be a function handle Rk_handle(t_k) for time-varying covariance
                    %   Phi_handle  - Function handle to compute the state transition matrix Phi_k from t0 to t_k
                    %                 Usage: Phi_k = Phi_handle(t_k, t0)
                    %
                    % Outputs:
                    %   x0hat       - Estimated initial state (nx x 1), where nx is the dimension of the state vector x_k
                    %   P0          - Covariance matrix of the estimation error (nx x nx)
                    %
                    % Example usage:
                    %   [x0hat, P0] = obj.LUMVE_estimator(t0, tspan_meas, z_hist, Hk_handle, Rk, Phi_handle);
                    %
                    % Notes:
                    %   - The system is assumed to be linear; time-varying dynamics and measurements are supported via function handles.
                    %   - Measurement noise is assumed to be zero-mean Gaussian with known covariance R_k, which can be time-varying.
                    %   - The function handles Hk_handle and Phi_handle allow for flexibility in specifying system dynamics and measurements.
                
                    % Determine the number of measurements (num_meas) and measurement dimension (nz)
                    [num_meas, nz] = size(z_hist);    % num_meas: number of measurements, nz: measurement vector dimension
                
                    % Initialize cell arrays to collect H_k, z_k, and R_k
                    H_list = cell(num_meas, 1);       % Cell array to store H_k matrices
                    z_list = cell(num_meas, 1);       % Cell array to store z_k vectors
                    R_list = cell(num_meas, 1);       % Cell array to store R_k matrices
                
                    % Loop over each measurement to build H_combined, z_combined, and R_combined
                    for k = 1:num_meas
                        % Measurement time t_k
                        t_k = tspan_meas(k);
                
                        % Compute the state transition matrix Phi_k from t0 to t_k
                        % This propagates the initial state x0 to the state at time t_k
                        Phi_k = Phi_handle(t_k, t0);
                
                        % Compute the measurement matrix H_k at time t_k
                        Htil_k = Hk_handle(t_k);
                
                        % Compute the effective measurement matrix Hk_eff at time t_k
                        % Hk_eff maps the initial state x0 to the expected measurement z_k at time t_k
                        Hk_eff = Htil_k * Phi_k;
                
                        % Store Hk_eff
                        H_list{k} = Hk_eff;
                
                        % Extract the measurement z_k at time t_k and ensure it's a column vector (nz x 1)
                        z_k = z_hist(k, :)';    % Transpose to convert from row to column vector
                
                        % Store z_k
                        z_list{k} = z_k;
                
                        % Get measurement noise covariance R_k at time t_k
                        if isa(Rk, 'function_handle')
                            R_k = Rk(t_k);    % If Rk is a function handle, compute R_k at time t_k
                        else
                            R_k = Rk;         % Else, use the constant Rk provided
                        end
                
                        % Store R_k
                        R_list{k} = R_k;
                    end
                
                    % Determine state dimension nx from Hk_eff
                    nx = size(H_list{1}, 2);   % State dimension inferred from H_k matrices
                
                    % Combine H_list, z_list, R_list into combined matrices
                    H_combined = cell2mat(H_list);          % Combined measurement matrix (num_meas*nz x nx)
                    z_combined = cell2mat(z_list);          % Combined measurement vector (num_meas*nz x 1)
                
                    % Build R_combined as block-diagonal matrix
                    R_combined = blkdiag(R_list{:});        % Combined measurement noise covariance (num_meas*nz x num_meas*nz)
                
                    % Compute the estimator matrix M_combined using the LUMVE formula
                    % M_combined = (H_combined' * R_combined^{-1} * H_combined)^{-1} * H_combined' * R_combined^{-1}
                    % This matrix maps the combined measurements z_combined to the estimated initial state x0hat
                    % It minimizes the variance of the estimation error while ensuring the estimate is unbiased
                    R_inv = inv(R_combined);  % Inverse of the combined measurement noise covariance
                    M_combined = inv(H_combined' * R_inv * H_combined) * (H_combined' * R_inv);
                
                    % Compute the estimated initial state x0hat (nx x 1)
                    % This is the best linear unbiased estimate of the initial state x0 given the measurements
                    x0hat = M_combined * z_combined;
                
                    % Compute the covariance matrix of the estimation error P0 (nx x nx)
                    % P0 = M_combined * R_combined * M_combined'
                    % This represents the uncertainty in the estimated initial state x0hat
                    % The diagonal elements of P0 contain the variances of the estimation errors for each state component
                    P0 = M_combined * R_combined * M_combined';
                end

    %% 2) Scaled Unscented Transform

    % Unscented Transform estimate Px mx
    function [mz, Pz, Pxy, X, Y, Wm, Wc] = unscentedTransform(obj, mx, Px, h, alpha, beta, kappa)
        % unscentedTransform computes the transformed mean, covariance, and cross-covariance
        % of a random variable passed through a nonlinear function using the
        % Scaled Unscented Transform (SUT).
        %
        % Inputs:
        %   mx    - Mean vector of the input random variable (n x 1)
        %   Px    - Covariance matrix of the input random variable (n x n)
        %   h     - Function handle for the nonlinear transformation (h: R^n -> R^m)
        %   alpha - Spread parameter (usually small, e.g., 1e-3)
        %   beta  - Parameter to incorporate prior knowledge of the distribution (for Gaussian, beta=2)
        %   kappa - Secondary scaling parameter (usually 0)
        %
        % Outputs:
        %   mz  - Transformed mean vector (m x 1)
        %   Pz  - Transformed covariance matrix (m x m)
        %   Pxy - Cross-covariance matrix (n x m)
        %   X   - Sigma points matrix (n x 2n+1)
        %   Y   - Transformed sigma points matrix (m x 2n+1)
        %   Wm  - Weights for computing the mean (1 x 2n+1)
        %   Wc  - Weights for computing the covariance (1 x 2n+1)

        n = length(mx); % Dimension of the input
        lambda = alpha^2 * (n + kappa) - n;
        n_plus_lambda = n + lambda;

        % Calculate square root of (n + lambda) * Px
        S = chol(n_plus_lambda * Px, 'lower');
        mx = mx(:); % nsure mx is a column vector

        % Generate sigma points
        X = zeros(n, 2 * n + 1); % Sigma points matrix
        X(:, 1) = mx;            % First sigma point is the mean

        for i = 1:n
            X(:, i + 1)       = mx + S(:, i);
            X(:, i + n + 1)   = mx - S(:, i);
        end

        % Compute weights
        Wm = [lambda / n_plus_lambda, repmat(1 / (2 * n_plus_lambda), 1, 2 * n)];
        Wc = Wm;
        Wc(1) = Wc(1) + (1 - alpha^2 + beta);

        % Propagate sigma points through nonlinear function h
        n_sigma = size(X, 2);
        m = length(h(mx)); % Dimension of the output

        Y = zeros(m, n_sigma); % Transformed sigma points matrix

        for i = 1:n_sigma
            Y(:, i) = h(X(:, i));
        end

        % Compute transformed mean
        mz = Y * Wm';

        % Compute transformed covariance
        Pz = zeros(m, m);
        for i = 1:n_sigma
            dy = Y(:, i) - mz;
            Pz = Pz + Wc(i) * (dy * dy');
        end

        % Compute cross-covariance between input and output
        Pxy = zeros(n, m);
        for i = 1:n_sigma
            dx = X(:, i) - mx;
            dy = Y(:, i) - mz;
            Pxy = Pxy + Wc(i) * (dx * dy');
        end

    end

    %% 3) Kalman Filters
    
    %% Extended Discrete-Continuous Kalman Filter
    function [mkp, Pkp, zkhat, K_k, W_k, C_k] = kalman_filter_step(obj, mkm, Pkm, zk, Hk, Rk, param)
        % Performs one step of the discrete Kalman filter at time k
        %
        % Inputs:
        %   mkm : Prior state estimate at time k (n_x x 1)
        %   Pkm    : Prior estimate covariance at time k (n_x x n_x)
        %   z_k          : Measurement at time k (n_z x 1)
        %   H_k          : Measurement matrix at time k (n_z x n_x)
        %   R_k          : Measurement noise covariance at time k (n_z x n_z)
        %   param        : Structure of parameters
        %
        % Outputs:
        %   xhat_k_plus : Posterior state estimate at time k (n_x x 1)
        %   P_plus_k    : Posterior estimate covariance at time k (n_x x n_x)

        % Gain computation
        [K_k, W_k, C_k] = obj.kalman_gain(Pkm, Hk, Rk, param);         

        % Update step
        [mkp, Pkp, zkhat] = obj.kalman_update(mkm, Pkm, K_k, W_k, C_k, zk, Hk, param);
    end

    function [Kk, Wk, Ck] = kalman_gain(obj, Pkm, Hk, Rk, param)
        % Kalman filter gain computation at time k
        %
        % Inputs:
        %   Pkm : Prior estimate covariance at time k (n_x x n_x)
        %   Hk       : Measurement matrix at time k (n_z x n_x)
        %   Rk       : Measurement noise covariance at time k (n_z x n_z)
        %   param     : Structure of parameters
        %
        % Outputs:
        %   K_k : Kalman gain at time k (n_x x n_z)
        %   W_k : Innovation covariance at time k (n_z x n_z)
        %   C_k : Cross-covariance at time k (n_x x n_z)

        % Placeholder: Calculate H_k if needed using measurement model
        % H_k = calculate_measurement_matrix(xhat_k_minus, param);

        % Compute innovation covariance
        Wk = Hk * Pkm * Hk' + Rk;

        % epsilon = 1e-4;
        % Wk = Hk * Pkm * Hk' + Rk + epsilon * eye(size(Rk));
     
        % Compute cross-covariance
        Ck = Pkm * Hk';

        % Compute Kalman gain
        Kk = Ck / Wk;
    end

    function [mkp, Pkp, zkhat] = kalman_update(obj, mkm, Pkm, Kk, Wk, Ck, zk, Hk, param)
        % Kalman filter update step at time k
        %
        % Inputs:
        %   mkm : Prior state estimate at time k (n_x x 1)
        %   Pkm    : Prior estimate covariance at time k (n_x x n_x)
        %   Kk          : Kalman gain at time k (n_x x n_z)
        %   Wk          : Innovation covariance at time k (n_z x n_z)
        %   Ck          : Cross-covariance at time k (n_x x n_z)
        %   zk          : Measurement at time k (n_z x 1)
        %   Hk          : Measurement matrix at time k (n_z x n_x)
        %   param       : Structure of parameters
        %
        % Outputs:
        %   mkp1 : Posterior state estimate at time k (n_x x 1)
        %   Pkp1    : Posterior estimate covariance at time k (n_x x n_x)

        % Ensure zk and zkhat are column vectors
        zk = zk(:); % Convert zk to column vector if not already
        zkhat = Hk * mkm; % Compute predicted measurement
        zkhat = zkhat(:); % Convert zkhat to column vector if not already

        % Compute measurement innovation (difference)
        innovation = zk - zkhat;

        % Update state estimate
        mkp = mkm + Kk * innovation;

        % Update covariance
        Pkp = Pkm - Ck * Kk' - Kk * Ck' + Kk * Wk * Kk';
    end


    
    %% 9) Data Extraction 
        % Read NASA Ephemeris Data 
            function [XYZ_matrix] = read_ephemeris(obj, filename)
                % Function to read ephemeris data from a file and return an m x 4 matrix
                % containing Julian Date, X, Y, and Z positions as columns.
                %
                % Input:
                %   filename - String specifying the path to the ephemeris file.
                %
                % Output:
                %   XYZ_matrix - m x 4 matrix where each row contains the Julian Date, X, Y, and Z
                %                positions corresponding to a specific timestamp.
                
                % Open the file
                fileID = fopen(filename, 'r');
                
                % Initialize empty arrays for Julian Date, X, Y, Z coordinates
                JulianDate = [];
                X = [];
                Y = [];
                Z = [];
                
                % Variables to track if we're between $$SOE and $$EOE
                data_started = false;
                
                % Read through the file line by line
                while ~feof(fileID)
                    line = fgetl(fileID);
                    
                    % Check if we've reached the start of ephemeris data
                    if contains(line, '$$SOE')
                        data_started = true;  % Start reading data
                        continue;
                    end
                    
                    % Check if we've reached the end of ephemeris data
                    if contains(line, '$$EOE')
                        break;  % Stop reading data
                    end
                    
                    % Process only if we're between $$SOE and $$EOE
                    if data_started
                        % Split the line by commas
                        data = strsplit(line, ',');
                        
                        % Check if the line contains the expected number of data points (JDTDB, X, Y, Z)
                        if length(data) >= 5
                            % Convert the strings to numerical values and append to arrays
                            JulianDate = [JulianDate; str2double(data{1})];
                            X = [X; str2double(data{3})];
                            Y = [Y; str2double(data{4})];
                            Z = [Z; str2double(data{5})];
                        end
                    end
                end
                
                % Close the file
                fclose(fileID);
                
                % Stack JulianDate, X, Y, and Z as columns in a single matrix
                XYZ_matrix = [JulianDate, X, Y, Z];
            end
        
    end
end 
