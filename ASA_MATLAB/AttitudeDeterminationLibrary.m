classdef AttitudeDeterminationLibrary
    % ATTITUDEDETERMINATIONLIBRARY  Sensors, estimation, DCM/quat math, and RW control.
    %
    %   Toolbox-style class with frequently used routines for spacecraft
    %   attitude determination and control.  Organized into small, composable
    %   helpers covering sensor simulation, static attitude estimation
    %   (Wahba/QUEST/TRIAD), quaternion/DCM utilities, reaction-wheel (RW)
    %   configuration & control, and a few convenience plotting & inertia tools.
    %
    % FEATURES (by module – representative calls only)
    %   • Vector ops
    %       - calculateAngleBetweenVectors      % θ = acos( a·b / |a||b| )
    %
    %   • Sensors
    %       - Gyroscope: simulateGyroscope_noised, getGyroBiasAtTimeKp1,
    %                    plotGyroscopeMeasurements
    %       - Star tracker: simulateStarTrackerMeasurements (quat noise model),
    %                       calculateCenterOfLight (centroid from image)
    %
    %   • Actuators – Reaction Wheels (RW)
    %       - Config matrices:      compute_W4_configuration('pyramid'|'nasa')
    %       - RW dynamics + control (body): systemDynamicsControlled_RCW
    %         (PD quaternion control; choose Law1 or Law2)
    %       - Torque distribution:  pseudoinverse_method, minimax_method
    %
    %   • Attitude determination (static) – Wahba's problem
    %       - triad_method                          % TRIAD DCM
    %       - quest_method, davenport_qmethod       % optimal quaternion + DCM
    %       - estimateCovariance_Wahbas, calculateRealCovariance_Wahbas
    %
    %   • Direction Cosine Matrices (DCM)
    %       - checkDCMConstraints                   % unit/orthogonality checks
    %       - eulerAxisAngleToDCM, dcmToEulerAxisAngle
    %       - dcmFromEulerAngleSeq / dcmFromEulerAngleSeq_Sym
    %
    %   • Quaternion operations (scalar-last convention q = [q1 q2 q3 q4])
    %       - quaternionProduct, quaternionCrossMatrix, quatconj
    %       - quaternionKinematicsFixedOmega, Xi (kinematics matrix)
    %       - dcm2quat / quat2dcm / axisAngleToQuaternion / sequential rotations
    %
    %   • Misc utilities
    %       - Trig helpers: dualTrigInverse
    %       - Gravity/centers: centerOfMass, centerOfGravity, approxGravityForce_f2,
    %                          computeGravityMoment
    %       - Inertias & plots: InertiaSolidCylinder, plotInertiaEllipsoid,
    %                           plotParticles, plotRotatingBox
    %
    % CONVENTIONS & UNITS
    %   • Frames:  Body (B) and Inertial (I) unless noted. DCMs map v_I→v_B as BN.
    %   • Quaternions: scalar-last, right-handed; propagation uses q̇ = 0.5*Xi(q)*ω.
    %   • Angles: radians internally; some plotting helpers convert to deg.
    %   • Star-tracker noise input R in arcsec^2 (converted to rad^2 internally).
    %
    % DEPENDENCIES
    %   • Statistics & Machine Learning Toolbox  – mvnrnd (sensor noise)
    %   • Symbolic Math Toolbox (optional)       – dcmFromEulerAngleSeq_Sym
    %
    % QUICK EXAMPLES
    %   % 1) Gyro simulation (noisy vs true)
    %   adc = AttitudeDeterminationLibrary();
    %   dt = 0.1; N = 1000; t = (0:N-1)*dt;
    %   wtrue = [0.01;0.02;0.0] .* ones(3,N);      % rad/s
    %   b0 = deg2rad([0.05;0.05;0.05]); su = 1e-6; sn = 1e-5;
    %   b  = b0; wmeas = zeros(3,N);
    %   for k=1:N-1
    %       b  = adc.getGyroBiasAtTimeKp1(b, su, dt);
    %       wmeas(:,k+1) = adc.simulateGyroscope_noised(wtrue(:,k+1),zeros(3),su,sn,dt,b,b);
    %   end
    %   adc.plotGyroscopeMeasurements(t, wtrue, wmeas, su, sn, b0, "Gyro demo");
    %
    %   % 2) TRIAD & QUEST from two inertial/body vectors
    %   rI = [1;0;0]; sI = [0;1;0];           % inertial refs
    %   rB = [0.98;0.05;0.01]; sB = [-0.02;0.99;0.00];
    %   BNtriad = adc.triad_method(rB, sB, rI, sI);
    %   [qQuest, BNquest] = adc.quest_method([1 1],[rB sB],[rI sI],2,50);
    %
    %   % 3) DCM from 3-1-3 Euler sequence (row convention)
    %   BN = adc.dcmFromEulerAngleSeq([3 1 3],[0.1 0.2 0.3],'row');
    %   adc.checkDCMConstraints(BN);
    %
    %   % 4) RW configuration & closed-loop attitude control
    %   params = struct('a',1,'b',1,'c',1,'d',1);                % pyramid
    %   [W4,~,~] = adc.compute_W4_configuration('pyramid',params);
    %   J = diag([12,10,8]); Jinv = inv(J);
    %   qc = [0;0;0;1]; kp = 0.02; kd = 0.2;
    %   X0 = [0;0;0;1;  0.02;0;0;  0;0;0];                        % [q; ω; h]
    %   f  = @(t,X) adc.systemDynamicsControlled_RCW(t,X,J,Jinv,qc,kp,kd,adc,1);
    %   [T,Y] = ode45(f,[0 200],X0,odeset('RelTol',1e-9,'AbsTol',1e-9));
    %
    % NOTES
    %   • QUEST/Davenport return unit quaternions; always normalize inputs.
    %   • Control laws assume torque authority in body axes; RW mapping W must
    %     be consistent with wheel spin-axis definitions.
    %   • Sensor models are simple (white + random walk); tune (su,sn) to match
    %     vendor specs.
    %
    % AUTHOR
    %   Moacir Fonseca Becker
    %   Purdue University
    %
    % LAST MODIFIED
    %   08/13/2025

    methods

        %% - - - - - - - - VECTOR OPERATIONS - - - - - - - -
        % Calculate angle between two vectors
        function angle = calculateAngleBetweenVectors(obj,vec1, vec2)
            % calculateAngleBetweenVectors calculates the angle between two vectors in radians.
            %
            % Inputs:
            %   vec1 - first vector (either row or column)
            %   vec2 - second vector (either row or column)
            %
            % Output:
            %   angle - the angle between vec1 and vec2 in radians
            %
            % This function converts input vectors to column vectors (if they are
            % not already) and computes the angle between them using the dot
            % product formula.
            %
            % The formula for calculating the angle between two vectors a and b is:
            %    theta = acos((a · b) / (|a| * |b|))
            %
            % Example usage:
            %   vec1 = [1; 2; 3];
            %   vec2 = [4; 5; 6];
            %   angle = calculateAngle(vec1, vec2);

            % Ensure both vectors are column vectors
            vec1 = vec1(:);
            vec2 = vec2(:);

            % Calculate the dot product
            dotProduct = dot(vec1, vec2);

            % Calculate the magnitudes (norms) of the vectors
            normVec1 = norm(vec1);
            normVec2 = norm(vec2);

            % Calculate the angle using the dot product formula
            angle = acos(dotProduct / (normVec1 * normVec2));
        end

        %% .
        %% .
        %% .
        %% - - - - - - - - SENSORS - - - - - - - -

        %% -> GYROSCOPE
        % Discretized noised measurement error model
        function wkp1_noised = simulateGyroscope_noised(obj, wkp1_true, S_true, su, sn, dt, bk, bkp1)
            % simulateGyroAngVel_noised computes the noisy angular velocity at time k+1
            %
            % Inputs:
            %   wkp1_true - 3x1 true angular velocity vector at time
            %   k+1 (ω_k+1^true) rad/s
            %   S_true - 3x3 scaling and misalignment matrix (S)
            %   su - gyro noise parameter (σ_u)
            %   sn - gyro noise parameter (σ_v)
            %   dt - discrete time step (Δt)
            %   bk - 3x1 gyro bias at time k (β_k^true)
            %   bkp1 - 3x1 gyro bias at time k+1 (β_k+1^true)
            %
            % Outputs:
            %   wkp1_noised - 3x1 noisy angular velocity measurement at time k+1 (ω_k+1)

            % Generate a zero-mean Gaussian noise process with identity covariance matrix
            R = eye(3);   % Covariance matrix
            mu = zeros(3,1);  % Mean of the noise process
            Nvk = mvnrnd(mu, R, 1)'; % Generate a 3x1 noise vector N_v_k


            % Calculate the noisy angular velocity based on the given formula (4.54a)
            wkp1_noised = (eye(3) + S_true) * wkp1_true + 0.5 * (bk + bkp1) + sqrt((sn^2 / dt) + (1 / 12) * su^2 * dt) * Nvk;
        end

        % Gyro bias at timestep
        function bkp1 = getGyroBiasAtTimeKp1(obj,bk, su, dt)
            % getGyroBiasAtTimeKp1 computes the gyro bias at time k+1
            %
            % Inputs:
            %   bk - 3x1 gyro bias at time k (β_k^true)
            %   su - gyro noise parameter (σ_u)
            %   dt - discrete time step (Δt)
            %
            % Outputs:
            %   bkp1 - 3x1 gyro bias at time k+1 (β_k+1^true)

            % Generate a zero-mean Gaussian noise process with identity covariance matrix
            R = eye(3);
            mu = zeros(3,1);
            Nuk = mvnrnd(mu, R, 1)';

            % Calculate the gyro bias at time k+1 based on formula (4.54b)
            bkp1 = bk + su * sqrt(dt) * Nuk;
        end

        % Plot of gyro measurements
        function plotGyroscopeMeasurements(obj, t, wtrue, w_noised, su, sn, gyro_bias, figureTitle)
            % plotGyroscopeMeasurements: Plots the true vs noisy angular velocity
            % components for a gyroscope measurement error model simulation in degrees per second.
            %
            % Inputs:
            %   t           - Time vector
            %   wtrue       - 3xN matrix of true angular velocity components [rad/s]
            %   w_noised    - 3xN matrix of noisy angular velocity components [rad/s]
            %   su          - Noise parameter sigma_u [rad/s^(3/2)]
            %   sn          - Noise parameter sigma_n [rad/s^(1/2)]
            %   gyro_bias   - Initial bias [rad/s] (3x1 vector)
            %   figureTitle - Custom title for the figures (string)

            % Convert from radians per second to degrees per second
            wtrue_deg = rad2deg(wtrue);
            w_noised_deg = rad2deg(w_noised);
            gyro_bias_deg = rad2deg(gyro_bias);

            % Take a single value of bias since all components are the same
            single_bias_deg = gyro_bias_deg(1);

            % Define the parameters string to display in degrees using only the single bias value
            parameters_str = sprintf(['$\\sigma_u = %.2e$ rad/s$^{3/2}$\n', ...
                '$\\sigma_v = %.2e$ rad/s$^{1/2}$\n', ...
                'Initial Bias (deg/s): %.5f'], ...
                su, sn, single_bias_deg);

            % Define line styles and widths for clarity
            trueLineStyle = '-';    % Solid line for true angular velocity
            noisyLineStyle = '--';  % Dashed line for noisy angular velocity
            trueLineWidth = 1.5;    % Thinner line for true
            noisyLineWidth = 1.8;   % Slightly thicker line for noisy

            % Define colors
            trueColor = 'b';   % Blue
            noisyColor = 'r';  % Red

            % Create a new figure for the true vs noisy angular velocity components
            figure;

            % Subplot for x-component
            subplot(3, 1, 1);
            plot(t, wtrue_deg(1, :), 'Color', trueColor, 'LineStyle', trueLineStyle, ...
                'LineWidth', trueLineWidth, 'DisplayName', 'True $\omega_x$');
            hold on;
            plot(t, w_noised_deg(1, :), 'Color', noisyColor, 'LineStyle', noisyLineStyle, ...
                'LineWidth', noisyLineWidth, 'DisplayName', 'Noisy $\omega_x$');
            xlabel('Time (s)', 'FontSize', 12, 'Interpreter', 'latex');
            ylabel('$\omega_x$ (deg/s)', 'FontSize', 12, 'Interpreter', 'latex');
            title([figureTitle ' - x Component'], 'FontSize', 12, 'Interpreter', 'latex');
            legend('FontSize', 12, 'Interpreter', 'latex', 'Location', 'best');
            grid on;
            text(t(end) * 0.6, max(wtrue_deg(1,:)) * 0.8, parameters_str, 'FontSize', 12, ...
                'BackgroundColor', 'white', 'EdgeColor', 'black', 'VerticalAlignment', 'top', 'Interpreter', 'latex');
            hold off;

            % Subplot for y-component
            subplot(3, 1, 2);
            plot(t, wtrue_deg(2, :), 'Color', trueColor, 'LineStyle', trueLineStyle, ...
                'LineWidth', trueLineWidth, 'DisplayName', 'True $\omega_y$');
            hold on;
            plot(t, w_noised_deg(2, :), 'Color', noisyColor, 'LineStyle', noisyLineStyle, ...
                'LineWidth', noisyLineWidth, 'DisplayName', 'Noisy $\omega_y$');
            xlabel('Time (s)', 'FontSize', 12, 'Interpreter', 'latex');
            ylabel('$\omega_y$ (deg/s)', 'FontSize', 12, 'Interpreter', 'latex');
            title([figureTitle ' - y Component'], 'FontSize', 12, 'Interpreter', 'latex');
            legend('FontSize', 12, 'Interpreter', 'latex', 'Location', 'best');
            grid on;
            hold off;

            % Subplot for z-component
            subplot(3, 1, 3);
            plot(t, wtrue_deg(3, :), 'Color', trueColor, 'LineStyle', trueLineStyle, ...
                'LineWidth', trueLineWidth, 'DisplayName', 'True $\omega_z$');
            hold on;
            plot(t, w_noised_deg(3, :), 'Color', noisyColor, 'LineStyle', noisyLineStyle, ...
                'LineWidth', noisyLineWidth, 'DisplayName', 'Noisy $\omega_z$');
            xlabel('Time (s)', 'FontSize', 12, 'Interpreter', 'latex');
            ylabel('$\omega_z$ (deg/s)', 'FontSize', 12, 'Interpreter', 'latex');
            title([figureTitle ' - z Component'], 'FontSize', 12, 'Interpreter', 'latex');
            legend('FontSize', 12, 'Interpreter', 'latex', 'Location', 'best');
            grid on;
            hold off;

            % Set figure background color
            set(gcf, 'Color', 'w');

            % Create a new figure for the error plot (w_noised - wtrue) in degrees
            figure;

            error_w_deg = w_noised_deg - wtrue_deg;

            % Define line styles and colors for error plot
            errorLineStyles = {'-', '--', '-.'};
            errorColors = {'b', 'r', 'k'};
            errorLineWidth = 1.5;

            hold on;
            for i = 1:3
                plot(t(2:end), error_w_deg(i, 2:end), 'Color', errorColors{i}, ...
                    'LineStyle', errorLineStyles{i}, 'LineWidth', errorLineWidth, ...
                    'DisplayName', sprintf('Error $\\omega_%c$', char('x' + i -1)));
            end
            hold off;

            xlabel('Time (s)', 'FontSize', 12, 'Interpreter', 'latex');
            ylabel('Error (deg/s)', 'FontSize', 12, 'Interpreter', 'latex');
            title(['Error: Noisy - True Angular Velocity (' figureTitle ')'], 'FontSize', 12, 'Interpreter', 'latex');
            legend('FontSize', 12, 'Interpreter', 'latex', 'Location', 'best');
            grid on;

            axis_limits = axis;

            x_position = axis_limits(1) + 0.05 * (axis_limits(2) - axis_limits(1));
            y_position = axis_limits(4) - 0.1 * (axis_limits(4) - axis_limits(3));

            text(x_position, y_position, parameters_str, 'FontSize', 12, ...
                'BackgroundColor', 'white', 'EdgeColor', 'black', 'VerticalAlignment', 'top', 'Interpreter', 'latex');

            % Set figure background color
            set(gcf, 'Color', 'w');

        end

        %% :)
        %% -> STAR TRACKER

        % Simulate star tracker measurements
        function q_meas = simulateStarTrackerMeasurements(obj, q_truth, R)
            % simulateStarTrackerMeasurementsv1 Simulates star tracker quaternion measurements.
            %
            % Inputs:
            %   q_truth - [4xN] matrix of true quaternions (each column is a quaternion).
            %   R       - 3x3 covariance matrix for the noise (in arcsec^2).
            %
            % Output:
            %   q_meas  - [4xN] matrix of simulated measured quaternions.

            % Convert the covariance matrix R from arcseconds^2 to radians^2
            R = R * (pi / (180 * 3600))^2;  % Convert arcseconds^2 to radians^2

            % Ensure q_truth is a 2D matrix with 4 rows
            if size(q_truth, 1) ~= 4
                error('q_truth must be a 4xN matrix, where each column is a quaternion.');
            end

            % Ensure the covariance matrix R is 3x3
            if ~isequal(size(R), [3, 3])
                error('R must be a 3x3 covariance matrix.');
            end

            % Get the number of quaternions
            N = size(q_truth, 2);

            % Initialize the output matrix
            q_meas = zeros(4, N);

            % Perform Cholesky decomposition of the covariance matrix
            L = chol(R, 'lower');

            % Loop through each quaternion
            for i = 1:N
                % Extract the true quaternion as a column vector
                q = q_truth(:, i);

                % Generate random noise vector (3x1) using Cholesky decomposition
                vartheta = L * randn(3, 1);  % 3x1 vector

                % Construct the small angle quaternion [1/2 * vartheta; 1]
                delta_q = [0.5 * vartheta; 1];

                % Normalize delta_q to ensure it's a valid quaternion
                delta_q = delta_q / norm(delta_q);

                % Compute the measured quaternion using quaternion multiplication
                q_meas_i = obj.quaternionProduct(delta_q, q);

                % Normalize the resulting quaternion to avoid singularities
                q_meas_i = q_meas_i / norm(q_meas_i);

                % Store the measured quaternion
                q_meas(:, i) = q_meas_i;
            end
        end


        % Center of Light Calculation
        function [CoLx, CoLy] = calculateCenterOfLight(obj,img)
            % calculateCenterOfLight Computes the center of light for a given image.
            %
            % INPUT:
            %   img - 2D matrix representing the light intensity values of the image.
            %
            % OUTPUT:
            %   CoLx - x-coordinate of the center of light
            %   CoLy - y-coordinate of the center of light

            % Get image dimensions
            [ypixels, xpixels] = size(img);

            % Initialize center of light variables
            CoLx = 0;
            CoLy = 0;

            % Calculate total intensity for normalization
            S_total = sum(img(:));

            % Calculate the weighted sum for x and y coordinates
            for x = 1:xpixels
                for y = 1:ypixels
                    % Light intensity at this pixel
                    Sxy = img(y, x);

                    % Accumulate weighted coordinates
                    CoLx = CoLx + Sxy * x;
                    CoLy = CoLy + Sxy * y;
                end
            end

            % Normalize by total intensity to get the true center of light
            CoLx = CoLx / S_total;
            CoLy = CoLy / S_total;
        end

        %% .
        %% .
        %% .
        %% - - - - - - - - ACTUATORS - - - - - - - -

        %% -> REACTION WHEELS
        %% :)
        %% - - 4 RCW Configuration Matrices
        function [W4, W4_pseudo,null_vector] = compute_W4_configuration(obj,config, params)
            % Computes the W4 matrix, null space vector, and pseudoinverse for the specified configuration
            % Inputs:
            % - config: 'pyramid' or 'nasa' to choose the configuration
            % - params: Struct containing parameters for the configurations
            %           For 'pyramid', params should include fields 'a', 'b', 'c', and 'd'.
            %           For 'nasa', params should include fields 'alpha', 'beta', and 'gamma'.
            % Outputs:
            % - W4: Transformation matrix for reaction wheels
            % - null_vector: Null space vector
            % - W4_pseudo: Pseudoinverse of W4

            if strcmp(config, 'pyramid')
                % Unpack parameters for the pyramid configuration
                a = params.a;
                b = params.b;
                c = params.c;
                d = params.d;

                % Pyramid configuration matrix (Equation 4.66)
                W4 = [ a -a 0 0;
                    b b c c;
                    0 0 d -d ];

                % Null space vector for pyramid configuration (Equation 4.67)
                null_vector = (1 / sqrt(2 * (b^2 + c^2))) * [ c; c; -b; -b ];

                % Check if the sum of squares is equal to 1
                sum_of_squares = sum(null_vector.^2);
                fprintf('Pyramid configuration null vector sum of squares: %.2f\n', sum_of_squares);

            elseif strcmp(config, 'nasa')
                % Unpack parameters for the NASA configuration
                alpha = params.alpha;
                beta = params.beta;
                gamma = params.gamma;

                % NASA standard configuration matrix
                W4 = [ 1 0 0 alpha;
                    0 1 0 beta;
                    0 0 1 gamma ];

                % Null space vector for NASA configuration
                null_vector = (1 / sqrt(1 + alpha^2 + beta^2 + gamma^2)) * [ alpha; beta; gamma; -1 ];

                % Check if the sum of squares is equal to 1
                sum_of_squares = sum(null_vector.^2);
                fprintf('NASA standard configuration null vector sum of squares: %.2f\n', sum_of_squares);

            else
                error('Invalid configuration. Choose either "pyramid" or "nasa".');
            end


            % Pseudoinverse of W4
            W4_pseudo = W4' /(W4 * W4');
        end

        %% :)
        %% - - RCW Dynamics and Control
        % - Reaction Wheels Quaternion-Based Dynamics with Control
        function Xdot = systemDynamicsControlled_RCW(obj, t, X, J, Jinv, qc, kp, kd, adc, controlLaw)
            % Updated systemDynamicsControlled function that includes angular momentum h in the state vector.
            %
            % Inputs:
            %   t          - Current time (required by ODE solvers)
            %   X          - Current state vector [q1; q2; q3; q4; omega1; omega2; omega3; h1; h2; h3]
            %   J          - Inertia matrix of the spacecraft (3x3 matrix)
            %   Jinv       - Inverse of the inertia matrix (3x3 matrix)
            %   qc         - Desired (control) quaternion [qc1; qc2; qc3; qc4]
            %   kp         - Proportional gain for the controller (scalar)
            %   kd         - Derivative gain for the controller (scalar)
            %   adc        - Object containing necessary methods, e.g., Xi and skew_symmetric
            %   controlLaw - Identifier for the control law to be used (1 for Law1, 2 for Law2)
            %
            % Outputs:
            %   Xdot       - Time derivative of the state vector [q_dot; omega_dot; G_]

            % Extract quaternion, angular velocity, and angular momentum from state vector
            q = X(1:4);        % Quaternion [q1; q2; q3; q4]
            omega = X(5:7);    % Angular velocity vector [omega1; omega2; omega3]
            h = X(8:10);       % Angular momentum of the body [h1; h2; h3]

            % Compute quaternion state errors
            dq13 = obj.Xi(qc)' * q;
            dq4 = q' * qc;
            dq = [dq13; dq4];  % Assemble the error vector [dq1; dq2; dq3; dq4]

            % Compute control input based on the selected control law
            if controlLaw == 1
                L = obj.computeControlTorque_Law1(dq, omega, kp, kd);  % Control Law 1
            elseif controlLaw == 2
                L = obj.computeControlTorque_Law2(dq, omega, kp, kd);  % Control Law 2
            else
                error('Invalid controlLaw identifier. Use 1 for Law1 or 2 for Law2.');
            end

            % Compute natural dynamics of the system
            q_dot = 0.5 * obj.Xi(q) * omega;  % Time derivative of the quaternion

            % Compute omega_dot using Euler's equation: J * omega_dot = -omega x H_total + L
            omega_dot = Jinv * (-obj.skew_symmetric(omega) * J * omega + L);

            % Compute h_dot: h_dot = -omega x h - L (reaction torque from wheels)
            h_dot = -obj.skew_symmetric(omega) * h - L;

            % Assemble the state derivative
            Xdot = [q_dot; omega_dot; h_dot];
        end

        % RCW Control Torque Law 1 (7.17b) from Crassidis
        function L = computeControlTorque_RCW_Law1(obj, dq, omega, kp, kd)
            % computeControlTorque_Law1 Computes the control torque using Control Law 1.
            % Based on equation (7.17b) from Crassidis
            % Syntax:
            %   L = computeControlTorque_Law1(dq, omega, kp, kd)
            %
            % Inputs:
            %   dq   - Quaternion error vector [dq1; dq2; dq3; dq4]
            %   omega- Angular velocity vector [omega1; omega2; omega3]
            %   kp   - Proportional gain for the controller (scalar)
            %   kd   - Derivative gain for the controller (scalar)
            %
            % Outputs:
            %   L    - Control torque vector [L1; L2; L3; L4] (assuming 4 wheels or actuators)

            % Extract components of quaternion error
            dq4 = dq(4);          % Scalar part of the quaternion error
            dq13 = dq(1:3);      % Vector part [dq1; dq2; dq3]

            % Calculate the control torque L using Control Law 1
            % Formula:
            % L = -kp * sign(dq4) * dq13 - kd * (1 + dq13' * dq13) * omega
            L = -kp * sign(dq4) * dq13 ...
                - kd * (1 + dq13' * dq13) * omega;
        end

        % RCW Control Torque Law 2 (7.17a) from Crassidis
        function L = computeControlTorque_RCW_Law2(obj, dq, omega, kp, kd)
            % computeControlTorque_Law2 Computes the control torque using Control Law 2.
            % Based on equation (7.17a) from Crassidis
            % Syntax:
            %   L = computeControlTorque_Law2(dq, omega, kp, kd)
            %
            % Inputs:
            %   dq   - Quaternion error vector [dq1; dq2; dq3; dq4]
            %   omega- Angular velocity vector [omega1; omega2; omega3]
            %   kp   - Proportional gain for the controller (scalar)
            %   kd   - Derivative gain for the controller (scalar)
            %
            % Outputs:
            %   L    - Control torque vector [L1; L2; L3; L4] (assuming 4 wheels or actuators)

            % Extract components of quaternion error
            dq4 = dq(4);          % Scalar part of the quaternion error
            dq13 = dq(1:3);      % Vector part [dq1; dq2; dq3]

            % Calculate the control torque L using Control Law 2
            % Formula:
            % L = -kp * sign(dq4) * dq13 - kd * omega
            L = -kp * sign(dq4) * dq13 ...
                - kd * omega;
        end

        %% :)
        %% - - RCW Torque Distribution

        %% - - - Pseudoinverse Method
        function [h_wheels_history, omega_wheels, L_wheels] = pseudoinverse_method(obj, W, h_values, J_wheel)
            % Computes wheel angular momenta, angular velocities, and torques using the pseudoinverse method.
            %
            % Inputs:
            %   W                 - Wheel configuration matrix (3 x N)
            %   h_values          - Angular momentum of the wheels over time (num_steps x 3)
            %   J_wheel           - Inertia of the wheels (scalar)
            %
            % Outputs:
            %   h_wheels_history  - Wheel angular momenta over time (num_steps x N)
            %   omega_wheels      - Wheel angular velocities over time (num_steps x N)
            %   L_wheels          - Wheel torques over time (num_steps x N)

            num_steps = size(h_values, 1);
            N = size(W, 2);  % Number of wheels

            % Compute pseudoinverse of W
            W_pinv = W' / (W * W');

            % Initialize outputs
            h_wheels_history = zeros(num_steps, N);
            omega_wheels = zeros(num_steps, N);
            L_wheels = zeros(num_steps, N);

            for t = 1:num_steps
                % Angular momentum of the wheels at time t
                h = h_values(t, :)';  % 3x1 vector (body frame)

                % Compute wheel angular momenta using pseudoinverse
                h_wheels = W_pinv * h;  % Nx1 vector (wheel frame)

                % Store h_wheels in history
                h_wheels_history(t, :) = h_wheels';

                % Compute wheel angular velocities
                omega_wheels(t, :) = (h_wheels / J_wheel)';  % 1xN vector

                % Compute wheel torques
                if t > 1
                    % Compute derivative of wheel angular momenta
                    delta_h_wheels = h_wheels - prev_h_wheels;
                    L_wheels(t, :) = (delta_h_wheels / (t - (t - 1)))';  % Approximate derivative
                else
                    L_wheels(t, :) = zeros(1, N);  % Initial torques are zero
                end
                prev_h_wheels = h_wheels;
            end
        end


        %% - - - Minimax Method
        % Main function to run them all (requires more validation)
        function [H_wheels_minimax, omega_wheels_minimax, L_wheels_minimax] = minimax_method(obj, W, omega_values, h_values, J, dt, J_wheel)
            % Function to compute wheel angular momenta, angular velocities, and torques using the minimax method.

            N = size(W, 2);          % Number of wheels
            num_steps = size(omega_values, 1);

            % Initialize outputs
            H_wheels_minimax = zeros(num_steps, N);
            omega_wheels_minimax = zeros(num_steps, N);
            L_wheels_minimax = zeros(num_steps, N);

            for t = 1:num_steps
                omega = omega_values(t, :)';    % Angular velocity at time t (3x1 vector)
                h = h_values(t, :)';            % Angular momentum of the body at time t (3x1 vector)

                % Implement the minimax method for this time step
                % - - - Minimax Implementation Start - - -

                % Initialize variables for tracking the maximum dot product
                max_dot_wij_Htotal = -inf;

                % Loop over all possible wheel pairs combinations (i, j)
                for i = 1:N-1
                    for j = i+1:N
                        % Get spin axis vectors for wheels i and j
                        w_i = W(:, i);
                        w_j = W(:, j);

                        % Get spin axes of remaining wheels k ≠ i, j
                        k_list = setdiff(1:N, [i, j]);
                        w_k_list = W(:, k_list);

                        % Compute s_{ijk} for k ≠ i, j
                        s_ijk = obj.compute_sijk(w_i, w_j, w_k_list);

                        % Compute v_{ij}
                        v_ij = obj.compute_vij(s_ijk, w_k_list);

                        % Compute w_{ij}
                        [w_ij, ~] = obj.compute_wij(w_i, w_j, v_ij);

                        % Compute dot product w_{ij} ⋅ H_total
                        dot_wij_Htotal = dot(w_ij, h);

                        % Keep track of maximum dot product and corresponding variables
                        if dot_wij_Htotal > max_dot_wij_Htotal
                            max_dot_wij_Htotal = dot_wij_Htotal;
                            max_i = i;
                            max_j = j;
                            selected_v_ij = v_ij;
                            selected_w_i = w_i;
                            selected_w_j = w_j;
                            selected_s_ijk = s_ijk;
                            selected_k_list = k_list;
                        end
                    end
                end

                % After finding the optimal pair, compute H_i^w, H_j^w, H_0
                [Hiw, Hjw, H0] = obj.compute_Hiw_Hjw_H0(selected_w_i, selected_w_j, selected_v_ij, h);

                % Compute H_k^w for k ≠ i, j
                H_kw = obj.compute_Hkw(H0, selected_s_ijk, selected_k_list, N);

                % Assign H_iw and H_jw to their respective indices
                H_kw(max_i) = Hiw;
                H_kw(max_j) = Hjw;

                % Store H_kw for this time step
                H_wheels_minimax(t, :) = H_kw';

                % Compute wheel angular velocities
                omega_wheels_minimax(t, :) = H_wheels_minimax(t, :) / J_wheel;

                % - - - Minimax Implementation End - - -
            end
        end

        function s_ijk = compute_sijk(obj, w_i, w_j, w_k_list)
            % Computes s_{ijk} for k ≠ i, j
            cross_wi_wj = cross(w_i, w_j);
            num_k = size(w_k_list, 2);
            s_ijk = zeros(num_k, 1);
            for idx = 1:num_k
                w_k = w_k_list(:, idx);
                s_ijk(idx) = dot(cross_wi_wj, w_k);
            end
        end

        function v_ij = compute_vij(obj, s_ijk, w_k_list)
            % Computes v_{ij}
            num_k = length(s_ijk);
            v_ij = zeros(3,1);
            for idx = 1:num_k
                w_k = w_k_list(:, idx);
                sign_sijk = sign(s_ijk(idx));
                v_ij = v_ij + w_k * sign_sijk;
            end
        end

        function d_ij = compute_dij(obj, w_i, w_j, v_ij)
            % Computes d_{ij}
            cross_wi_wj = cross(w_i, w_j);
            numerator = dot(cross_wi_wj, v_ij);
            denominator = norm(cross_wi_wj);
            d_ij = numerator / denominator;
        end

        function [Hiw, Hjw, H0] = compute_Hiw_Hjw_H0(obj, w_i, w_j, v_ij, H_B_w)
            % Computes H_i^w, H_j^w, and H_0
            cross_wi_wj = cross(w_i, w_j);
            denominator = dot(cross_wi_wj, v_ij);
            if abs(denominator) < 1e-6
                error('Denominator is too small, possible numerical instability.');
            end
            numerator_matrix = [cross(w_j, v_ij)'; cross(v_ij, w_i)'; cross_wi_wj'];
            H_values = (1 / denominator) * numerator_matrix * H_B_w;
            Hiw = H_values(1);
            Hjw = H_values(2);
            H0 = H_values(3);
        end

        function H_kw = compute_Hkw(obj, H0, s_ijk, k_list, N)
            % Computes H_k^w for k ≠ i, j
            H_kw = zeros(N, 1);
            num_k = length(s_ijk);
            for idx = 1:num_k
                k = k_list(idx);
                sign_sijk = sign(s_ijk(idx));
                H_kw(k) = H0 * sign_sijk;
            end
            % H_kw for i and j will be assigned later
        end

        function [w_ij, denominator] = compute_wij(obj, w_i, w_j, v_ij)
            % Computes w_{ij} and its denominator
            cross_wi_wj = cross(w_i, w_j);
            denominator = dot(cross_wi_wj, v_ij);
            w_ij = cross_wi_wj / denominator;
        end


        %% .
        %% .
        %% .
        %% - - - - - - ATTITUDE DETERMINATION METHODS - - - - - -

        %% -> Static Determination Methods
        % TRIAD Method
        function BN_triad = triad_method(obj,v1_B, v2_B, v1_I, v2_I)
            % TRIAD_METHOD Performs TRIAD algorithm for attitude determination
            %
            % Usage:
            %   BN_triad = AttitudeDeterminationLibrary.triad_method(v1_B, v2_B, v1_I, v2_I)
            %
            % Inputs:
            %   v1_B - 3x1 body-frame vector
            %   v2_B - 3x1 body-frame vector
            %   v1_I - 3x1 inertial-frame vector
            %   v2_I - 3x1 inertial-frame vector
            %
            % Outputs:
            %   BN_triad - Direction cosine matrix from inertial to body frame

            % Normalize the vectors
            v1_B = v1_B / norm(v1_B);
            v2_B = v2_B / norm(v2_B);
            v1_I = v1_I / norm(v1_I);
            v2_I = v2_I / norm(v2_I);

            % Construct TRIAD basis for the body frame
            t1_B = v1_B;
            t2_B = cross(v1_B, v2_B) / norm(cross(v1_B, v2_B));
            t3_B = cross(t1_B, t2_B);

            % Construct TRIAD basis for the inertial frame
            t1_I = v1_I;
            t2_I = cross(v1_I, v2_I) / norm(cross(v1_I, v2_I));
            t3_I = cross(t1_I, t2_I);

            % Form the rotation matrix from inertial to body frame
            BN_triad = [t1_B, t2_B, t3_B] * [t1_I, t2_I, t3_I]';
        end

        % - Using Wahba's Problem Definition -
        % QUEST Method
        function [optimal_quaternion, BN_quest, loss, taste, lambda_max] = quest_method(obj,weights_vector, body_vectors, inertial_vectors, num_vectors_quest, max_iter)
            % QUEST Method for attitude estimation
            % Input:
            %   weights_vector: 1xN vector of weights for the measurements
            %   body_vectors: 3xN matrix of body frame vectors (measured with noise)
            %   inertial_vectors: 3xN matrix of corresponding vectors in the inertial frame
            %   num_vectors_quest: Number of vectors to use in the QUEST method
            % Output:
            %   optimal_quaternion: Optimal quaternion estimated by the QUEST method
            %   BN_quest: Inertial to body DCM obtained via the QUEST method

            % Initialize the B matrix and z vector
            B = zeros(3, 3);
            z = zeros(3, 1);

            % Check if max_iter is provided, if not, default to 10
            if nargin < 6
                max_iter = 50;
            end

            % Compute B matrix and z vector
            for i = 1:num_vectors_quest
                ai = weights_vector(i);
                bi = body_vectors(:, i);  % Body frame vector (noisy)
                ri = inertial_vectors(:, i);  % Inertial frame vector (ideal)

                % Update the B matrix (weighted outer product)
                B = B + ai * (bi * ri');

                % Update the z vector (weighted cross product)
                z = z + ai * cross(bi, ri);
            end

            % Compute the trace of B matrix [OK]
            trB = trace(B);

            % Construct S matrix (Symmetric part of B) [OK]
            S = B + B';

            % Compute other components [OK]
            det_S = det(S);

            % Compute κ (kappa)
            kappa = trace(adjoint(S));

            % Define characteristic equation (quartic form)
            psi_quest = @(lambda) (lambda^2 - trB^2 + kappa) * (lambda^2 - trB^2 - norm(z)^2) ...
                - (lambda - trB) * (z' * S * z + det_S) - z' * S^2 * z;

            % Derivative of characteristic equation based [OK]
            psi_dtdLambda = @(lambda) 2 * lambda * (lambda^2 - trB^2 + kappa) ...
                - (z' * S * z + det_S) ...
                - 2 * lambda * (-lambda^2 + trB^2 + norm(z)^2);

            % Use Newton-Raphson to find the largest eigenvalue (lambda_max)
            lambda_old = 1;         % Initial guess for lambda
            tol = 1e-12;            % Convergence tolerance
            iter = 0;

            while iter < max_iter
                lambda_new = lambda_old - psi_quest(lambda_old) / psi_dtdLambda(lambda_old);
                if abs(lambda_new - lambda_old) < tol
                    break;
                end
                lambda_old = lambda_new;
                iter = iter + 1;
            end
            lambda_max = lambda_new;

            % Calculate rho
            rho = lambda_max + trB;

            % Compute the quaternion (without normalization scalar alpha)
            q13 = adjoint(rho * eye(3) - S) * z;

            % The fourth component of the quaternion
            q4 = det(rho * eye(3) - S);

            % The full quaternion
            optimal_quaternion = [q13; q4];

            % Normalize the quaternion
            optimal_quaternion = optimal_quaternion / norm(optimal_quaternion);

            % Convert quaternion to DCM
            BN_quest = obj.quat2dcm(optimal_quaternion);

            % Compute the loss function value (λ₀ - λ_max)
            lambda_0 = sum(weights_vector);
            loss = lambda_0 - lambda_max;

            % Compute TASTE = 2 * L(A)
            taste = 2 * loss;

        end

        % Davenport's Q-Method
        function [optimal_quaternion, BN_qmethod, loss, taste] = davenport_qmethod(obj,weights_vector, body_vectors, inertial_vectors, num_vectors_Qmethod)
            % Function to implement Davenport's Q-Method for attitude determination
            % Input:
            %   noisy_vectors: 3xN matrix of noisy vectors in the body frame
            %   inertial_vectors: 3xN matrix of corresponding vectors in the inertial frame
            %   num_vectors_Qmethod: Number of vectors to use in the Q-method
            % Output:
            %   optimal_quaternion: Optimal quaternion found using the Q-method
            %   BN_qmethod: Inertial to body DCM obtained via Q-method



            % Construction of B Attitude Profile Matrix
            B = zeros(3, 3);
            z = zeros(3, 1);  % Initialize z vector

            % Compute the B matrix and z vector
            for i = 1:num_vectors_Qmethod
                ai = weights_vector(i);
                bi = body_vectors(:, i);  % Body frame vector (noisy)
                ri = inertial_vectors(:, i);            % Inertial frame vector (ideal)

                % Update the B matrix (weighted outer product)
                B = B + ai * (bi * ri');

                % Update the z vector (weighted cross product)
                z = z + ai * cross(bi, ri);
            end


            % Compute the trace of the B matrix
            trace_B = trace(B);

            % Construction of K Matrix from Davenport's method
            K_matrix = [B + B' - trace_B * eye(3), z;
                z', trace_B];

            % Compute the eigenvalues and eigenvectors of the K matrix
            [eigenvectors, eigenvalues_matrix] = eig(K_matrix);

            % Extract the eigenvalues from the diagonal matrix
            eigenvalues = diag(eigenvalues_matrix);

            % Find the largest eigenvalue and its corresponding index
            [lambda_max, index] = max(eigenvalues);

            % The optimal quaternion is the eigenvector corresponding to the largest eigenvalue
            optimal_quaternion = eigenvectors(:, index);

            % Normalize the quaternion to ensure it has unit length
            optimal_quaternion = optimal_quaternion / norm(optimal_quaternion);

            % Find the DCM matrix from the quaternion above
            BN_qmethod = obj.quat2dcm(optimal_quaternion);

            % Calculate the loss L(A) = λ₀ - λ_max
            lambda_0 = sum(weights_vector);
            loss = lambda_0 - lambda_max;

            % Calculate TASTE = 2 * L(A)
            taste = 2 * loss;
        end

        % Covariance Matrix From Measurements in Wahba's
        function P_theta = estimateCovariance_Wahbas(obj,lambda_max, measured_vectors, inertial_vectors, weights, DCM_est, c)
            % calculateCovarianceMatrix calculates the covariance matrix for Wahba's problem
            % using the QUEST
            %
            % INPUTS:
            %   lambda_max       - The largest eigenvalue obtained from the QUEST method (scalar)
            %   measured_vectors - A 3xN matrix containing N measured vectors in the body frame
            %   inertial_vectors - A 3xN matrix containing N corresponding vectors in the inertial frame
            %   weights          - An Nx1 vector containing the weights for each measurement
            %   DCM_est          - The estimated Direction Cosine Matrix (DCM) from the QUEST method (3x3 matrix)
            %   c                - The scaling factor (scalar), calculated using the chosen formula for covariance estimation
            %
            % OUTPUT:
            %   P_theta          - The resulting covariance matrix (3x3) for Wahba's problem, which measures the uncertainty
            %                      in the attitude estimation.
            %
            % DESCRIPTION:
            % This function computes the covariance matrix \( P_{\theta} \) using the formula:
            %
            % \[
            % P_{\theta} \approx c (\lambda_{\max} I_3 - B A_{q}^T)^{-1}
            % \]
            %
            % where \( B \) is the matrix formed by the weighted outer products of the measured and inertial vectors,
            % \( A_{q} \) (input `DCM_est`) is the estimated DCM from the QUEST method, and \( \lambda_{\max} \) is the
            % largest eigenvalue from the QUEST method. The scaling factor \( c \) determines the influence of measurement
            % variances in the calculation.
            %
            % The function validates the inputs, calculates the B matrix based on the provided weights and vectors, and
            % finally computes \( P_{\theta} \) using the specified formula.

            % Validate that the number of measured vectors and weights match
            N = size(measured_vectors, 2);
            if length(weights) ~= N
                error('The number of weights must match the number of vectors.');
            end

            % Calculate the B matrix based on the provided weights, measured_vectors, and inertial_vectors
            B = zeros(3, 3);
            for i = 1:N
                B = B + weights(i) * measured_vectors(:, i) * inertial_vectors(:, i)';
            end

            % Calculate P_theta using the provided formula
            P_theta = c * pinv(lambda_max * eye(3) - B * DCM_est');

        end

        % Covariance Matrix From Real Data in Wahba's
        function P_theta_real = calculateRealCovariance_Wahbas(obj,N, std_dev, true_vector)
            % calculateRealCovariance calculates the real covariance matrix using the Fisher information matrix.
            %
            % INPUTS:
            %   N            - The number of measurement vectors (scalar)
            %   std_dev      - The standard deviation in degrees for x and y directions (scalar)
            %   true_vector  - The true body frame vector (3x1 column vector)
            %
            % OUTPUT:
            %   P_theta_real - The calculated covariance matrix (3x3) based on the inverse of the Fisher information matrix
            %
            % DESCRIPTION:
            % This function computes the real covariance matrix \( P_{\theta} \) using the Fisher information matrix.
            % It calculates the weights inversely proportional to the variances and sums them to form the Fisher information matrix,
            % which is then inverted to obtain the covariance matrix.
            %
            % The covariance matrix is a measure of uncertainty in the attitude estimation.

            % Convert standard deviation from degrees to radians
            sigma_x = deg2rad(std_dev);
            sigma_y = deg2rad(std_dev);

            % Initialize the Fisher Information matrix and weights
            F = zeros(3, 3);
            weights = zeros(N, 1);

            % Calculate the weights inversely proportional to the variances
            for i = 1:N
                weights(i) = 1 / (sigma_x^2);  % Assuming equal variances for x and y directions
            end

            % Calculate the Fisher Information matrix F
            for i = 1:N
                F = F + weights(i) * (eye(3) - true_vector * true_vector');
            end

            % Calculate the real covariance matrix P as the inverse of F
            P_theta_real = pinv(F);  % Use pinv to handle potential numerical stability issues

        end
        %% .
        %% .
        %% .
        %% - - - - - - DIRECTION COSINE MATRICES - - - - - -
        %% -> Constraints Checker
        function checkDCMConstraints(obj, DCM)
            % checkDCMConstraints verifies that a 3x3 direction cosine matrix (DCM)
            % satisfies the unit vector constraints (each row and each column has norm 1)
            % and the orthogonality constraints (DCM * DCM' = I and DCM' * DCM = I).
            %
            % This method prints the results of each check.
            %
            % INPUT:
            %   DCM - a 3x3 numeric matrix representing a DCM.
            %
            % Example:
            %   obj.checkDCMConstraints(DCM);
            %

            tol = 1e-6;  % tolerance for checking equality

            % Verify that DCM is 3x3.
            if ~isequal(size(DCM), [3,3])
                fprintf('Error: DCM must be a 3x3 matrix.\n');
                return;
            end

            % Check unit vector constraints

            % Check each row has norm 1.
            rowNorms = sqrt(sum(DCM.^2, 2));
            if all(abs(rowNorms - 1) < tol)
                fprintf('Row unit vector constraints satisfied.\n');
            else
                fprintf('Row unit vector constraints NOT satisfied. Row norms: %s\n', mat2str(rowNorms,6));
            end

            % Check each column has norm 1.
            colNorms = sqrt(sum(DCM.^2, 1));
            if all(abs(colNorms - 1) < tol)
                fprintf('Column unit vector constraints satisfied.\n');
            else
                fprintf('Column unit vector constraints NOT satisfied. Column norms: %s\n', mat2str(colNorms,6));
            end

            % Check orthogonality constraints

            % For row vectors, DCM * DCM' should equal the identity.
            rowOrth = DCM * DCM';
            errRow = max(max(abs(rowOrth - eye(3))));
            if errRow < tol
                fprintf('Row orthogonality constraint satisfied. Max error: %e\n', errRow);
            else
                fprintf('Row orthogonality constraint NOT satisfied. Max error: %e\n', errRow);
            end

            % For column vectors, DCM' * DCM should equal the identity.
            colOrth = DCM' * DCM;
            errCol = max(max(abs(colOrth - eye(3))));
            if errCol < tol
                fprintf('Column orthogonality constraint satisfied. Max error: %e\n', errCol);
            else
                fprintf('Column orthogonality constraint NOT satisfied. Max error: %e\n', errCol);
            end
        end

        %% -> Euler Axis And Angle Only
        % DCM from Euler Axis and Euler Angle
        function DCM = eulerAxisAngleToDCM(obj,e, theta, convention)
            % eulerAxisAngleToDCM calculates the Direction Cosine Matrix (DCM)
            % based on a given Euler axis and rotation angle.
            %
            % ASSUMPTIONS:
            %   - By default, assumes column vector convention.
            %   - If 'row' convention is specified, the output is transposed.
            %
            % INPUTS:
            %   e          - [3x1] Unit vector representing the Euler axis
            %   theta      - Scalar, rotation angle in radians around the Euler axis
            %   convention - (Optional) String specifying the output format:
            %                - 'col' (default) returns DCM in column vector convention.
            %                - 'row' returns the transposed DCM.
            %
            % OUTPUT:
            %   DCM - [3x3] Direction Cosine Matrix
            %         - If 'col' (default), returns the standard DCM.
            %         - If 'row', returns the transposed DCM (equivalent to C matrix).
            %
            % USAGE EXAMPLE:
            %   e = [0; 0; 1];
            %   theta = pi/4;
            %   DCM_col = AttitudeDeterminationLibrary.eulerAxisAngleToDCM(e, theta, 'col');
            %   DCM_row = AttitudeDeterminationLibrary.eulerAxisAngleToDCM(e, theta, 'row');


            % Ensure the axis vector e is normalized
            e = e / norm(e);

            % Extract components of the Euler axis
            e1 = e(1);
            e2 = e(2);
            e3 = e(3);

            % Calculate cosine and sine of the rotation angle
            c = cos(theta);
            s = sin(theta);

            % Calculate the components of the DCM
            DCM = [c + (1 - c)*e1^2, (1 - c)*e1*e2 + s*e3, (1 - c)*e1*e3 - s*e2;
                (1 - c)*e1*e2 - s*e3, c + (1 - c)*e2^2, (1 - c)*e2*e3 + s*e1;
                (1 - c)*e1*e3 + s*e2, (1 - c)*e2*e3 - s*e1, c + (1 - c)*e3^2];

            % Check convention and transpose if necessary
            if nargin > 2 && strcmpi(convention, 'row')
                DCM = DCM';
            end
        end

        % Euler Axis and Euler Angle from DCM
        function [e, theta] = dcmToEulerAxisAngle(obj,A)
            % dcmToEulerAxisAngle converts a Direction Cosine Matrix (DCM) to the Euler axis and angle representation
            %
            % INPUT:
            %   A - 3x3 Direction Cosine Matrix
            %
            % OUTPUT:
            %   e     - 3x1 unit vector representing the Euler axis
            %   theta - Rotation angle in radians around the Euler axis

            % Calculate the rotation angle theta
            theta = acos((trace(A) - 1) / 2);

            % Check for special cases
            if cos(theta) == 1
                % Case: theta = 0
                disp('Theta is 0, A is an identity matrix, and e does not exist.');
                e = [];
                return;
            elseif cos(theta) == -1
                % Case: theta = pi
                disp('Theta is pi, special case where A = -I + 2*e*e^T.');

                % We need to normalize any non-zero column of A + I to find e
                ApI = A + eye(3);  % A + I

                [max_val, max_idx] = max(abs(ApI(:)));  % Find the largest element

                if max_val == 0
                    error('Cannot determine the axis, matrix A appears to be invalid.');
                end

                % Extract the column corresponding to the largest element in A plus
                % identity

                [row, col] = ind2sub(size(ApI), max_idx);
                e = ApI(:,col) / norm(ApI(:,col));
                return;
            elseif -1 < cos(theta) && cos(theta) < 1
                % Normal case: -1 < cos(theta) < 1
                e = (1 / (2 * sin(theta))) * [A(2,3) - A(3,2);
                    A(3,1) - A(1,3);
                    A(1,2) - A(2,1)];
                return;
            else
                error('Unexpected value for theta, check the input matrix A.');
            end
        end

        %% -> Euler Angles Only

        % Symbolic DCM From Euler Angles Rotation Sequence
        function DCM = dcmFromEulerAngleSeq_Sym(obj, seq, angles, convention)
            % dcmFromEulerAngleSeq_Sym returns the symbolic DCM for a given Euler rotation sequence.
            %
            %   DCM = dcmFromEulerAngleSeq_Sym_C(seq, angles, convention) returns a 3x3 symbolic rotation matrix
            %   corresponding to the sequence of rotations defined by the vector seq.
            %
            %   For the row vector convention:
            %       The transformation is: v_B = v_R * C,
            %       and the composite DCM is built as: C = C1 * C2 * C3.
            %
            %   For the column vector convention:
            %       The transformation is: v_B = L * v_R,
            %       where L = C' (i.e., the transpose of the row convention result).
            %
            %   Inputs:
            %       seq        - 1x3 vector of rotation axes (each element must be 1, 2, or 3)
            %       angles     - 1x3 vector of symbolic rotation angles (e.g., [phi, theta, psi])
            %       convention - (Optional) A string, either 'row' or 'col'. Default is 'row'.
            %
            %   Output:
            %       DCM        - 3x3 symbolic rotation matrix. For 'row' convention, DCM = C;
            %                    for 'col' convention, DCM = L = C'.
            %
            %   Example:
            %       syms phi theta psi real;
            %       DCM_row = obj.dcmFromEulerAngleSeq_Sym_C([3 1 3], [phi, theta, psi], 'row');
            %       DCM_col = obj.dcmFromEulerAngleSeq_Sym_C([3 1 3], [phi, theta, psi], 'col');
            %

            % Default to row convention if not provided.
            if nargin < 4 || isempty(convention)
                convention = 'row';
            end

            % Validate that both seq and angles are vectors of length 3.
            if numel(seq) ~= 3 || numel(angles) ~= 3
                error('Both seq and angles must be vectors of length 3.');
            end

            % Compute the composite symbolic DCM using the row convention.
            % That is, we form C = C1 * C2 * C3 such that v_B = v_R * C.
            DCM = sym(eye(3));
            for k = 1:3
                a = angles(k);
                switch seq(k)
                    case 1  % Rotation about x-axis
                        R_row = [1,      0,       0;
                            0, cos(a), -sin(a);
                            0, sin(a),  cos(a)];
                    case 2  % Rotation about y-axis
                        R_row = [ cos(a), 0, sin(a);
                            0,      1,      0;
                            -sin(a), 0, cos(a)];
                    case 3  % Rotation about z-axis
                        R_row = [cos(a), -sin(a), 0;
                            sin(a),  cos(a), 0;
                            0,           0, 1];
                    otherwise
                        error('Invalid axis. Each element of seq must be 1, 2, or 3.');
                end
                DCM = simplify(DCM * R_row);
            end

            % If the column convention is requested, simply transpose the row result.
            if strcmpi(convention, 'col')
                DCM = DCM.';
            elseif ~strcmpi(convention, 'row')
                error('Invalid convention. Use "row" or "col".');
            end
        end

        % DCM From Euler Angles Rotation Sequence - Body Centered
        function DCM = dcmFromEulerAngleSeq(obj, seq, angles, convention)
            % dcmFromEulerAngleSeq_Num returns the numerical DCM for a given Euler
            % rotation sequence.
            %
            %   DCM = dcmFromEulerAngleSeq_Num(seq, angles, convention)
            %
            %   INPUTS:
            %       seq        - 1x3 vector of rotation axes (each element must be 1, 2, or 3).
            %                    For example: [1 2 3] or [1 3 1].
            %       angles     - 1x3 vector of rotation angles (in radians). For instance,
            %                    [phi, theta, psi].
            %       convention - (Optional) A string, either 'row' or 'col'. Default is 'row'.
            %
            %   OUTPUT:
            %       DCM        - 3x3 numerical rotation matrix.
            %
            %   DESCRIPTION:
            %       The function builds the composite rotation matrix as:
            %           DCM = R1 * R2 * R3,
            %       where Rk is the rotation matrix about axis seq(k) by angle angles(k).
            %
            %       For the 'row' convention, the transformation is:
            %           v_B = v_R * DCM.
            %       For the 'col' convention, the final result is transposed.


            % Default to 'row' convention if not provided.
            if nargin < 3 || isempty(convention)
                convention = 'row';
            end

            % Validate that both seq and angles are vectors of length 3.
            if numel(seq) ~= 3 || numel(angles) ~= 3
                error('Both seq and angles must be vectors of length 3.');
            end

            % Initialize the DCM as an identity matrix.
            DCM = eye(3);

            % Loop over each rotation to form the composite DCM.
            for k = 1:3
                a = angles(k);
                switch seq(k)
                    case 1  % Rotation about x-axis
                        R = [1,      0,       0;
                            0, cos(a), -sin(a);
                            0, sin(a),  cos(a)];
                    case 2  % Rotation about y-axis
                        R = [ cos(a), 0, sin(a);
                            0,      1,      0;
                            -sin(a), 0, cos(a)];
                    case 3  % Rotation about z-axis
                        R = [cos(a), -sin(a), 0;
                            sin(a),  cos(a), 0;
                            0,           0, 1];
                    otherwise
                        error('Invalid axis. Each element of seq must be 1, 2, or 3.');
                end
                DCM = DCM * R;
            end

            % Apply the requested convention.
            if strcmpi(convention, 'col')
                DCM = DCM.';
            elseif ~strcmpi(convention, 'row')
                error('Invalid convention. Use "row" or "col".');
            end
        end


        function DCM = dcmFromSingleEulerAngle(obj, axis, angle, convention)
            % dcmFromSingleEulerAngle returns the rotation matrix for a given axis and angle.
            %
            %   DCM = dcmFromSingleEulerAngle(axis, angle, convention) returns a 3x3
            %   rotation matrix (DCM) based on the given rotation axis and angle.
            %
            %   Inputs:
            %       axis       - Rotation axis (1 = x-axis, 2 = y-axis, 3 = z-axis)
            %       angle      - Rotation angle in radians
            %       convention - 'col' for column vector convention (default)
            %                    'row' for row vector convention (returns L')
            %
            %   Output:
            %       L - The 3x3 rotation matrix.
            %
            %   Example:
            %       L = dcmFromSingleEulerAngle(3, pi/4, 'row') % Returns DCM for row-vector convention
            %

            % Compute cosine and sine of the angle
            c = cos(angle);
            s = sin(angle);

            % Generate the DCM for the given axis
            switch axis
                case 1  % Rotation about x-axis
                    DCM = [1, 0, 0;
                        0, c, s;
                        0, -s, c];
                case 2  % Rotation about y-axis
                    DCM = [ c, 0, -s;
                        0, 1,  0;
                        s, 0,  c];
                case 3  % Rotation about z-axis
                    DCM = [ c, s, 0;
                        -s, c, 0;
                        0, 0, 1];
                otherwise
                    error('Invalid rotation axis. Axis must be 1, 2, or 3.');
            end

            % If row convention is selected, return the transpose of L
            if nargin < 3 || strcmpi(convention, 'col')
                % Default: Column vector convention (return as-is)
                return;
            elseif strcmpi(convention, 'row')
                DCM = DCM'; % Transpose for row convention
            else
                error('Invalid convention. Use "col" for column or "row" for row.');
            end
        end



        % DCM Via Single Euler Angle and Axis - Row Vector
        % Convention
        function C = dcmSingleEulerAngle_2(obj,axis, angle)
            % dcmSingleEulerAngle returns the rotation matrix for a given axis and angle
            % ASSUMES:
            %   Row vector convention, so it outputs C matrix
            % INPUTS:
            %   axis  - Rotation axis (1 = x-axis, 2 = y-axis, 3 = z-axis)
            %   angle - Rotation angle in radians
            %
            % OUTPUT:
            %   L - The 3x3 rotation matrix in column vector format

            c = cos(angle);
            s = sin(angle);

            switch axis
                case 1  % Rotation about x-axis
                    C = [1, 0, 0;
                        0, c, -s;
                        0, s, c];
                case 2  % Rotation about y-axis
                    C = [c, 0, s;
                        0, 1, 0;
                        -s, 0, c];
                case 3  % Rotation about z-axis
                    C = [c, -s, 0;
                        s, c, 0;
                        0, 0, 1];
                otherwise
                    error('Invalid rotation axis. Axis must be 1, 2, or 3.');
            end
        end


        %% -> DCM from Space Rotations Only (Initial Frame)
        function dcm = dcmFromSpaceRotations(obj, seq, angles, convention)
            % spaceDCM   Compute the space-fixed (extrinsic) direction cosine matrix (DCM)
            %            for a given rotation sequence and corresponding angles.
            %
            %   dcm = spaceRotationsDCM(seq, angles)
            %   dcm = spaceRotationsDCM(seq, angles, convention)
            %
            %   Inputs:
            %       seq        - 1x3 vector representing the rotation sequence, e.g., [1 2 3],
            %                    [2 3 1], [1 2 1], [3 2 3], etc.
            %       angles     - 1x3 vector of rotation angles (in radians) corresponding to the
            %                    rotation about each axis in the sequence.
            %       convention - (optional) string specifying the output convention. If the value
            %                    is 'col', then the transpose of the computed DCM is returned.
            %                    Default is to return the matrix as computed.
            %
            %   Output:
            %       dcm        - 3x3 Direction Cosine Matrix.
            %
            %   Supported rotation sequences:
            %       Space-three (Tait–Bryan):  1-2-3, 2-3-1, 3-1-2, 1-3-2, 2-1-3, 3-2-1
            %       Space-two   (Proper Euler): 1-2-1, 1-3-1, 2-1-2, 2-3-2, 3-1-3, 3-2-3
            %
            %   Example:
            %       % Compute the DCM for a 1-2-3 rotation with angles [theta1, theta2, theta3]
            %       theta1 = pi/6; theta2 = pi/4; theta3 = pi/3;
            %       dcm = spaceDCM([1 2 3], [theta1, theta2, theta3], 'col');
            %

            % Check input sizes
            if numel(seq) ~= 3 || numel(angles) ~= 3
                error('Both the rotation sequence and angles must be vectors of length 3.');
            end

            % Set default convention if not provided
            if nargin < 3
                convention = 'row';
            end

            % Precompute sines and cosines for the three angles
            c1 = cos(angles(1));  c2 = cos(angles(2));  c3 = cos(angles(3));
            s1 = sin(angles(1));  s2 = sin(angles(2));  s3 = sin(angles(3));

            % Convert sequence vector to a string for easy matching in the switch-case
            seqStr = sprintf('%d%d%d', seq(1), seq(2), seq(3));

            switch seqStr
                case '123'
                    % Space-three: 1-2-3
                    dcm = [ c2*c3,             s1*s2*c3 - s3*c1,   c1*s2*c3 + s3*s1;
                        c2*s3,             s1*s2*s3 + c3*c1,   c1*s2*s3 - c3*s1;
                        -s2,                s1*c2,             c1*c2 ];
                case '231'
                    % Space-three: 2-3-1
                    dcm = [ c2*c3,             s1*s2*c3 - s3*c1,   c1*s2*c3 + s3*s1;
                        c2*s3,             s1*s2*s3 + c3*c1,   c1*s2*s3 - c3*s1;
                        -s2,                s1*c2,             c1*c2 ];
                case '312'
                    % Space-three: 3-1-2
                    dcm = [ c1*c2,            -s2,               s1*c2;
                        c1*s2*c3 + s3*s1,  c2*c3,             s1*s2*c3 - s3*c1;
                        c1*s2*s3 - c3*s1,  c2*s3,             s1*s2*s3 + c3*c1 ];
                case '132'
                    % Space-three: 1-3-2
                    dcm = [ s1*s2*s3 + c3*c1,   c1*s2*s3 - c3*s1,   c2*s3;
                        s1*c2,             c1*c2,            -s2;
                        s1*s2*c3 - s3*c1,   c1*s2*c3 + s3*s1,   c2*c3 ];
                case '213'
                    % Space-three: 2-1-3
                    dcm = [ -s1*s2*s3 + c3*c1,   -c2*s3,    c1*s2*s3 + c3*s1;
                        s1*s2*c3 + s3*c1,    c2*c3,   -c1*s2*c3 + s3*s1;
                        -s1*c2,             s2,       c1*c2 ];
                case '321'
                    % Space-three: 3-2-1
                    dcm = [ c1*c2,            -s1*c2,       s2;
                        c1*s2*s3 + c3*s1,  -s1*s2*s3 + c3*c1, -c2*s3;
                        -c1*s2*c3 + s3*s1,   s1*s2*c3 + s3*c1, c2*c3 ];
                case '121'
                    % Space-two: 1-2-1
                    dcm = [ c2,       s1*s2,         c1*s2;
                        s2*s3,   -s1*c2*s3 + c3*c1, -c1*c2*s3 - c3*s1;
                        -s2*c3,    s1*c2*c3 + s3*c1,  c1*c2*c3 - s3*s1 ];
                case '131'
                    % Space-two: 1-3-1
                    dcm = [ c2,        -c1*s2,        s1*s2;
                        s2*c3,     c1*c2*c3 - s3*s1, -s1*c2*c3 - s3*c1;
                        s2*s3,     c1*c2*s3 + c3*s1, -s1*c2*s3 + c3*c1 ];
                case '212'
                    % Space-two: 2-1-2
                    dcm = [ -s1*c2*s3 + c3*c1,   s2*s3,   c1*c2*s3 + c3*s1;
                        s1*s2,             c2,     -c1*s2;
                        -s1*c2*c3 - s3*c1,   s2*c3,   c1*c2*c3 - s3*s1 ];
                case '232'
                    % Space-two: 2-3-2
                    dcm = [ c1*c2*c3 - s3*s1,   -s2*c3,   s1*c2*c3 + s3*c1;
                        c1*s2,             c2,       s1*s2;
                        -c1*c2*s3 - c3*s1,   s2*s3,   -s1*c2*s3 + c3*c1 ];
                case '313'
                    % Space-two: 3-1-3
                    dcm = [ -s1*c2*s3 + c3*c1,   -c1*c2*s3 - c3*s1,   s2*s3;
                        s1*c2*c3 + s3*c1,    c1*c2*c3 - s3*s1,   -s2*c3;
                        s1*s2,              c1*s2,              c2 ];
                case '323'
                    % Space-two: 3-2-3
                    dcm = [ c1*c2*c3 - s3*s1,   -s1*c2*c3 - s3*c1,    s2*c3;
                        c1*c2*s3 + c3*s1,   -s1*c2*s3 + c3*c1,    s2*s3;
                        -c1*s2,             s1*s2,              c2 ];
                otherwise
                    error('Rotation sequence not recognized. Please check your sequence.');
            end

            % If the convention is 'col', return the transpose of the computed DCM.
            if strcmpi(convention, 'col')
                dcm = dcm.';
            end

        end

        function dcm = dcmFromSpaceRotations_Sym(obj, seq, thetas)
            % dcmFromSpaceRotations_Sym returns a symbolic DCM from a given rotation sequence.
            %
            %   dcm = obj.dcmFromSpaceRotations_Sym(seq, thetas)
            %
            %   Inputs:
            %       seq    - 1x3 numeric vector specifying the rotation sequence.
            %                For example, [1 2 3] for a Tait–Bryan sequence or [1 3 1] for a proper Euler sequence.
            %
            %       thetas - 1x3 vector of symbolic rotation angles [theta1, theta2, theta3].
            %
            %   Output:
            %       dcm    - 3x3 symbolic direction–cosine matrix corresponding to the specified sequence.
            %

            % Extract individual symbolic angles
            theta1 = thetas(1);
            theta2 = thetas(2);
            theta3 = thetas(3);

            % Precompute symbolic sine and cosine functions
            c1 = cos(theta1); s1 = sin(theta1);
            c2 = cos(theta2); s2 = sin(theta2);
            c3 = cos(theta3); s3 = sin(theta3);

            % Check if the rotation sequence is proper Euler (first and third axes equal)
            if seq(1) == seq(3)
                % Use proper Euler (Space-two) formulas
                if isequal(seq, [1,2,1])
                    % Space-two: 1-2-1
                    dcm = [ c2,         s1*s2,            c1*s2;
                        s2*s3,     -s1*c2*s3 + c3*c1,  -c1*c2*s3 - c3*s1;
                        -s2*c3,      s1*c2*c3 + s3*c1,   c1*c2*c3 - s3*s1 ];
                elseif isequal(seq, [1,3,1])
                    % Space-two: 1-3-1
                    dcm = [ c2,         -c1*s2,            s1*s2;
                        s2*c3,      c1*c2*c3 - s3*s1,  -s1*c2*c3 - s3*c1;
                        s2*s3,      c1*c2*s3 + c3*s1,  -s1*c2*s3 + c3*c1 ];
                elseif isequal(seq, [2,1,2])
                    % Space-two: 2-1-2
                    dcm = [ -s1*c2*s3 + c3*c1,   s2*s3,         c1*c2*s3 + c3*s1;
                        s1*s2,              c2,           -c1*s2;
                        -s1*c2*c3 - s3*c1,   s2*c3,         c1*c2*c3 - s3*s1 ];
                elseif isequal(seq, [2,3,2])
                    % Space-two: 2-3-2
                    dcm = [ c1*c2*c3 - s3*s1,    -s2*c3,        s1*c2*c3 + s3*c1;
                        c1*s2,              c2,             s1*s2;
                        -c1*c2*s3 - c3*s1,    s2*s3,        -s1*c2*s3 + c3*c1 ];
                elseif isequal(seq, [3,1,3])
                    % Space-two: 3-1-3
                    dcm = [ -s1*c2*s3 + c3*c1,   -c1*c2*s3 - c3*s1,   s2*s3;
                        s1*c2*c3 + s3*c1,     c1*c2*c3 - s3*s1,   -s2*c3;
                        s1*s2,               c1*s2,              c2 ];
                elseif isequal(seq, [3,2,3])
                    % Space-two: 3-2-3
                    dcm = [ c1*c2*c3 - s3*s1,    -s1*c2*c3 - s3*c1,  s2*c3;
                        c1*c2*s3 + c3*s1,    -s1*c2*s3 + c3*c1,  s2*s3;
                        -c1*s2,             s1*s2,             c2 ];
                else
                    error('Proper Euler sequence not recognized.');
                end
            else
                % Use Tait–Bryan (Space-three) formulas
                if isequal(seq, [1,2,3])
                    % Space-three: 1-2-3
                    dcm = [ c2*c3,         s1*s2*c3 - s3*c1,   c1*s2*c3 + s3*s1;
                        c2*s3,         s1*s2*s3 + c3*c1,   c1*s2*s3 - c3*s1;
                        -s2,           s1*c2,             c1*c2 ];
                elseif isequal(seq, [2,3,1])
                    % Space-three: 2-3-1
                    dcm = [ c1*c2,         -s2,              s1*c2;
                        c1*s2*c3 + s3*s1, c2*c3,          s1*s2*c3 - s3*c1;
                        c1*s2*s3 - c3*s1, c2*s3,          s1*s2*s3 + c3*c1 ];
                elseif isequal(seq, [3,1,2])
                    % Space-three: 3-1-2
                    dcm = [ s1*s2*s3 + c3*c1,   c1*s2*s3 - c3*s1,   c2*s3;
                        s1*c2,              c1*c2,             -s2;
                        s1*s2*c3 - s3*c1,   c1*s2*c3 + s3*s1,   c2*c3 ];
                elseif isequal(seq, [1,3,2])
                    % Space-three: 1-3-2
                    dcm = [ c2*c3,            -c1*s2*c3 + s3*s1,    s1*s2*c3 + s3*c1;
                        s2,               c1*c2,               -s1*c2;
                        -c2*s3,           c1*s2*s3 + c3*s1,     -s1*s2*s3 + c3*c1 ];
                elseif isequal(seq, [2,1,3])
                    % Space-three: 2-1-3
                    dcm = [ -s1*s2*s3 + c3*c1,  -c2*s3,             c1*s2*s3 + c3*s1;
                        s1*s2*c3 + s3*c1,   c2*c3,              -c1*s2*c3 + s3*s1;
                        -s1*c2,            s2,                 c1*c2 ];
                elseif isequal(seq, [3,2,1])
                    % Space-three: 3-2-1
                    dcm = [ c1*c2,             -s1*c2,             s2;
                        c1*s2*s3 + c3*s1,   -s1*s2*s3 + c3*c1,   -c2*s3;
                        -c1*s2*c3 + s3*s1,  s1*s2*c3 + s3*c1,    c2*c3 ];
                else
                    error('Sequence not recognized.');
                end
            end

        end




        % Euler Angles from a 313 Matrix
        function [phi, theta, psi] = dcm_to_euler_313(obj,A)
            % dcm_to_euler_313 Extracts 3-1-3 Euler angles from a Direction Cosine Matrix (DCM).
            %
            % Syntax:
            %   [phi, theta, psi] = dcm_to_euler_313(A)
            %
            % Inputs:
            %   A - [3x3] Direction Cosine Matrix representing rotation from body frame to inertial frame.
            %
            % Outputs:
            %   phi   - Rotation angle about the first Z-axis (in radians).
            %   theta - Rotation angle about the X-axis (in radians).
            %   psi   - Rotation angle about the second Z-axis (in radians).
            %
            % Description:
            %   This function computes the 3-1-3 Euler angles (\( \phi, \theta, \psi \)) from a given DCM.
            %   It handles the singularity at \( \theta = \pi/2 \) by setting \( \phi \) to zero when necessary.
            %
            % Example:
            %   A = eye(3); % No rotation
            %   [phi, theta, psi] = dcm_to_euler_313(A); % Should return [0; 0; 0]

            % Check for singularity (gimbal lock)
            if abs(A(3,1)) ~= 1
                theta = acos(A(3,1));
                phi = atan2(A(3,2), A(3,3));
                psi = atan2(A(2,1), A(1,1));
            else
                % Gimbal lock occurs
                psi = 0; % Can set psi to zero
                if A(3,1) == -1
                    theta = pi/2;
                    phi = psi + atan2(A(1,2), A(1,3));
                else
                    theta = -pi/2;
                    phi = -psi + atan2(-A(1,2), -A(1,3));
                end
            end
        end


        %% .
        %% .
        %% .
        %% - - - - - - KINEMATIC EQUATIONS - RATES OF CHANGE - - - - - -
        %% -> DCM Rate of Change
        function Cdot = DCMdot(obj, C, omega, convention)
            % DCMdot Computes the time derivative of the Direction Cosine Matrix (DCM)
            % using Poisson's Kinematical Equations.
            %
            %   Cdot = obj.DCMdot(C, omega, convention)
            %
            %   INPUTS:
            %       C          - [3x3] Direction Cosine Matrix representing the current orientation.
            %                    For the 'row' convention, C is defined such that v_B = v_R * C.
            %                    For the 'col' convention, C is defined such that v_B = C * v_R.
            %       omega      - [3x1] Angular velocity vector expressed in the body (B) frame.
            %       convention - (Optional) A string, either 'row' or 'col'. Default is 'row'.
            %
            %   OUTPUT:
            %       Cdot       - [3x3] Time derivative of the Direction Cosine Matrix.
            %
            %   DESCRIPTION:
            %       For the row convention, the kinematic equation is:
            %           Cdot = C * skew_symmetric(omega)
            %       For the column convention, the transformation is the transpose of the row
            %       formulation. Thus, the derivative is computed as:
            %           Cdot = (C * skew_symmetric(omega))'

            % Default to 'row' convention if not provided.
            if nargin < 4 || isempty(convention)
                convention = 'row';
            end

            % Compute the skew-symmetric matrix of the angular velocity vector.
            omega_tilde = obj.skew_symmetric(omega);

            % Compute the DCM derivative based on the specified convention.
            switch lower(convention)
                case 'row'
                    Cdot = C * omega_tilde;
                case 'col'
                    Cdot = (C * omega_tilde)';  % This yields the column-convention result.
                otherwise
                    error('Invalid convention. Use "row" or "col".');
            end
        end

        %% -> Euler Angles Rates of Change Body Sequence
        function theta_dot = getEulerRatesBodySeq(obj, omega, EulerAngles, seq)
            % getEulerRates  Compute the Euler angle rates using body-fixed formulas.
            %
            %   theta_dot = obj.getEulerRates(omega, euler, seq)
            %
            %   Inputs:
            %     omega - 3x1 angular velocity vector in the body frame: [w1; w2; w3]
            %     euler - 3x1 Euler angles vector: [theta1; theta2; theta3]
            %     seq   - 1x3 vector defining the Euler sequence (e.g., [1 2 3], [2 1 3], etc.)
            %
            %   Output:
            %     theta_dot - 3x1 vector of Euler angle rates.

            % Extract Euler angles (theta1 is unused in these formulas)
            theta2 = EulerAngles(2);
            theta3 = EulerAngles(3);

            % Extract body-frame angular velocities
            w1 = omega(1);
            w2 = omega(2);
            w3 = omega(3);

            % Precompute common trigonometric terms
            c2 = cos(theta2);
            s2 = sin(theta2);
            c3 = cos(theta3);
            s3 = sin(theta3);

            % Initialize output
            theta_dot = zeros(3,1);

            % Determine if the sequence is proper (Body-two) or  (Body-three)
            if seq(1) == seq(3)
                % Proper Euler angles (Body-two)
                switch mat2str(seq)
                    case '[1 2 1]'
                        theta_dot(1) = (w2*s3 + w3*c3) / s2;
                        theta_dot(2) = w2*c3 - w3*s3;
                        theta_dot(3) = w1 - ((w2*s3 + w3*c3)*c2)/s2;
                    case '[1 3 1]'
                        theta_dot(1) = (-w2*c3 + w3*s3) / s2;
                        theta_dot(2) = w2*s3 + w3*c3;
                        theta_dot(3) = w1 + ((w2*c3 - w3*s3)*c2)/s2;
                    case '[2 1 2]'
                        theta_dot(1) = (w1*s3 - w3*c3) / s2;
                        theta_dot(2) = w1*c3 + w3*s3;
                        theta_dot(3) = ((-w1*s3 + w3*c3)*c2)/s2 + w2;
                    case '[2 3 2]'
                        theta_dot(1) = (w1*c3 + w3*s3) / s2;
                        theta_dot(2) = -w1*s3 + w3*c3;
                        theta_dot(3) = -((w1*c3 + w3*s3)*c2)/s2 + w2;
                    case '[3 1 3]'
                        theta_dot(1) = (w1*s3 + w2*c3) / s2;
                        theta_dot(2) = w1*c3 - w2*s3;
                        theta_dot(3) = -((w1*s3 + w2*c3)*c2)/s2 + w3;
                    case '[3 2 3]'
                        theta_dot(1) = (-w1*c3 + w3*s3) / s2;
                        theta_dot(2) = w1*s3 + w3*c3;
                        theta_dot(3) = ((w1*c3 - w3*s3)*c2)/s2 + w3;
                    otherwise
                        error('Body-fixed proper Euler sequence %s not implemented.', mat2str(seq));
                end
            else
                %  angles (Body-three)
                switch mat2str(seq)
                    case '[1 2 3]'
                        theta_dot(1) = (w1*c3 - w2*s3) / c2;
                        theta_dot(2) = w1*s3 + w2*c3;
                        theta_dot(3) = ((-w1*c3 + w2*s3)*s2)/c2 + w3;
                    case '[2 3 1]'
                        theta_dot(1) = (w2*c3 - w3*s3) / c2;
                        theta_dot(2) = w2*s3 + w3*c3;
                        theta_dot(3) = w1 + ((-w2*c3 + w3*s3)*s2)/c2;
                    case '[3 1 2]'
                        theta_dot(1) = (-w1*s3 + w2*c3) / c2;
                        theta_dot(2) = w1*c3 + w3*s3;
                        theta_dot(3) = ((w1*s3 - w3*c3)*s2)/c2 + w2;
                    case '[1 3 2]'
                        theta_dot(1) = (w1*c3 + w3*s3) / c2;
                        theta_dot(2) = -w1*s3 + w3*c3;
                        theta_dot(3) = ((w1*c3 + w3*s3)*s2)/c2 + w2;
                    case '[2 1 3]'
                        theta_dot(1) = (w1*s3 + w2*c3) / c2;
                        theta_dot(2) = w1*c3 - w2*s3;
                        theta_dot(3) = ((w1*s3 + w2*c3)*s2)/c2 + w3;
                    case '[3 2 1]'
                        theta_dot(1) = (w2*s3 + w3*c3) / c2;
                        theta_dot(2) = w2*c3 - w3*s3;
                        theta_dot(3) = w1 + ((w2*s3 + w3*c3)*s2)/c2;
                    otherwise
                        error('Body-fixed  sequence %s not implemented.', mat2str(seq));
                end
            end

        end


        % -> Angular Velocity from Euler Angle Rates in Body Sequence of Rotations
        function omega_Bfrm = omegaFromEulerRatesBodySeq(obj, EulerAngles, theta_dot, seq)
            % omegaFromEulerRatesBodySeq  Computes the angular velocity vector (in the body frame)
            %                             from Euler–angle rates for a body–fixed rotation sequence.
            %
            %   omega_Bfrm = obj.omegaFromEulerRatesBodySeq(EulerAngles, theta_dot, seq)
            %
            %   INPUTS:
            %       EulerAngles - [3x1] Euler angles [theta1; theta2; theta3] in radians.
            %       theta_dot   - [3x1] Euler–angle rates [theta1_dot; theta2_dot; theta3_dot].
            %       seq         - [1x3] Numeric vector specifying the body rotation sequence.
            %                     For example, [1 2 3] for a body-three 1-2-3 sequence,
            %                     [3 2 3] for a body-two 3-2-3 sequence, etc.
            %
            %   OUTPUT:
            %       omega_Bfrm  - [3x1] Angular velocity vector in the body frame.
            %
            %   EXAMPLE:
            %       seq = [1 2 3];  % body-three: 1-2-3
            %       EulerAngles = [0.1; 0.2; 0.3];
            %       theta_dot   = [0.01; 0.02; 0.03];
            %       omega_B = obj.omegaFromEulerRatesBodySeq(EulerAngles, theta_dot, seq);

            % Extract individual Euler angles and Euler–angle rates
            theta1   = EulerAngles(1);
            theta2   = EulerAngles(2);
            theta3   = EulerAngles(3);

            theta1_dot = theta_dot(1);
            theta2_dot = theta_dot(2);
            theta3_dot = theta_dot(3);

            % Compute sine and cosine values for the Euler angles
            s1 = sin(theta1);  c1 = cos(theta1);
            s2 = sin(theta2);  c2 = cos(theta2);
            s3 = sin(theta3);  c3 = cos(theta3);

            % Initialize omega components
            omega1 = 0;
            omega2 = 0;
            omega3 = 0;

            % ----------------------------
            % BODY–THREE (Tait–Bryan) Sequences
            % ----------------------------
            if isequal(seq, [1 2 3])
                % Body-three: 1-2-3
                omega1 = theta1_dot*c2*c3 + theta2_dot*s3;
                omega2 = -theta1_dot*c2*s3 + theta2_dot*c3;
                omega3 = theta1_dot*s2 + theta3_dot;

            elseif isequal(seq, [2 3 1])
                % Body-three: 2-3-1
                omega1 = theta1_dot*s2 + theta3_dot;
                omega2 = theta1_dot*c2*c3 + theta2_dot*s3;
                omega3 = -theta1_dot*c2*s3 + theta2_dot*c3;

            elseif isequal(seq, [3 1 2])
                % Body-three: 3-1-2
                omega1 = -theta1_dot*c2*s3 + theta2_dot*c3;
                omega2 = theta1_dot*s2 + theta3_dot;
                omega3 = theta1_dot*c2*c3 + theta2_dot*s3;

            elseif isequal(seq, [1 3 2])
                % Body-three: 1-3-2
                omega1 = theta1_dot*c2*c3 - theta2_dot*s3;
                omega2 = -theta1_dot*s2 + theta3_dot;
                omega3 = theta1_dot*c2*s3 + theta2_dot*c3;

            elseif isequal(seq, [2 1 3])
                % Body-three: 2-1-3
                omega1 = theta1_dot*c2*s3 + theta2_dot*c3;
                omega2 = theta1_dot*c2*c3 - theta2_dot*s3;
                omega3 = -theta1_dot*s2 + theta3_dot;

            elseif isequal(seq, [3 2 1])
                % Body-three: 3-2-1
                omega1 = -theta1_dot*s2 + theta3_dot;
                omega2 = theta1_dot*c2*s3 + theta2_dot*c3;
                omega3 = theta1_dot*c2*c3 - theta2_dot*s3;

                % ----------------------------
                % BODY–TWO (Proper Euler) Sequences
                % ----------------------------
            elseif isequal(seq, [1 2 1])
                % Body–two: 1-2-1
                omega1 = theta1_dot*c2 + theta3_dot;
                omega2 = theta1_dot*s2*s3 + theta2_dot*c3;
                omega3 = theta1_dot*s2*c3 - theta2_dot*s3;

            elseif isequal(seq, [1 3 1])
                % Body–two: 1-3-1
                omega1 = theta1_dot*c2 + theta3_dot;
                omega2 = -theta1_dot*s2*c3 + theta2_dot*s3;
                omega3 = theta1_dot*s2*s3 + theta2_dot*c3;

            elseif isequal(seq, [2 1 2])
                % Body–two: 2-1-2
                omega1 = theta1_dot*s2*s3 + theta2_dot*c3;
                omega2 = theta1_dot*c2 + theta3_dot;
                omega3 = -theta1_dot*s2*c3 + theta2_dot*s3;

            elseif isequal(seq, [2 3 2])
                % Body–two: 2-3-2
                omega1 = theta1_dot*s2*c3 - theta2_dot*s3;
                omega2 = theta1_dot*c2 + theta3_dot;
                omega3 = theta1_dot*s2*s3 + theta2_dot*c3;

            elseif isequal(seq, [3 1 3])
                % Body–two: 3-1-3
                omega1 = theta1_dot*s2*s3 + theta2_dot*c3;
                omega2 = theta1_dot*s2*c3 - theta2_dot*s3;
                omega3 = theta1_dot*c2 + theta3_dot;

            elseif isequal(seq, [3 2 3])
                % Body–two: 3-2-3
                omega1 = -theta1_dot*s2*c3 + theta2_dot*s3;
                omega2 = theta1_dot*s2*s3 + theta2_dot*c3;
                omega3 = theta1_dot*c2 + theta3_dot;

            else
                error('Invalid Euler angle sequence. Use a valid body-fixed 1x3 numeric sequence.');
            end

            % Construct the angular velocity vector (3x1) in the body frame
            omega_Bfrm = [omega1; omega2; omega3];
        end




        % -> Angular Velocity from Euler Angle Rates in Space Sequence of Rotations
        function omega_Bfrm = omegaFromEulerRatesSpaceSeq(obj, EulerAngles, theta_dot, seq)
            % omegaFromEulerRatesSpaceSeq Computes the angular velocity vector from
            % Euler–angle rates for a given rotation sequence.
            %
            %   omega_Bfrm = obj.omegaFromEulerRatesSpaceSeq(theta_dot, EulerAngles, seq)
            %
            %   INPUTS:
            %       theta_dot  - [3x1] Vector of Euler–angle rates [theta1_dot; theta2_dot; theta3_dot]
            %       EulerAngles- [3x1] Euler angles [theta1; theta2; theta3] in radians.
            %       seq        - [1x3] Numeric vector specifying the rotation sequence.
            %                    For example:
            %                       [1 2 3] for a 1-2-3 sequence,
            %                       [1 3 1] for a proper Euler (1-3-1) sequence, etc.
            %
            %   OUTPUT:
            %       omega_Bfrm - [3x1] Angular velocity vector computed from the Euler rates.
            %

            % Extract individual Euler angles and Euler–angle rates
            theta1   = EulerAngles(1);
            theta2   = EulerAngles(2);
            theta3   = EulerAngles(3);

            theta1_dot = theta_dot(1);
            theta2_dot = theta_dot(2);
            theta3_dot = theta_dot(3);

            % Compute sine and cosine values for the Euler angles
            s1 = sin(theta1);  c1 = cos(theta1);
            s2 = sin(theta2);  c2 = cos(theta2);
            s3 = sin(theta3);  c3 = cos(theta3);

            % Initialize omega components
            omega1 = 0;
            omega2 = 0;
            omega3 = 0;

            % Select the appropriate transformation based on the sequence vector.
            % For the two "Space-three" sequences, we use the space-fixed formulas.
            if isequal(seq, [1 2 3])
                % Here we choose the space-fixed (Space–three) 1-2-3 formula:
                omega1 = theta1_dot - theta3_dot * s2;
                omega2 = theta2_dot * c1 + theta3_dot * s1 * c2;
                omega3 = -theta2_dot * s1 + theta3_dot * c1 * c2;

            elseif isequal(seq, [2 3 1])
                % Space–three: 2-3-1 formula:
                omega1 = -theta2_dot * s1 + theta3_dot * c1 * c2;
                omega2 = theta1_dot - theta3_dot * s2;
                omega3 = theta2_dot * c1 + theta3_dot * s1 * c2;

                % Otherwise, assume a body-fixed transformation.
            elseif isequal(seq, [3 1 2])
                % Body–three: 3-1-2
                omega1 = -theta1_dot * c2 * s3 + theta2_dot * c3;
                omega2 = theta1_dot * s2 + theta3_dot;
                omega3 = theta1_dot * c2 * c3 + theta2_dot * s3;

            elseif isequal(seq, [1 3 2])
                % Body–three: 1-3-2
                omega1 = theta1_dot * c2 * c3 - theta2_dot * s3;
                omega2 = -theta1_dot * s2 + theta3_dot;
                omega3 = theta1_dot * c2 * s3 + theta2_dot * c3;

            elseif isequal(seq, [2 1 3])
                % Body–three: 2-1-3
                omega1 = theta1_dot * c2 * s3 + theta2_dot * c3;
                omega2 = theta1_dot * c2 * c3 - theta2_dot * s3;
                omega3 = -theta1_dot * s2 + theta3_dot;

            elseif isequal(seq, [3 2 1])
                % Body–three: 3-2-1
                omega1 = -theta1_dot * s2 + theta3_dot;
                omega2 = theta1_dot * c2 * s3 + theta2_dot * c3;
                omega3 = theta1_dot * c2 * c3 - theta2_dot * s3;

                % Now the proper Euler (Body–two) sequences:
            elseif isequal(seq, [1 2 1])
                % Body–two: 1-2-1
                omega1 = theta1_dot * c2 + theta3_dot;
                omega2 = theta1_dot * s2 * s3 + theta2_dot * c3;
                omega3 = theta1_dot * s2 * c3 - theta2_dot * s3;

            elseif isequal(seq, [1 3 1])
                % Body–two: 1-3-1
                omega1 = theta1_dot * c2 + theta3_dot;
                omega2 = -theta1_dot * s2 * c3 + theta2_dot * s3;
                omega3 = theta1_dot * s2 * s3 + theta2_dot * c3;

            elseif isequal(seq, [2 1 2])
                % Body–two: 2-1-2
                omega1 = theta1_dot * s2 * s3 + theta2_dot * c3;
                omega2 = theta1_dot * c2 + theta3_dot;
                omega3 = -theta1_dot * s2 * c3 + theta2_dot * s3;

            elseif isequal(seq, [2 3 2])
                % Body–two: 2-3-2
                omega1 = theta1_dot * s2 * c3 - theta2_dot * s3;
                omega2 = theta1_dot * c2 + theta3_dot;
                omega3 = theta1_dot * s2 * s3 + theta2_dot * c3;

            elseif isequal(seq, [3 1 3])
                % Body–two: 3-1-3
                omega1 = theta1_dot * s2 * s3 + theta2_dot * c3;
                omega2 = theta1_dot * s2 * c3 - theta2_dot * s3;
                omega3 = theta1_dot * c2 + theta3_dot;

            elseif isequal(seq, [3 2 3])
                % Body–two: 3-2-3
                omega1 = -theta1_dot * s2 * c3 + theta2_dot * s3;
                omega2 = theta1_dot * s2 * s3 + theta2_dot * c3;
                omega3 = theta1_dot * c2 + theta3_dot;

            else
                error('Invalid Euler angle sequence. Use a valid 1x3 numeric sequence.');
            end

            % Construct the angular velocity vector (3x1)
            omega_Bfrm = [omega1; omega2; omega3];
        end


        function theta_dot = getEulerRatesSpaceSeq(obj, omega, EulerAngles, seq)
            % getEulerRatesSpaceSeq Computes the time derivatives of Euler angles
            % given the angular velocity in the space frame.
            %
            %   theta_dot = obj.eulerAngleRates(omega, theta, sequence)
            %
            %   INPUTS:
            %       omega    - [3x1] Angular velocity vector in the space frame.
            %       theta    - [3x1] Euler angles [theta1; theta2; theta3] in radians.
            %       sequence - [1x3] numeric vector specifying the Euler angle sequence.
            %                  For example:
            %                     [1 2 3] for a 1-2-3 sequence,
            %                     [2 3 1] for a 2-3-1 sequence,
            %
            %   OUTPUT:
            %       theta_dot - [3x1] Time derivatives of the Euler angles
            %                   [theta1_dot; theta2_dot; theta3_dot].

            % Extract individual Euler angles
            theta1 = EulerAngles(1);
            theta2 = EulerAngles(2);
            theta3 = EulerAngles(3);

            % Compute sine and cosine values for the Euler angles
            s1 = sin(theta1);   c1 = cos(theta1);
            s2 = sin(theta2);   c2 = cos(theta2);
            s3 = sin(theta3);   c3 = cos(theta3);

            % Extract components of the angular velocity
            omega1 = omega(1);
            omega2 = omega(2);
            omega3 = omega(3);

            % Compute Euler angle rates based on the provided sequence vector
            if isequal(seq, [1 2 3])
                % For sequence 1-2-3 ()
                theta1_dot = omega1 + ((omega2 * s1 + omega3 * c1) * s2) / c2;
                theta2_dot = omega2 * c1 - omega3 * s1;
                theta3_dot = (omega2 * s1 + omega3 * c1) / c2;
            elseif isequal(seq, [2 3 1])
                % For sequence 2-3-1 ()
                theta1_dot = ((omega1 * c1 + omega3 * s1) * s2) / c2 + omega2;
                theta2_dot = -omega1 * s1 + omega3 * c1;
                theta3_dot = (omega1 * c1 + omega3 * s1) / c2;
            elseif isequal(seq, [3 1 2])
                % For sequence 3-1-2 ()
                theta1_dot = (omega1 * s1 + omega2 * c1) * s2 / c2 + omega3;
                theta2_dot = omega1 * c1 - omega2 * s1;
                theta3_dot = (omega1 * s1 + omega2 * c1) / c2;
            elseif isequal(seq, [1 3 2])
                % For sequence 1-3-2 ()
                theta1_dot = omega1 + ((-omega2 * c1 + omega3 * s1) * s2) / c2;
                theta2_dot = omega2 * s1 + omega3 * c1;
                theta3_dot = (omega2 * c1 - omega3 * s1) / c2;
            elseif isequal(seq, [2 1 3])
                % For sequence 2-1-3 ()
                theta1_dot = ((omega1 * c1 - omega3 * s1) * s2) / c2 + omega2;
                theta2_dot = omega1 * c1 + omega3 * s1;
                theta3_dot = (-omega1 * s1 + omega3 * c1) / c2;
            elseif isequal(seq, [3 2 1])
                % For sequence 3-2-1 (
                theta1_dot = ((- omega1 * c1 + omega2 * s1) * s2) / c2 + omega3;
                theta2_dot = omega1 * c1 + omega2 * c1;
                theta3_dot = (omega1 * c1 - omega2 * s1) / c2;
            elseif isequal(seq, [1 2 1])
                % For sequence 1-2-1 (Proper Euler)
                theta1_dot = omega1 - ((omega2 * s1 + omega3 * c1) * c2) / s2;
                theta2_dot = omega2 * c1 - omega3 * s1;
                theta3_dot = (omega2 * s1 + omega3 * c1) / s2;
            elseif isequal(seq, [1 3 1])
                % For sequence 1-3-1 (Proper Euler)
                theta1_dot = omega1 + ((omega2 * c1 - omega3 * s1) * c2) / s2;
                theta2_dot = omega2 * s1 + omega3 * c1;
                theta3_dot = (-omega2 * c1 + omega3 * s1) / s2;
            else
                error('Invalid Euler angle sequence. Use a valid space rotation sequence.');
            end

            % Construct the Euler angle rates vector
            theta_dot = [theta1_dot; theta2_dot; theta3_dot];
        end

        %% -> Angular Velocity in Body Frame from Quaternion Derivative
        function omega = quatDot2omega(obj, q, q_dot)
            % quat2omega Compute the angular velocity from the quaternion derivative.
            %
            %   omega = obj.quat2omega(q, q_dot)
            %
            %   This function computes the 4x1 angular velocity vector from the time
            %   derivative of a quaternion using the transformation matrix E.
            %
            %   Inputs:
            %       q     - 4x1 quaternion vector [ε1; ε2; ε3; ε4], where the first three
            %               elements represent the vector part and the fourth element is
            %               the scalar part.
            %
            %       q_dot - 4x1 time derivative of the quaternion [dε1/dt; dε2/dt; dε3/dt; dε4/dt]
            %
            %   Output:
            %       omega - 4x1 angular velocity vector, computed as
            %
            %                   omega = 2 * (q_dot' * E)' = 2 * E' * q_dot
            %
            %               where the transformation matrix E is defined by:
            %
            %                   E = [  ε4,  -ε3,   ε2,   ε1;
            %                         ε3,   ε4,  -ε1,   ε2;
            %                        -ε2,   ε1,   ε4,   ε3;
            %                        -ε1,  -ε2,  -ε3,   ε4 ]
            %
            %   Note:
            %       The last element of omega is theoretically zero from the computation,
            %       but this function does not force it to be zero.

            % Ensure q and q_dot are row vectors.
            q = q(:)';
            q_dot = q_dot(:)';

            % Extract quaternion components.
            q1 = q(1);
            q2 = q(2);
            q3 = q(3);
            q4 = q(4);

            % Define the transformation matrix E using the quaternion components.
            E = [ q4,   -q3,   q2,   q1;
                q3,    q4,  -q1,   q2;
                -q2,    q1,   q4,   q3;
                -q1,   -q2,  -q3,   q4 ];

            % Compute the angular velocity using the relation:
            omega = 2 * (q_dot * E);

        end

        %% -> Quaternion Rate of Change
        function qDot = qDot(obj, q, omega)
            % qDot Computes the time derivative of a quaternion
            % using the kinematic equations for quaternion propagation.
            %
            % This function determines how the quaternion evolves over time given the
            % angular velocity in the body frame.
            %
            % INPUTS:
            %   q     - [1x4] Quaternion in the body frame, where:
            %           - q(1:3) corresponds to epsilon_bar (vector portion)
            %           - q(4) is epsilon_4 (scalar portion)
            %   omega - [1x3] Angular velocity vector in the body frame
            %
            % OUTPUT:
            %   q_dot - [1x4] Time derivative of the quaternion (rate of change)
            %           - q_dot(1:3) corresponds to epsilon_bar_dot
            %           - q_dot(4) is epsilon_4_dot

            % Ensure q and omega are row vectors
            q = q(:)';
            omega = omega(:)';

            % Extract q13 (vector part) and q4 (scalar part)
            q13 = q(1:3);
            q4 = q(4);

            % Compute epsilon_bar_dot using the first equation
            q13_dot = (1/2) * (q4 * omega + cross(q13, omega));

            % Compute epsilon_4_dot using the second equation
            q4_dot = -(1/2) * dot(omega, q13);

            % Construct the quaternion derivative as a row vector
            qDot = [q13_dot, q4_dot];
        end


        function qDot = qDot_2(obj, q, omega)
            % qDot_2  Computes the time derivative of a quaternion given the
            %                angular velocity in the body frame.
            %
            %   qDot = obj.qDotFromOmega(q, omega)
            %
            %   INPUTS:
            %       q     - [1x4] row vector representing the quaternion, where
            %               q(1:3) is the vector part, and q(4) is the scalar part.
            %       omega - [1x3] row vector of the angular velocity [w1, w2, w3] in the body frame.
            %
            %   OUTPUT:
            %       qDot  - [1x4] row vector for the quaternion derivative:
            %               qDot(1:3) is the vector part derivative,
            %               qDot(4)   is the scalar part derivative.
            %
            %   Equations:
            %     dq1/dt = 0.5*( w1*q4 - w2*q3 + w3*q2 )
            %     dq2/dt = 0.5*( w1*q3 + w2*q4 - w3*q1 )
            %     dq3/dt = 0.5*( -w1*q2 + w2*q1 + w3*q4 )
            %     dq4/dt = 0.5*( w1*q1 + w2*q2 + w3*q3 )

            % Ensure q and omega are row vectors
            q     = q(:).';
            omega = omega(:).';

            % Extract quaternion components
            q1 = q(1);
            q2 = q(2);
            q3 = q(3);
            q4 = q(4);

            % Extract angular velocity components
            w1 = omega(1);
            w2 = omega(2);
            w3 = omega(3);

            % Compute quaternion time derivatives
            q1_dot = 0.5*( w1*q4 - w2*q3 + w3*q2 );
            q2_dot = 0.5*( w1*q3 + w2*q4 - w3*q1 );
            q3_dot = 0.5*( -w1*q2 + w2*q1 + w3*q4 );
            q4_dot = 0.5*( w1*q1 + w2*q2 + w3*q3 );

            % Combine into a single row vector
            qDot = [q1_dot, q2_dot, q3_dot, q4_dot];
        end
        %% .
        %% .
        %% .
        %% - - - - - - DYADICS OPERATIONS - - - - - -
        %% -> Rotation Dyadic from Euler Axis / Angle
        function R = rotationDyadic_fromAxisAngle(obj, lambda, theta)
            % rotationDyadic_fromAxisAngle Computes the rotation dyadic based on the input parameters.
            % Syntax:
            %   R = rotation_dyadic(lambda, theta)
            %
            % Inputs:
            %   lambda - [3x1] Unit vector defining the axis of rotation.
            %   theta  - Scalar, rotation angle in radians.
            %
            % Outputs:
            %   R - [3x3] Rotation dyadic matrix. Just like L matrix
            %   but. If you want C matrix for row vector format, you'll
            %   need to tranpose the output of this function.
            %
            % Description:
            %   This function computes the rotation dyadic matrix using the formula:
            %       R = U * cos(theta) - U_cross * sin(theta) + (lambda * lambda') * (1 - cos(theta))
            %   where:
            %       - U is the identity matrix (3x3),
            %       - U_cross is the cross-product matrix of lambda.
            %   The rotation dyadic matrix represents a simple rotation transformation
            %   in 3D space.

            % Ensure lambda is a unit vector
            lambda = lambda / norm(lambda);

            % Compute the identity matrix (U)
            U = eye(3);

            % Compute the cross-product matrix of lambda
            U_cross = [   0,     -lambda(3),  lambda(2);
                lambda(3),     0,    -lambda(1);
                -lambda(2),  lambda(1),     0 ];

            % Compute the dyadic product of lambda
            lambda_dyadic = lambda * lambda';

            % Compute the rotation dyadic matrix
            R = U * cos(theta) - U_cross * sin(theta) + lambda_dyadic * (1 - cos(theta));
        end

        %% .
        %% .
        %% .
        %% - - - - - - EULER AXIS / ANGLE OPERATIONS - - - - - -
        function [theta, lambda_hat] = quat2eulerAxisAngle(obj, q)
            % quat2eulerAxisAngle  Converts a quaternion into a single rotation
            %                      (Euler axis and Euler angle).
            %
            %   [theta, lambda_hat] = obj.quat2eulerAxisAngle(q)
            %
            %   This function takes a quaternion q = [e1, e2, e3, e4], where e4 is the
            %   scalar part, and computes the equivalent single rotation:
            %       - theta (rotation angle in radians),
            %       - lambda_hat (3x1 unit vector representing the rotation axis).
            %
            %   Inputs:
            %       q - 4x1 (or 1x4) quaternion, where q(1:3) is the vector part
            %           and q(4) is the scalar part.
            %
            %   Outputs:
            %       theta     - Rotation angle in radians, in [0, 2*pi].
            %       lambda_hat- 3x1 unit vector representing the Euler axis. If the
            %                   rotation angle is near zero, lambda_hat defaults to
            %                   [1; 0; 0].
            %
            %   The formulas are:
            %       theta      = 2 * acos(e4)
            %       lambda_hat = e_vec / sin(theta/2),   (if sin(theta/2) > some small tol)


            % Ensure q is a column vector
            q = q(:);

            % Extract scalar and vector parts
            q13 = q(1:3);    % vector part
            q4   = q(4);     % scalar part

            % Compute the rotation angle theta
            theta = 2 * acos(q4);

            % Compute the Euler axis
            lambda_hat = q13 / sin(theta/2);
        end




        %% .
        %% .
        %% .
        %% - - - - - - QUATERNION OPERATIONS - - - - - -

        % Successive Rotations with Quaternions
        function qOut = sequentialQuatRotations(obj, qABp, qBpB, C_ABp, C_BpB)
            % sequentialQuatRotations  Multiply two quaternions for successive rotations,
            %                          optionally transforming one quaternion vector part
            %                          if direction cosine matrices are provided.
            %
            %   qOut = obj.sequentialQuatRotations(qABp, qBpB, C_ABp, C_BpB)
            %
            %   INPUTS:
            %     qABp - 1×4 quaternion representing rotation from frame A to B'
            %            [vABp(1), vABp(2), vABp(3), sABp] where sABp is the scalar part.
            %     qBpB - 1×4 quaternion representing rotation from frame B' to B
            %            [vBpB(1), vBpB(2), vBpB(3), sBpB] where sBpB is the scalar part.
            %     C_ABp - 3×3 direction cosine matrix from frame A to frame B'
            %     C_BpB - 3×3 direction cosine matrix from frame B' to frame B
            %
            %   OUTPUT:
            %     qOut - 1×4 quaternion representing the overall rotation from A to B.
            %            The first three elements are the vector part, and the fourth
            %            element is the scalar part: [ vOut, sOut ].
            %
            %   If C_ABp and C_BpB are provided (and non-empty), the intermediate vector
            %   vBpB is expressed in the "origin" frame before computing the standard
            %   quaternion composition rule. Otherwise, it is assumed both quaternions
            %   are expressed in the same frame, so no additional transformation is done.


            % Extract vector and scalar parts from each quaternion
            vABp = qABp(1:3);  % vector part of qABp
            vABp = vABp(:);    % ensure column vector
            sABp = qABp(4);    % scalar part of qABp

            vBpB = qBpB(1:3);  % vector part of qBpB
            vBpB = vBpB(:);    % ensure column vector
            sBpB = qBpB(4);    % scalar part of qBpB

            % Optionally convert the intermediate-to-final frame quaternion vector
            % into the original frame if direction cosine matrices are provided
            if nargin >= 5 && ~isempty(C_ABp) && ~isempty(C_BpB)
                % Transform vBpB from B' frame to A frame
                % vBpB -> vBpB_inA
                vBpB = (vBpB' * C_BpB' * C_ABp')';
            end

            % Compute the resulting vector part using the standard quaternion composition
            vOut = (vABp * sBpB) + (vBpB * sABp) + cross(vBpB, vABp);

            % Compute the resulting scalar part
            sOut = (sABp * sBpB) - dot(vABp, vBpB);

            % Combine into one quaternion [vector, scalar]
            qOut = [vOut; sOut];
        end

        % Full Quaternion from Quaternion Vector Portion
        function[q, q4] = getFullQuatFromVec(obj, q_vec)
            % getFullQuatFromVec Constructs a full quaternion given its vector part.
            %
            % Syntax:
            %   q = get_quaternion(q_vec)
            %
            % Inputs:
            %   q_vec - [1x3] or [3x1] Vector part of the quaternion (q1, q2, q3).
            %
            % Outputs:
            %   q - [1x4] Full quaternion [q1, q2, q3, q4], where q4 is the scalar part.
            %
            % Description:
            %   This function calculates the full quaternion given its vector portion.
            %   It computes the scalar component using the quaternion norm constraint:
            %       q4 = sqrt(1 - norm(q_vec)^2)
            %   The function assumes the quaternion is normalized.

            % Ensure q_vec is a row vector
            q_vec = q_vec(:)';

            % Compute the scalar part
            q4 = sqrt(1 - norm(q_vec)^2);

            % Construct the full quaternion
            q = [q_vec, q4];
        end

        % Quaternion to DCM  - Method 1
        function DCM = quat2dcm(obj, q, convention)
            % quat2dcm  Converts a quaternion to a Direction Cosine Matrix (DCM)
            %
            %   A_BI = quat2dcm(convention, q)
            %
            %   INPUTS:
            %     convention - a string specifying the DCM convention. Use:
            %                  'row' : the DCM is assumed to be in row-vector convention.
            %                          In this case, the computed DCM is transposed before output.
            %                  'col' : the DCM is assumed to be in column-vector convention.
            %                          The computed DCM is output directly.
            %
            %     q          - a 4x1 quaternion vector [q1; q2; q3; q4], where q4 is the scalar part.
            %                  The quaternion is assumed to be nonzero.
            %
            %   OUTPUT:
            %     DCM       - a 3x3 Direction Cosine Matrix. For the 'row' convention the final
            %                  output is the transpose of the computed matrix.
            %
            %   DESCRIPTION:
            %     This function first normalizes the input quaternion and then computes the DCM
            %     using the following formulas (which assume row-vector convention):
            %
            %       q4 = 0.5 * sqrt(1 + L(1,1) + L(2,2) + L(3,3));
            %       q1 = (L(3,2) - L(2,3)) / (4 * q4);
            %       q2 = (L(1,3) - L(3,1)) / (4 * q4);
            %       q3 = (L(2,1) - L(1,2)) / (4 * q4);


            % Quaternion must be normalized
            q = q / norm(q);  % Normalize the quaternion to unit length

            % Extract the components of the quaternion
            q1 = q(1);
            q2 = q(2);
            q3 = q(3);
            q4 = q(4);  % q4 is the scalar part

            % Compute the DCM based on the quaternion
            DCM = [
                q1^2 - q2^2 - q3^2 + q4^2, 2 * (q1 * q2 + q3 * q4), 2 * (q1 * q3 - q2 * q4);
                2 * (q2 * q1 - q3 * q4), -q1^2 + q2^2 - q3^2 + q4^2, 2 * (q2 * q3 + q1 * q4);
                2 * (q3 * q1 + q2 * q4), 2 * (q3 * q2 - q1 * q4), -q1^2 - q2^2 + q3^2 + q4^2
                ];

            % Adjust the output based on the convention:
            %   For 'row' convention, output the transpose of the computed matrix.
            %   For 'col' convention, output the computed matrix as is.
            if strcmpi(convention, 'row')
                DCM = DCM';
            elseif ~strcmpi(convention, 'col')
                error('Invalid convention specified. Use ''row'' or ''col''.');
            end

        end

        function DCM = quat2dcm_2(obj, q, convention)
            % quat2dcm_2  Converts a quaternion into a Direction Cosine Matrix (DCM)
            %
            %   C_out = quat2dcm_2(convention, q)
            %
            %   INPUTS:
            %       convention - a string specifying the desired output convention:
            %                    'row' : return the DCM in row-vector convention.
            %                    'col' : return the DCM in column-vector convention.
            %
            %       q          - a 4x1 (or 1x4) quaternion vector [q1; q2; q3; q4],
            %                    where q4 is the scalar part. The quaternion must be
            %                    normalized for correct results.
            %
            %   OUTPUT:
            %       C_out      - a 3x3 Direction Cosine Matrix corresponding to the
            %                    quaternion. The formulas below assume a row-vector
            %                    convention. If the desired output is column-vector,
            %                    the computed matrix is transposed.
            %
            %   DESCRIPTION:
            %     The DCM is computed using the following formulas (which assume the
            %     DCM is constructed in row-vector convention):


            % Extract quaternion components
            q1 = q(1); % x-component
            q2 = q(2); % y-component
            q3 = q(3); % z-component
            q4 = q(4); % scalar component

            % Initialize DCM
            DCM = zeros(3,3);

            % Populate DCM elements using quaternion-to-DCM formulas
            DCM(1,1) = 1 - 2*q2^2 - 2*q3^2;
            DCM(1,2) = 2*(q1*q2 - q3*q4);
            DCM(1,3) = 2*(q3*q1 + q2*q4);

            DCM(2,1) = 2*(q1*q2 + q3*q4);
            DCM(2,2) = 1 - 2*q3^2 - 2*q1^2;
            DCM(2,3) = 2*(q2*q3 - q1*q4);

            DCM(3,1) = 2*(q3*q1 - q2*q4);
            DCM(3,2) = 2*(q2*q3 + q1*q4);
            DCM(3,3) = 1 - 2*q1^2 - 2*q2^2;

            % Adjust the output based on the desired convention.
            % For 'col', we swap the off-diagonal differences by returning the transpose.
            if strcmpi(convention, 'col')
                DCM = DCM';
            elseif strcmpi(convention, 'row')
                DCM = DCM;
            else
                error('Invalid convention. Use ''row'' or ''col''.');
            end

        end

        function [q4, q] = getQuaternionScalar(obj, q_vec)
            % compute_quaternion_scalar Computes the scalar part of a quaternion.
            %
            % Syntax:
            %   q4 = compute_quaternion_scalar(q_vec)
            %
            % Inputs:
            %   q_vec - [3x1] Vector part of the quaternion (q1, q2, q3).
            %
            % Outputs:
            %   q4 - Scalar part of the quaternion.
            %
            % Description:
            %   This function calculates the scalar component (q4) of a quaternion
            %   given its vector part, using the quaternion norm constraint:
            %       q4 = sqrt(1 - norm(q_vec)^2)
            %   The function assumes the quaternion is normalized.

            % Ensure q_vec is a column vector
            q_vec = q_vec(:);

            % Compute the scalar part
            q4 = sqrt(1 - norm(q_vec)^2);
            q = [q_vec;q4];
        end

        % DCM to Quaternion
        function q = dcm2quat(obj, DCM, convention)
            % dcm2quat  Converts a Direction Cosine Matrix (DCM) to a quaternion.
            %
            %   q = dcm2quat(convention, A)
            %
            %   INPUTS:
            %     convention - a string specifying the DCM convention:
            %                  'col' : the DCM is in column-vector format
            %                  'row' : the DCM is in row-vector format
            %                          (i.e. the antisymmetric differences are swapped)
            %
            %     DCM          - a 3x3 Direction Cosine Matrix representing the rotation
            %                  from body frame to inertial frame.
            %
            %   OUTPUT:
            %     q          - a 4x1 quaternion [q1; q2; q3; q4], where q4 is the scalar part.
            %
            %   DESCRIPTION:
            %     This function uses a method (similar to that described by Crassidis)
            %     which forms four candidate quaternion vectors based on the trace of the DCM.
            %     The candidate with the largest norm is then chosen and normalized.


            % Compute the trace of the DCM
            trA = trace(DCM);

            if strcmpi(convention, 'col')
                % Candidate quaternion vectors (standard formulas for column-vector DCM)
                q1_vec = [ 1 + 2*DCM(1,1) - trA;
                    DCM(1,2) + DCM(2,1);
                    DCM(1,3) + DCM(3,1);
                    DCM(2,3) - DCM(3,2) ];
                q2_vec = [ DCM(2,1) + DCM(1,2);
                    1 + 2*DCM(2,2) - trA;
                    DCM(2,3) + DCM(3,2);
                    DCM(3,1) - DCM(1,3) ];
                q3_vec = [ DCM(3,1) + DCM(1,3);
                    DCM(3,2) + DCM(2,3);
                    1 + 2*DCM(3,3) - trA;
                    DCM(1,2) - DCM(2,1) ];
                q4_vec = [ DCM(2,3) - DCM(3,2);
                    DCM(3,1) - DCM(1,3);
                    DCM(1,2) - DCM(2,1);
                    1 + trA ];
            elseif strcmpi(convention, 'row')
                % Candidate quaternion vectors for row-vector convention:
                % Swap the antisymmetric terms by changing their sign.
                q1_vec = [ 1 + 2*DCM(1,1) - trA;
                    DCM(1,2) + DCM(2,1);
                    DCM(1,3) + DCM(3,1);
                    -(DCM(2,3) - DCM(3,2)) ];
                q2_vec = [ DCM(2,1) + DCM(1,2);
                    1 + 2*DCM(2,2) - trA;
                    DCM(2,3) + DCM(3,2);
                    -(DCM(3,1) - DCM(1,3)) ];
                q3_vec = [ DCM(3,1) + DCM(1,3);
                    DCM(3,2) + DCM(2,3);
                    1 + 2*DCM(3,3) - trA;
                    -(DCM(1,2) - DCM(2,1)) ];
                q4_vec = [ -(DCM(2,3) - DCM(3,2));
                    -(DCM(3,1) - DCM(1,3));
                    -(DCM(1,2) - DCM(2,1));
                    1 + trA ];
            else
                error('Invalid convention. Use ''row'' or ''col''.');
            end

            % Compute the norms of the candidate quaternion vectors.
            candidateNorms = [ norm(q1_vec), norm(q2_vec), norm(q3_vec), norm(q4_vec) ];

            % Find the candidate with the largest norm.
            [~, max_idx] = max(candidateNorms);

            % Select the corresponding quaternion candidate.
            switch max_idx
                case 1
                    q = q1_vec;
                case 2
                    q = q2_vec;
                case 3
                    q = q3_vec;
                case 4
                    q = q4_vec;
            end

            % Normalize the selected quaternion to unit length.
            q = q / norm(q);
        end

        function q = dcm2quat_2(obj, DCM, convention)
            % dcm2quatAttitude  Converts a 3x3 Direction Cosine Matrix (DCM) to a quaternion.
            %
            %   q = dcm2quatAttitude(convention, L)
            %
            %   INPUTS:
            %     convention - a string that specifies the DCM convention:
            %                  'row' indicates that L is in row-vector convention
            %                  'col' indicates that L is in column-vector convention
            %
            %     L          - a 3x3 Direction Cosine Matrix.
            %
            %   OUTPUT:
            %     q          - a 4x1 quaternion [q1; q2; q3; q4], where q4 is the scalar part.
            %

            if strcmpi(convention, 'row')
                % DCM is in row-vector convention; use the standard formulas.
                q4 = 0.5 * sqrt( 1 + DCM(1,1) + DCM(2,2) + DCM(3,3) );
                q1 = ( DCM(3,2) - DCM(2,3) ) / (4 * q4);
                q2 = ( DCM(1,3) - DCM(3,1) ) / (4 * q4);
                q3 = ( DCM(2,1) - DCM(1,2) ) / (4 * q4);
            elseif strcmpi(convention, 'col')
                % DCM is in column-vector convention; use the transposed differences.
                q4 = 0.5 * sqrt( 1 + DCM(1,1) + DCM(2,2) + DCM(3,3) );
                q1 = ( DCM(2,3) - DCM(3,2) ) / (4 * q4);
                q2 = ( DCM(3,1) - DCM(1,3) ) / (4 * q4);
                q3 = ( DCM(1,2) - DCM(2,1) ) / (4 * q4);
            else
                error('Invalid convention. Use ''row'' or ''col''.');
            end

            q = [q1; q2; q3; q4];
        end

        % Euler axis/angle to Quaternion

        function [q, qVec, qScalar] = axisAngleToQuaternion(obj, theta, lambda_hat)
            % axisAngleToQuaternion Converts an axis-angle representation to a quaternion.
            %
            %   [Q, qVec, qScalar] = obj.axisAngleToQuaternion(theta, lambda_hat)
            %
            %   This function computes the quaternion corresponding to a rotation of
            %   angle theta (radians) about a unit vector lambda_hat. The quaternion
            %   convention used here places the scalar part in the 4th component.
            %
            %   Inputs:
            %       theta      - Rotation angle in radians.
            %       lambda_hat - 3x1 unit vector (or 1x3) representing the Euler axis.
            %
            %   Outputs:
            %       Q         - 4x1 quaternion [epsilon_x; epsilon_y; epsilon_z; epsilon_4],
            %                   where the first three elements are the vector part and
            %                   the fourth is the scalar part.
            %       qVec      - 3x1 vector part of the quaternion, Q(1:3).
            %       qScalar   - Scalar part of the quaternion, Q(4).
            %

            % Ensure lambda_hat is a column vector
            lambda_hat = lambda_hat(:);

            % Compute the vector part (epsilon)
            halfTheta = theta / 2;
            qVec = lambda_hat .* sin(halfTheta);

            % Compute the scalar part (epsilon_4)
            qScalar = cos(halfTheta);

            % Combine into a single quaternion
            q = [qVec; qScalar];
        end

        % Quaternion Kinematics Equation
        function q_dot = quaternionKinematicsFixedOmega(obj,q, omega)
            % quaternionKinematics Computes the time derivative of the quaternion.
            %
            % Syntax:
            %   q_dot = quaternionKinematics(q, omega)
            %
            % Inputs:
            %   q      - [4x1] Quaternion vector [q1; q2; q3; q4], where q4 is the scalar part.
            %   omega  - [3x1] Angular velocity vector in the body frame [omega_x; omega_y; omega_z].
            %
            % Outputs:
            %   q_dot - [4x1] Time derivative of the quaternion [q1_dot; q2_dot; q3_dot; q4_dot].
            %
            % Description:
            %   This function calculates the quaternion derivatives based on the current quaternion
            %   and angular velocity. The quaternion kinematics equation is used:
            %
            %       q_dot = 0.5 * Xi(q) * omega
            %
            %   where Xi(q) is a matrix constructed from the quaternion components.
            %
            % Example:
            %   q = [0; 0; 0; 1]; % Initial quaternion (no rotation)
            %   omega = [1; 0; 0]; % Angular velocity about X-axis
            %   q_dot = quaternionKinematics(q, omega);



            % Quaternion kinematics equation: q_dot = 0.5 * Xi(q) * omega
            Xi_q = obj.Xi(q);
            q_dot = 0.5 * Xi_q * omega;
        end


        % Quaternion Dynamics Equation
        function state_dot = rigidBodyDynamicsQuaternions(obj,~, state, I_body, torque)
            % rigidBodyDynamics Computes the derivatives of the state vector for a torque-free rigid body.
            %
            % Syntax:
            %   state_dot = rigidBodyDynamics(~, state, I_body)
            %
            % Inputs:
            %   ~        - Placeholder for time variable (unused in this function).
            %   state    - [7x1] State vector containing:
            %               state(1:4) - Quaternion [q1; q2; q3; q4], where q4 is the scalar part.
            %               state(5:7) - Angular velocity vector in the body frame [omega_x; omega_y; omega_z].
            %   I_body   - [3x3] Inertia tensor matrix in the body frame, assumed to be diagonal.
            %
            % Outputs:
            %   state_dot - [7x1] Derivative of the state vector containing:
            %               state_dot(1:4) - Quaternion derivatives [q1_dot; q2_dot; q3_dot; q4_dot].
            %               state_dot(5:7) - Angular velocity derivatives [omega_x_dot; omega_y_dot; omega_z_dot].
            %
            % Description:
            %   This function calculates the time derivatives of the state vector for a rigid body
            %   undergoing torque-free rotation. It integrates both the quaternion kinematics and
            %   Euler's equations for rotational dynamics.
            %
            % Example:
            %   % Define inertia tensor
            %   I_body = diag([Ixx, Iyy, Izz]);
            %   % Define initial state vector
            %   state0 = [q0; omega0];
            %   % Compute state derivatives
            %   state_dot = rigidBodyDynamics(t, state0, I_body);



            % Check number of input arguments and set default torque if not provided
            if nargin < 5 || isempty(torque)
                torque = [0; 0; 0]; % Default: No external torque (Torque-free motion)
            end

            % Unpack state vector
            q = state(1:4);          % Quaternion [q1; q2; q3; q4], q4 is scalar part
            omega = state(5:7);      % Angular velocity in body frame [omega_x; omega_y; omega_z]

            % Quaternion kinematics
            q_dot = obj.quaternionKinematicsFixedOmega(q, omega);

            % Euler's equations (Torque-free dynamics)
            torque = [0; 0; 0]; % No external torque
            omega_dot = I_body \ (torque - cross(omega, I_body * omega));

            % Combine derivatives into state derivative vector
            state_dot = [q_dot; omega_dot];
        end

        % Quaternion Cross Matrix
        function q_cross = quaternionCrossMatrix(obj,q)
            % quaternionCrossMatrix Constructs the [q⊗] matrix for a given quaternion q.
            %
            % Inputs:
            %   q - [4x1] Quaternion vector [q1; q2; q3; q4], where q4 is the scalar part.
            %
            % Output:
            %   q_cross - [4x4] matrix representation of [q⊗]

            % Ensure the quaternion is a column vector
            q = q(:);

            % Extract the scalar and vector parts of the quaternion
            q1 = q(1);
            q2 = q(2);
            q3 = q(3);
            q4 = q(4); % Scalar part

            % Construct the [q⊗] matrix
            q_cross = [q4,  q3, -q2,  q1;
                -q3,  q4,  q1,  q2;
                q2, -q1,  q4,  q3;
                -q1, -q2, -q3,  q4];
        end

        % Quaternion product ⊗
        function result = quaternionProduct(obj, q, q_prime)
            % quaternionProduct Computes the quaternion product q ⊗ q_prime based on the formula.
            %
            % Inputs:
            %   q       - [4x1] Quaternion vector [q1; q2; q3; q4], where q4 is the scalar part.
            %   q_prime - [4x1] Quaternion vector [q1'; q2'; q3'; q4'], where q4' is the scalar part.
            %
            % Output:
            %   result  - [4x1] Quaternion result of the product q ⊗ q_prime.

            % Ensure both quaternions are column vectors
            q = q(:);
            q_prime = q_prime(:);

            % Extract the components of the quaternions
            q_v = q(1:3);      % Vector part of q
            q4 = q(4);         % Scalar part of q
            q_prime_v = q_prime(1:3); % Vector part of q_prime
            q4_prime = q_prime(4);    % Scalar part of q_prime

            % Compute the vector part of the result
            vector_part = q4 * q_prime_v + q4_prime * q_v - cross(q_v, q_prime_v);

            % Compute the scalar part of the result
            scalar_part = q4 * q4_prime - dot(q_v, q_prime_v);

            % Combine into the result quaternion
            result = [vector_part; scalar_part];
        end

        % - Math Operations with Quaternions
        function q_conj = quatconj(obj, q)
            % quatconj Computes the conjugate of a quaternion.
            %   q_conj = quatconj(q) returns the conjugate of quaternion q.
            %   Supports both row (1x4) and column (4x1) vectors.
            %
            %   Inputs:
            %       q - Quaternion vector [q1, q2, q3, q4] or [q1; q2; q3; q4]
            %           where q4 is the scalar part.
            %
            %   Outputs:
            %       q_conj - Conjugate quaternion with the same orientation as input.

            % Validate input size
            if ~(isrow(q) || iscolumn(q))
                error('Input quaternion must be a row or column vector.');
            end

            if length(q) ~= 4
                error('Input quaternion must have exactly 4 elements.');
            end

            % Compute conjugate based on input orientation
            if isrow(q)
                q_conj = [-q(1:3), q(4)];
            elseif iscolumn(q)
                q_conj = [-q(1:3); q(4)];
            end
        end

        function S = skew_symmetric(obj, v)
            % Constructs a skew-symmetric matrix from a vector
            S = [  0,   -v(3),  v(2);
                v(3),   0,   -v(1);
                -v(2), v(1),    0  ];
        end

        % Xi Matrix
        function Xi_q = Xi(obj,q)
            % Xi Constructs the Xi(q) matrix used in quaternion kinematics.
            %
            % Syntax:
            %   Xi_q = Xi(q)
            %
            % Inputs:
            %   q - [4x1] Quaternion vector [q1; q2; q3; q4], where q4 is the scalar part.
            %
            % Outputs:
            %   Xi_q - [4x3] Xi(q) matrix used in the quaternion kinematics equation.
            %
            % Description:
            %   The Xi(q) matrix is defined as:
            %
            %       Xi(q) = [
            %           -q1, -q2, -q3;
            %            q4, -q3,  q2;
            %            q3,  q4, -q1;
            %           -q2,  q1,  q4
            %       ]
            %
            %   This matrix is used to relate the angular velocity to the quaternion derivative.
            %
            % Example:
            %   q = [0.1; 0.2; 0.3; 0.9];
            %   Xi_q = Xi(q);

            % Constructs the Xi(q) matrix
            q1 = q(1);
            q2 = q(2);
            q3 = q(3);
            q4 = q(4); % Scalar part

            Xi_q = [
                q4, -q3,  q2;
                q3,  q4, -q1;
                -q2,  q1,  q4;
                -q1, -q2, -q3
                ];
        end

        %% .
        %% .
        %% .
        %% - - - - - - TRIG FUNCTIONS - - - - - -

        function angles = dualTrigInverse(obj, trigFunc, value)
            % dualTrigInverse: Returns all angles (in degrees) between -360 and 360
            % that satisfy the trigonometric equation for sine, cosine, or tangent.
            %
            % Usage:
            %   angles = dualTrigInverse(trigFunc, value)
            %
            % Inputs:
            %   trigFunc - A string: 'sin', 'cos', or 'tan'
            %   value    - The value for which you want to solve the equation.
            %
            % Outputs:
            %   angles   - A sorted vector containing all solutions in degrees within [-360, 360].


            angles = []; % initialize empty vector for solutions

            switch lower(trigFunc)
                case 'sin'
                    % Domain check: for sine, |value| must be <= 1.
                    if abs(value) > 1
                        error('For sine, value must be in [-1,1].');
                    end

                    % For sin(theta)=value the general solutions are:
                    %    theta = asind(value) + 360*k    and
                    %    theta = 180 - asind(value) + 360*k,   for any integer k.
                    theta0 = asind(value);  % principal value (in [-90,90])

                    % Loop over a few k-values; k = -2:2 is enough for the range [-360,360]
                    for k = -2:2
                        angle1 = theta0 + 360*k;
                        if (angle1 >= -360) && (angle1 <= 360)
                            angles(end+1) = angle1; %#ok<AGROW>
                        end
                        angle2 = 180 - theta0 + 360*k;
                        if (angle2 >= -360) && (angle2 <= 360)
                            angles(end+1) = angle2; %#ok<AGROW>
                        end
                    end

                case 'cos'
                    % Domain check: for cosine, |value| must be <= 1.
                    if abs(value) > 1
                        error('For cosine, value must be in [-1,1].');
                    end

                    % For cos(theta)=value the general solutions are:
                    %    theta = acosd(value) + 360*k    and
                    %    theta = -acosd(value) + 360*k,   for any integer k.
                    theta0 = acosd(value);  % principal value (in [0,180])

                    for k = -2:2
                        angle1 = theta0 + 360*k;
                        if (angle1 >= -360) && (angle1 <= 360)
                            angles(end+1) = angle1; %#ok<AGROW>
                        end
                        angle2 = -theta0 + 360*k;
                        if (angle2 >= -360) && (angle2 <= 360)
                            angles(end+1) = angle2; %#ok<AGROW>
                        end
                    end

                case 'tan'
                    % For tangent, value can be any real number.
                    % For tan(theta)=value the general solution is:
                    %    theta = atand(value) + 180*k,  for any integer k.
                    theta0 = atand(value);  % principal value (in [-90,90])

                    % For tan, period is 180°; loop over enough k-values to cover [-360,360]
                    for k = -3:3
                        angle = theta0 + 180*k;
                        if (angle >= -360) && (angle <= 360)
                            angles(end+1) = angle; %#ok<AGROW>
                        end
                    end

                otherwise
                    error('Unsupported trigonometric function. Use ''sin'', ''cos'', or ''tan''.');
            end

            % Remove any duplicate values (which can occur for special cases)
            angles = unique(angles);
            % Sort in ascending order
            angles = sort(angles);
        end

        %% .
        %% .
        %% .
        %% - - - - - - GRAVITY EQUATIONS - - - - - -

        % Center of gravity from a set of particles and one is attractive
        function [R_cg, F_net, F_hat, F_mag] = centerOfGravity(obj, pos_vector, m_vector, rP, mP, G)
            % centerOfGravity Computes the effective center of gravity (CG) of a system
            %                 of particles due to gravitational attraction by a particle P.
            %
            %   [x_cg, y_cg] = centerOfGravity(pos_vector, m_vector, rP, mP, G)
            %
            %   INPUTS:
            %       pos_vector - An N×2 matrix of (x,y) positions for the system particles.
            %       m_vector   - An N×1 vector of masses for the system particles.
            %       rP         - A 1×2 vector representing the (x,y) position of the attractor P.
            %       mP         - Scalar, mass of the attractor P.
            %       G          - Gravitational constant (scalar).
            %
            %
            %   The net gravitational force is computed by summing the forces acting on
            %   each particle due to P:
            %
            %       F_i = - (G*mP*m_i/||r_i||^3)*r_i,  where r_i = (position_i - rP)
            %
            %   The net force gives a line of action. The equivalent CG distance along that
            %   line is:
            %
            %       R_cg = sqrt((G*mP*m_B)/||F_net||),
            %
            %   where m_B = sum(m_vector). The center of gravity is then:
            %
            %       r_cg = rP + R_cg * (F_net/||F_net||).
            %
            %   Example:
            %       pos_vector = [4, -2; 11, -10; 6, -5; 14, 5];
            %       m_vector = [2.5; 2; 3; 6];
            %       rP = [15, -8];
            %       mP = 4; G = 1;
            %       [x_cg, y_cg] = centerOfGravity(pos_vector, m_vector, rP, mP, G);

            % Number of system particles
            numParticles = size(pos_vector, 1);

            % Compute net gravitational force from P on all particles
            F_net = [0; 0];
            for i = 1:numParticles
                r_i = pos_vector(i,:) - rP;  % vector from P to particle i
                F_i = - (G * mP * m_vector(i) / norm(r_i)^3) * r_i(:);
                F_net = F_net + F_i;
            end
            F_mag = norm(F_net);

            % Total mass of the attracted particles
            mB = sum(m_vector);

            % Compute equivalent CG distance along the line of action
            R_cg = sqrt((G * mP * mB) / F_mag);

            % Unit vector along net force (direction of line of action)
            F_hat = -F_net / F_mag;

        end

        % Center of mass of a set of particles
        function [x_cm, y_cm] = centerOfMass(pos_vector, m_vector)
            % centerOfMass Computes the center of mass of a system of particles
            %
            %   Inputs:
            %       pos_vector - Nx2 matrix, where each row contains [x, y] coordinates of a particle
            %       m_vector   - Nx1 vector of masses corresponding to each particle
            %
            %   Outputs:
            %       x_cm - x-coordinate of the center of mass
            %       y_cm - y-coordinate of the center of mass

            % Compute total mass
            M_total = sum(m_vector);

            % Compute center of mass coordinates
            x_cm = sum(pos_vector(:,1) .* m_vector) / M_total;
            y_cm = sum(pos_vector(:,2) .* m_vector) / M_total;
        end

        % Approximate Gravity force value up to f2 term
        function [F_approx, F_particle]= approxGravityForce_f2(mu_body, m, R, C, I1, I2, I3, a1, a2, a3)
            % approxGravityForce Computes the approximate gravitational force vector.
            %
            %   F = approxGravityForce(G, m_prime, m, R, C, I1, I2, I3, a1, a2, a3)
            %
            %   INPUTS:
            %       mu_body - Gravitational parameter of the planet / attracting body
            %       m       - Mass of the body on which the force acts.
            %       R       - Distance from the center of attracting body to C.M. of
            %       the attracted body
            %       C       - 3x3 Direction Cosine Matrix from inertial frame A to body
            %       frame B. Assumed to be in the row vector convention. Cij = ai * bj
            %       I1, I2, I3 - Principal moments of inertia of the body.
            %       a1, a2, a3 - 3x1 unit vectors defining the inertial frame.
            %                    (If not provided, defaults to standard basis.)
            %
            %   OUTPUT:
            %       F      - 3x1 approximate gravitational force vector in the inertial frame.
            %
            %   The force is computed as:
            %
            %       F = - (G*m_prime*m/R^2) * ( a1 + f2 )
            %
            %   where the second term is:
            %
            %       f2 = (3/R^2)*{ 0.5*[ I1*(1-3C(1,1)^2) + I2*(1-3C(1,2)^2) + I3*(1-3C(1,3)^2) ] * a1 +
            %                       [ I1*C(2,1)*C(1,1) + I2*C(2,2)*C(1,2) + I3*C(2,3)*C(1,3) ] * a2 +
            %                       [ I1*C(3,1)*C(1,1) + I2*C(3,2)*C(1,2) + I3*C(3,3)*C(1,3) ] * a3 }.


            % Set default inertial frame basis if not provided
            if nargin < 9 || isempty(a1)
                a1 = [1; 0; 0];
            end
            if nargin < 10 || isempty(a2)
                a2 = [0; 1; 0];
            end
            if nargin < 11 || isempty(a3)
                a3 = [0; 0; 1];
            end

            % Particle Term force term:
            F_particle = - (mu_body * m / R^2) * a1;

            % Compute terms for the f2 correction
            term1 = 0.5 * ( I1*(1 - 3*C(1,1)^2) + I2*(1 - 3*C(1,2)^2) + I3*(1 - 3*C(1,3)^2) );
            term2 = I1 * C(2,1)*C(1,1) + I2 * C(2,2)*C(1,2) + I3 * C(2,3)*C(1,3);
            term3 = I1 * C(3,1)*C(1,1) + I2 * C(3,2)*C(1,2) + I3 * C(3,3)*C(1,3);

            % f2 correction term (note: m cancels out)
            f2 = (3 / (m * R^2)) * ( term1 * a1 + term2 * a2 + term3 * a3 );

            % Total force
            F_approx = - (mu_body * m / R^2) * (a1 + f2);
        end

        % Gravity moment
        function [M_Bstar1, F_approx, F_mag, F_hat, R_cg_vec, R_cm_vec, R_cm_cg] = computeGravityMoment(obj, mu_body, M, R_cm_scalar, F_approx)
            % computeGravityMomentGeneric Computes the approximate gravity force and related
            % quantities for a body that is rotated with respect to an inertial frame.
            %
            %   [M_Bstar1, F_approx, F_mag, F_hat, R_cg_vec, R_cm_vec, R_cm_cg, ] = ...
            %        computeGravityMomentGeneric(mu_body, M, R_cm_scalar, F_approx)
            %
            %   INPUTS:
            %       mu_body       - Gravitational parameter of the attractor body
            %       M             - Mass of the body (system) (kg).
            %       R_cm_scalar   - Scalar distance from the reference point (e.g., Earth)
            %                       to the center of mass (CM) of the system.
            %       F_approx      - 3x1 approximate gravitational force vector (in inertial frame).
            %
            %   OUTPUTS:
            %       F_approx      - 3x1 gravitational force vector (already computed, scaled by M).
            %       F_mag         - Magnitude of F_approx.
            %       F_hat         - 3x1 unit vector in the direction of -F_approx (line of action).
            %       R_cg_vec      - 3x1 position vector (in inertial frame) from the reference to the
            %                       computed center of gravity (CG) along the force line of action.
            %       R_cm_vec      - 3x1 position vector (in inertial frame) for the center of mass (CM).
            %                       Assumed to lie along the inertial x-axis.
            %       R_cm_cg       - 3x1 vector from the CM to the CG.
            %       M_Bstar1      - 3x1 moment computed as: - cross(R_cm_vec, F_approx).

            % Ensure F_approx is a column vector
            F_approx = F_approx(:);

            % Compute force magnitude and unit vector (F_hat is along -F_approx)
            F_mag = norm(F_approx);
            F_hat = - F_approx / F_mag;

            % Compute equivalent CG distance along the line of action:
            R_cg = sqrt((mu_body * M) / F_mag);

            % Compute position vector from reference to CG (along F_hat)
            R_cg_vec = R_cg * F_hat;

            % Define the system's center of mass (CM) position vector
            % Assuming CM lies along the inertial x-axis (a1 = [1;0;0])
            R_cm_vec = R_cm_scalar * [1; 0; 0];

            % Compute vector from CM to CG
            R_cm_cg = R_cg_vec - R_cm_vec;

            % Compute moment (torque) using two different formulas:
            M_Bstar1 = - cross(R_cm_vec, F_approx);
        end

        %% .
        %% .
        %% .
        %% - - - - - - INERTIAS - - - - - -
        function [I_axial, I_trans, m] = InertiaSolidCylinder(obj, r, h, param, varargin)
            % computeCylinderInertia computes the inertia components for a solid cylinder.
            %
            % Usage:
            %   [I_axial, I_trans, m] = computeCylinderInertia(r, h, param)
            %     where 'param' is the density (rho). Mass is computed as:
            %         m = rho * pi * r^2 * h.
            %
            %   [I_axial, I_trans, m] = computeCylinderInertia(r, h, param, 'mass')
            %     where 'param' is the mass m directly.
            %
            % Inputs:
            %   r       - radius of the cylinder [m]
            %   h       - height (or thickness) of the cylinder [m]
            %   param   - either density (rho) or mass, depending on the flag
            %
            % Optional:
            %   'mass'  - flag indicating that param is the mass directly
            %
            % Outputs:
            %   I_axial - moment of inertia about the cylinder's symmetry axis (axial)
            %             computed as I_axial = 1/2 * m * r^2.
            %   I_trans - moment of inertia about a transverse axis
            %             computed as I_trans = 1/12 * m * (3*r^2 + h^2).
            %   m       - mass of the cylinder.

            if nargin > 3 && strcmpi(varargin{1}, 'mass')
                m = param; % The provided parameter is the mass.
            else
                % 'param' is the density.
                rho = param;
                m = rho * pi * r^2 * h;
            end

            I_axial = 0.5 * m * r^2;
            I_trans = (1/12) * m * (3*r^2 + h^2);
        end

        % Inertia Ellipsoid Plotter
        function figHandles = plotInertiaEllipsoid(I11, I22, I33, unitVec, varargin)
            % plotInertiaEllipsoid plots the 3D inertia ellipsoid and its 2D projections.
            %
            %   figHandles = plotInertiaEllipsoid(I11, I22, I33, unitVec) uses the principal moments
            %   I11, I22, I33 to compute the ellipsoid semi-diameters via:
            %
            %       alpha_i = k * Iii^(-1/2)
            %
            %   The user can specify the notation for the unit vector axes (e.g., 's', 'b', etc.).
            %   The function then creates a 3D plot and three 2D projections.
            %
            %   Optional pairs:
            %       'ScaleFactor' - scaling factor k (default is 1).
            %       'Resolution'  - number of points for generating the ellipsoid (default 1000).
            %
            %   Output:
            %       figHandles - a structure containing the figure handles:
            %                    .h3D  - 3D inertia ellipsoid plot.
            %                    .h12  - Projection on the (s1,s2) plane.
            %                    .h13  - Projection on the (s1,s3) plane.
            %                    .h23  - Projection on the (s2,s3) plane.
            %
            %   Created by Moacir Becker

            % Parse optional parameters.
            p = inputParser;
            addParameter(p, 'ScaleFactor', 1, @(x) isnumeric(x) && isscalar(x));
            addParameter(p, 'Resolution', 1000, @(x) isnumeric(x) && isscalar(x));
            parse(p, varargin{:});

            k = p.Results.ScaleFactor;
            N = p.Results.Resolution;

            % Compute semi-diameters of the ellipsoid.
            alpha1 = k * I11^(-1/2);
            alpha2 = k * I22^(-1/2);
            alpha3 = k * I33^(-1/2);

            fprintf('Semi-diameters of Inertia Ellipsoid:\n');
            fprintf(' alpha1 = %.3f\n alpha2 = %.3f\n alpha3 = %.3f\n\n', alpha1, alpha2, alpha3);

            % Create ellipsoid data.
            [X, Y, Z] = ellipsoid(0, 0, 0, alpha1, alpha2, alpha3, N);

            % Determine axis length for principal axes.
            Laxis = 1.5 * max([alpha1, alpha2, alpha3]);

            % Define LaTeX notation for unit vectors
            u1 = ['$\hat{' unitVec '}_1$'];
            u2 = ['$\hat{' unitVec '}_2$'];
            u3 = ['$\hat{' unitVec '}_3$'];

            %% 1) 3D Plot of the Inertia Ellipsoid
            h3D = figure('Name','3D Inertia Ellipsoid','NumberTitle','off');
            surf(X, Y, Z, 'FaceAlpha',0.2, 'EdgeColor','none', 'FaceColor','k');
            hold on; axis equal; grid on;
            xlabel(u1, 'Interpreter','latex','FontSize',14);
            ylabel(u2, 'Interpreter','latex','FontSize',14);
            zlabel(u3, 'Interpreter','latex','FontSize',14);
            title(['3D Inertia Ellipsoid (Moacir Becker)'],'Interpreter','latex','FontSize',14);

            % Plot principal axes with quiver3.
            quiver3(0,0,0, Laxis,0,0, 'r', 'LineWidth',2, 'MaxHeadSize',0.5);
            quiver3(0,0,0, 0,Laxis,0, 'k', 'LineWidth',2, 'MaxHeadSize',0.5);
            quiver3(0,0,0, 0,0,Laxis, 'b', 'LineWidth',2, 'MaxHeadSize',0.5);

            % Add text labels.
            text(Laxis,0,0, u1, 'Interpreter','latex','FontSize',14, 'Color','r');
            text(0,Laxis,0, [u2 ' (Symmetry)'], 'Interpreter','latex','FontSize',14, 'Color','k');
            text(0,0,Laxis, u3, 'Interpreter','latex','FontSize',14, 'Color','b');

            %% 2) 2D Projections of the Ellipsoid with Proper 2D Arrows
            % Projection on the (s1, s2) plane (ignoring Z).
            h12 = figure('Name','Inertia Ellipsoid: s1-s2 Projection','NumberTitle','off');
            hold on; axis equal; grid on;
            plot(X(:), Y(:), 'Color', [0.5 0.5 0.5], 'MarkerSize',5);
            xlabel(u1, 'Interpreter','latex','FontSize',14);
            ylabel(u2, 'Interpreter','latex','FontSize',14);
            title(['Projection on (' u1 ',' u2 ') Plane (Moacir Becker)'],'Interpreter','latex','FontSize',14);
            % Add principal axes using quiver (2D arrows)
            quiver(0, 0, Laxis, 0, 'r', 'LineWidth',2, 'MaxHeadSize',0.5);
            quiver(0, 0, 0, Laxis, 'k', 'LineWidth',2, 'MaxHeadSize',0.5);
            text(Laxis, 0, u1, 'Interpreter','latex','Color','r','FontSize',14);
            text(0, Laxis, [u2 ' (Symmetry)'], 'Interpreter','latex','Color','k','FontSize',14);

            % Projection on the (s1, s3) plane (ignoring Y).
            h13 = figure('Name','Inertia Ellipsoid: s1-s3 Projection','NumberTitle','off');
            hold on; axis equal; grid on;
            plot(X(:), Z(:), 'Color', [0.5 0.5 0.5], 'MarkerSize',4);
            xlabel(u1, 'Interpreter','latex','FontSize',14);
            ylabel(u3, 'Interpreter','latex','FontSize',14);
            title(['Projection on (' u1 ',' u3 ') Plane (Moacir Becker)'],'Interpreter','latex','FontSize',14);
            % Add principal axes using quiver (2D arrows)
            quiver(0, 0, Laxis, 0, 'r', 'LineWidth',2, 'MaxHeadSize',0.5);
            quiver(0, 0, 0, Laxis, 'b', 'LineWidth',2, 'MaxHeadSize',0.5);
            text(Laxis, 0, u1, 'Interpreter','latex','Color','r','FontSize',14);
            text(0, Laxis, u3, 'Interpreter','latex','Color','b','FontSize',14);

            % Projection on the (s2, s3) plane (ignoring X).
            h23 = figure('Name','Inertia Ellipsoid: s2-s3 Projection','NumberTitle','off');
            hold on; axis equal; grid on;
            plot(Y(:), Z(:), 'Color', [0.5 0.5 0.5], 'MarkerSize',4);
            xlabel(u2, 'Interpreter','latex','FontSize',14);
            ylabel(u3, 'Interpreter','latex','FontSize',14);
            title(['Projection on (' u2 ',' u3 ') Plane (Moacir Becker)'],'Interpreter','latex','FontSize',14);
            % Add principal axes using quiver (2D arrows)
            quiver(0, 0, Laxis, 0, 'k', 'LineWidth',2, 'MaxHeadSize',0.5);
            quiver(0, 0, 0, Laxis, 'b', 'LineWidth',2, 'MaxHeadSize',0.5);
            text(Laxis, 0, [u2 ' (Symmetry)'], 'Interpreter','latex','Color','k','FontSize',14);
            text(0, Laxis, u3, 'Interpreter','latex','Color','b','FontSize',14);

            % Collect figure handles.
            figHandles.h3D = h3D;
            figHandles.h12 = h12;
            figHandles.h13 = h13;
            figHandles.h23 = h23;
        end






        %% .
        %% .
        %% .
        %% - - - - - - HELPER PLOTS - - - - - -

        function plotParticles(obj, pos_vector, m_vector, names, colors)
            % plotParticles Plots a set of particles given their positions.
            %
            %   plotParticles(pos_vector, m_vector, names, colors)
            %
            %   INPUTS:
            %       pos_vector - An N×2 matrix of (x,y) positions.
            %       m_vector   - An N×1 vector of masses (used to scale marker sizes).
            %       names      - (Optional) A cell array of strings with particle names.
            %       colors     - (Optional) An N×1 cell array of color specifiers or an
            %                    N×3 matrix of RGB triplets.
            %
            %   This function creates a scatter plot of the particle positions. The marker
            %   size is proportional to the particle mass. If names and colors are provided,
            %   each particle is plotted with its corresponding color and labeled.
            %
            %   Example:
            %       pos_vector = [4, -2; 11, -10; 6, -5; 14, 5; 15, -8];
            %       m_vector = [2.5; 2; 3; 6; 4];
            %       names = {'S1', 'S2', 'S3', 'S4', 'P'};
            %       colors = {'r', 'g', [0.93 0.69 0.13], 'm', 'k'};
            %       plotParticles(pos_vector, m_vector, names, colors);

            figure;
            hold on; grid on; axis equal;
            xlabel('x'); ylabel('y');
            title('Particle Positions', 'Interpreter', 'latex');

            numParticles = size(pos_vector, 1);
            defaultSize = 100; % default marker size if m_vector is not provided

            for i = 1:numParticles
                % Determine marker size proportional to mass
                if ~isempty(m_vector)
                    markerSize = 100 * m_vector(i);
                else
                    markerSize = defaultSize;
                end

                % Determine color (default blue if not provided)
                if nargin < 4 || isempty(colors)
                    colorVal = 'b';
                else
                    if iscell(colors)
                        colorVal = colors{i};
                    else
                        colorVal = colors(i, :);
                    end
                end

                % Plot particle using scatter
                scatter(pos_vector(i,1), pos_vector(i,2), markerSize, colorVal, 'o','filled', 'LineWidth', 2);

                % Add label if provided
                if nargin >= 3 && ~isempty(names)
                    text(pos_vector(i,1) + 0.5, pos_vector(i,2) + 0.5, names{i}, 'FontSize', 12, 'Interpreter', 'latex');
                end
            end
        end



        function plotRotatingBox(obj, t_true, q_true_hist, ADC)
            % plotRotatingBox Visualizes the rotation of a 3D box given quaternion history.
            %
            % Inputs:d
            %   t_true         - Time array [nx1] corresponding to each quaternion.
            %   q_true_hist    - Quaternion history [nx4] array with quaternions at each time step.
            %   ADC            - AttitudeDeterminationLibrary object to access quaternion conversion methods.
            %
            % This function plots a 3D box representing a body-fixed coordinate system, with three
            % of its faces colored for easy visualization of rotation.

            % Set up figure for 3D box animation
            figure;
            hold on;
            axis equal;
            xlabel('$x$', 'Interpreter', 'latex', 'FontSize', 12);
            ylabel('$y$', 'Interpreter', 'latex', 'FontSize', 12);
            zlabel('$z$', 'Interpreter', 'latex', 'FontSize', 12);
            title('Rotation of Body-Fixed Coordinate System in Inertial Frame', 'Interpreter', 'latex', 'FontSize', 14);
            grid on;
            axis([-1.5 1.5 -1.5 1.5 -1.5 1.5]);
            view(3);

            % Define vertices for the cube (centered at origin)
            vertices = 0.5 * [-1 -1 -1; 1 -1 -1; 1 1 -1; -1 1 -1; -1 -1 1; 1 -1 1; 1 1 1; -1 1 1];
            faces = [1 2 3 4; 5 6 7 8; 1 2 6 5; 2 3 7 6; 3 4 8 7; 4 1 5 8];

            % Colors for specific faces (X, Y, Z)
            faceColors = [
                1 0 0;  % Red for one face (e.g., X-axis)
                0 1 0;  % Green for another face (e.g., Y-axis)
                0 0 1;  % Blue for another face (e.g., Z-axis)
                0.8 0.8 0.8; % Light gray for the rest
                0.8 0.8 0.8; % Light gray for the rest
                0.8 0.8 0.8; % Light gray for the rest
                ];

            % Create a patch object for the cube with different colors for each face
            h = patch('Vertices', vertices, 'Faces', faces, ...
                'FaceColor', 'flat', 'FaceVertexCData', faceColors, ...
                'EdgeColor', 'k', 'FaceAlpha', 0.5);

            % Animate the box with rotation history
            for i = 1:5:length(t_true) % Plot every 5th point to reduce computational load
                % Extract the quaternion at the current time step
                q = q_true_hist(i, :)';

                % Convert quaternion to DCM (inertial to body)
                A_BI = ADC.quaternion_to_dcm(q);

                % Transpose the DCM to convert from body to inertial frame
                A_IB = A_BI';

                % Apply the DCM to rotate the vertices of the box
                rotated_vertices = (A_IB * vertices')';

                % Update patch vertices
                set(h, 'Vertices', rotated_vertices);
                drawnow;

                % Pause to visualize each frame
                pause(0.05);
            end
        end

    end
end

