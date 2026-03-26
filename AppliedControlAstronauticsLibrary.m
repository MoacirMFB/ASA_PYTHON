classdef AppliedControlAstronauticsLibrary
    % Class for applied control astronautics functions and methods
    
    methods (Static)
        %% 1) LINEAR SYSTEMS
        
            % 1.1) Continuous Non-Linear System Dynamics of Augmented State Vector
            function dYdt = nonlinearDynamicsWithSTM(t, Y, dynamicsFunc, jacobianFunc, problemParams, szx)
                % This function computes the time derivative of both the state vector x*(t)
                % and the state transition matrix (STM) Phi(t, 0) for a generic nonlinear dynamic system.
                % Integrating this function delivers the STM and non-linear dynamics history 
                % 
                % Inputs:
                %   t            - Scalar, current time
                %   Y            - Vector, augmented state vector including both the state and the STM
                %   dynamicsFunc - Handle to a function that computes the nonlinear dynamics of the system
                %   jacobianFunc - Handle to a function that computes the Jacobian matrix of the system
                %   problemParams - Structure containing problem-specific parameters needed for the dynamics
                %   szx          - Size of the state vector
                %
                % Outputs:
                %   dYdt         - Time derivative of the augmented state vector [xdot; vec(Phi_dot)]
            
                % Extract the state vector x*(t) from the augmented vector Y
                x_star = Y(1:szx);
            
                % Extract and reshape Phi(t, 0) from the augmented vector Y
                phi_vec = Y(szx + 1:end);
                Phi = reshape(phi_vec, szx, szx);
            
                % Compute the nonlinear dynamics (xdot) using the provided dynamics function
                xdot = dynamicsFunc(t, x_star, problemParams);
            
                % Compute the Jacobian matrix F(x*(t)) at the current state x*(t)
                J = jacobianFunc(t, x_star, problemParams);
            
                % Compute the derivative of the STM: Phi_dot = F(x*(t)) * Phi
                Phi_dot = J * Phi;
            
                % Flatten the matrix Phi_dot into a vector for integration
                Phi_dot_vec = Phi_dot(:);
            
                % Concatenate xdot and Phi_dot_vec to form the augmented derivative dYdt
                dYdt = [xdot; Phi_dot_vec];
            end
            
            % 1.2) Continuous Linear System Dynamics of Augmented State Vector
            function dYdt = discreteLinearDynamics(t, Y, A, B, c, uk, szx, szu)
             % augStateDiscreteLinearDynamics - Computes the time derivative of the
             % augmented state vector for a linear dynamic systems from tk to tk+1.
             % This function is used in the discretization process of the linear
             % dynamics also. 
               %
               % Inputs:
               %   t  - (not used) Scalar, current time step
               %   Y  - Vector, current augmented state vector that includes the state, and the flattened state transition matrix (STM), control mapping matrix,
               %        and the control-independent term
               %   A  - Matrix, system dynamics matrix (∂f/∂x) assumed constant
               %   B  - Matrix, control input matrix (∂f/∂u) assumed constant 
               %   c  - Vector, constant control-independent term (f(x,uk,t) - Ax - Buk)
               %   uk - Vector, current control input
               %   szx - Scalar, size of the state vector
               %
               % Outputs:
               %   dYdt - Vector, the time derivative of the augmented state vector:[dxdt; dAdt; dBdt ; cdot]
               %
               % Assumptions:
               %   The STM, control mapping matrix (B), and control-independent term (c) are assumed to be constant and come from the continuous-time linear system. 
               %   The function computes the time derivative  of these matrices and the state vector given the control input uk.     
            
               
               % Check if control input 'uk' was provided or is empty; if not, default to zero vector
               if isempty(uk)
                   uk = zeros(szu, 1); % Assuming control vector 'uk' should have a specific size
               end
               
               % Extract the state vector 'x' from the augmented state vector 'Y'
               x = Y(1:szx);
               
               % Extract and reshape 'phi' from 'Y'
               phi_vec = Y(szx + 1: szx + szx^2);
               Phi = reshape(phi_vec, szx, szx);
               
               % Calculate the rate of change for A, B, and c based on the algorithm
               Adot = A * Phi;
               Bdot = inv(Phi) * B; % Note: consider using 'pinv' to avoid potential inversion issues
               cdot = inv(Phi) * c;
               
               % Calculate the state dynamics at the current point 'x'
               xdot = A * x + B * uk + c;
               
               % Construct the time derivative of the augmented state vector
               dYdt = [xdot; Adot(:); Bdot(:); cdot(:)];
            end
           
            % 1.3) Discrete Time Linear System Discrete Matrices
            function [Ak, Bk, ck, xk_1] = linearDiscreteTimeMatrices(tk, tk_1, Yk, A, B, c, uk, szx, szu, optionsODE)    
              % linearDiscreteTimeMatrices - Integrates the equations of motion (EOM) over a specified time interval from tk to tk+1 using the augStateDiscreteLinearDynamics function.   
                % Inputs:
                %   tk - Scalar, initial time step
                %   tk_1 - Scalar, final time step
                %   Yk - Vector, initial augmented state vector at time tk Yk = [x0,vec(Inx),0_nxnu,0nx]
                %   A - Matrix, system dynamics matrix (∂f/∂x) which is constant for linear systems
                %   B - Matrix, control input matrix (∂f/∂u) which is constant for linear systems
                %   c - Vector, constant control-independent term (f(x,uk,t) - Ax - Buk)
                %   uk - Vector, control input, assumed constant over the integration
                %   interval due to Zero Order Hold Control 
                %   szx - Scalar, size of the state vector
                %   szu - Scalar, size of the control vector
                %   optionsODE - struct, options for the MATLAB ODE solver such as tolerances
                %
                % Outputs:
                %   Ak - Matrix, state transition matrix at final time step tk_1
                %   Bk - Matrix, control input matrix at final time step tk_1
                %   ck - Vector, control-independent term at final time step tk_1
                %   xk_1 - Vector, state vector at final time step tk_1
                %
                % Assumptions:
                %   The function assumes that the dynamics of the system are represented 
                %   by a linear discrete-time model and the matrices A, B, and c are constant.
                %   The control input uk is assumed to be constant over the interval from tk to tk+1.
            
            
                [t, Y] = ode45(@(t,Y) AppliedControlAstronauticsLibrary.discreteLinearDynamics(t, Y, A, B, c, uk, szx), [tk, tk_1], Yk, optionsODE);
              
                Yk_1 = Y(end,:)';                                                     % Y evaluated at t_(k+1) is last element
                xk_1 = Yk_1(1:szx);                                                   % Extract state vector at t_(k+1)
                Ak_vec = Yk_1(szx + 1 : szx + szx^2);                                 % Extract Ak vector from Y_(k+1)
                Ak = reshape(Ak_vec,szx,szx);                                         % Turn Ak into matrix
                Bk_vec = Yk_1(szx + szx^2 + 1 : szx + szx^2 + szx * szu);             % Extract Bk vector from Y_(k+1)
                Bk = Ak * reshape(Bk_vec,szx,szu);                                    % Turn Bk into matrix
                ck = Ak  * Yk_1(szx + szx^2 + szx * szu + 1 : szx + szx^2 + szx * szu + szx); % Extract ck from Y_(k+1)
                
            end 

        %% 2) 2 BODY PROBLEM        
            %% 2.1) Optimal Planar Planetary Transfer Functions  
                %% - - Minimum Energy - Unconstrained Control - Free vf       
                    function Psi = minEnergyShootingFunctionFreeVf(lambda0, t0, tf, npoints, x0, rf, mu, optionsODE)   
                
                        %   rf is the desired final positoion
                        
                        tspan = linspace(t0, tf, npoints);
                    
                        Psi = zeros(4,1);    
                        X0 = [x0; lambda0]; % Initial augmented state vector [state; costate]
                    
                        % Use the separated combinedDynamics function
                        [T, X] = ode45(@(t, X) AppliedControlAstronauticsLibrary.minEnergyCombinedDynamics(t, X, mu), tspan, X0, optionsODE);
                    
                        lambda3_tf = X(end,7);
                        lambda4_tf = X(end,8);
                    
                        final_position = X(end, 1:2) ;
                    
                        Psi(1:2) = final_position' - rf ;           
                        Psi(3) = lambda3_tf ;    
                        Psi(4) = lambda4_tf ;    
                    end
                
                %% - - Minimum Energy - Unconstrained Control - Fixed xf
                    
                    % [ d𝛌/dt ]   Costate Dynamics: Min Energy -- Unconstrained Control  
                    function dlambda_dt = minEnergyCostateDynamics(t, lambda, x)
                        % Costate dynamics for minimum energy transfer
                        % lambda: costate vector as a column vector 
                        % x: state vector [position; velocity] as a column vector
                    
                        r = x(1:2);             % Position vector  
                        r_mag = norm(r);
                        I2 = eye(2);
                        O2 = zeros(2);  
                        C22 = -1 * (I2/(r_mag^3) - 3*(r*r')/(r_mag^5));
                    
                        C = [O2, I2; C22, O2];
                        dlambda_dt = -(C' * lambda); % Ensure lambda is a column vector
                    end
                    
                    % [ u* ]        Optimal Control: Min Energy -- Unconstrained Control 
                    function u_star = minEnergyOptimalControl(lambda)
                        % Optimal control for minimum energy transfer
                        % lambda: can be a single column vector or a matrix where each row represents a timestep
                    
                        I2 = eye(2);
                        O2 = zeros(2,2);
                        B = [O2; I2]; 
                    
                        % For a single column vector, directly compute the optimal control
                        if isvector(lambda)
                            u_star = -0.5 * B' * lambda; 
                        else
                            % If lambda is a matrix,
                            [numRows, ~] = size(lambda);
                            u_star = zeros(numRows, 2); 
                            for i = 1:numRows
                                lambda_i = lambda(i, :)';
                                u_star(i, :) = (-0.5 * B' * lambda_i)'; % Compute u_star for each timestep
                            end
                        end
                    end
                    
                    % [ dxdt ]    State Dynamics: Min Energy -- Unconstrained Control 
                    function dxdt = minEnergyStateDynamics(t, x, u, mu)
                        % State dynamics for minimum energy transfer with control as perturbation
                        % x: state vector [position; velocity] as a column vector
                        % u: control input [u1; u2] as a column vector
                        % mu: gravitational parameter
                    
                        r = x(1:2); % Position vector as a column vector
                        v = x(3:4); % Velocity vector as a column vector
                        r_mag = norm(r); % Cubed norm of the position vector
                    
                        I2 = eye(2);
                        O2 = zeros(2,2);
                        B = [O2;I2];
                    
                        f0 = [v; - (mu/r_mag^3) * r];   %Unperturbed 2B dynamics     
                    
                        dxdt = f0 + B * u;              % Calculate control-perturbed dynamics
                    end
                    
                    % [ Psi ]      Shooting Function: Min Energy -- Unconstrained Control 
                    function Psi = minEnergyShootingFunction(lambda0, t0, tf, npoints, x0, xf, mu, optionsODE)
                        tspan = linspace(t0, tf, npoints);
                    
                        % Initial augmented state vector [state; costate]
                        X0 = [x0; lambda0];
                    
                        % Use the separated combinedDynamics function
                        [T, X] = ode45(@(t, X) AppliedControlAstronauticsLibrary.minEnergyCombinedDynamics(t, X, mu), tspan, X0, optionsODE);
                    
                        final_state = X(end, 1:4);
                        Psi = final_state' - xf;
                    end
                
                    % [ dXdt ]    Combined Dynamics: Min Energy -- Unconstrained Control 
                    function dXdt = minEnergyCombinedDynamics(t, X, mu)
                        % Extract state and costate from the augmented vector
                        x = X(1:4);         % State vector is 4-dimensional
                        lambda = X(5:end);  % Costate vector starts at the 5th element
                    
                        % Calculate optimal control
                        u_star = AppliedControlAstronauticsLibrary.minEnergyOptimalControl(lambda);
                    
                        % Calculate dynamics of state and costate
                        dxdt = AppliedControlAstronauticsLibrary.minEnergyStateDynamics(t, x, u_star, mu);
                        dlambda_dt = AppliedControlAstronauticsLibrary.minEnergyCostateDynamics(t, lambda, x);
                    
                        % Combine derivatives into a single vector
                        dXdt = [dxdt; dlambda_dt];
                    end
                    
                    % [ H ]         Hamiltonian: Min Energy -- Unconstrained Control 
                    function H = minEnergyHamiltonian(u_hist, lambda_hist, state_hist, mu)
                        % Calculate ||u||^2 for each row directly
                        u_norm_sq = sum(u_hist.^2, 2);
                    
                        % Position (r) and velocity (v) histories
                        r_hist = state_hist(:, 1:2);
                        v_hist = state_hist(:, 3:4);
                    
                        % Calculate the squared magnitude of r for each row and its cube root for gravitational force computation
                        r_mag = sqrt(sum(r_hist.^2, 2));
                        r_mag_cubed = r_mag.^3;
                    
                        % Unperturbed dynamics f0: [v; -mu*r/r^3]
                        f0 = [v_hist, (-mu ./ r_mag_cubed) .* r_hist];
                    
                        % Perturbed dynamics f: considering B*u where B = [0; I], only affects the velocity part
                        f = f0 + [zeros(size(u_hist)), u_hist];
                    
                        % Calculate lambda*f for each row, assuming lambda_hist and f are of the same size
                        lambda_dot_f = sum(lambda_hist .* f, 2);
                    
                        % Sum the two components to get H for each row/time step
                        H = u_norm_sq + lambda_dot_f;
                    end

                %% - - Minimum Time - Constrained Control
            
                    % [ d𝛌/dt ]  Costate Dynamics: Min Time -- Bounded Control 
                    function dlambda_dt = minTimeCostateDynamics(t, lambda, x) % #CHKD
                        % Costate dynamics for minimum time transfer
                        % lambda: costate vector as a column vector
                        % x: state vector [position; velocity] as a column vector
                        
                        r = x(1:2);             % Position vector  
                        r_mag = norm(r);
                        I2 = eye(2);
                        O2 = zeros(2);  
                        C22 = -1 * (I2/(r_mag^3) - 3*(r*r')/(r_mag^5));
                        
                        C = [O2, I2; C22, O2];
                        dlambda_dt = -(C' * lambda); % Ensure lambda is a column vector
                    end
                    
                    % [ u* ]       Optimal Control: Min Time  -- Bounded Control 
                    function u_star = minTimeConstrainedOptimalControl(lambda, umax) % #CHKD
                        % Optimal constrained control for minimum fuel optimization using tanh as a smoother approximation    
                        % umax is the max control input possible 
                        % lambda is the costate column vector or matrix with lambda history 
                        B = [zeros(2,2); eye(2)]; 
                    
                        % Calculate p vector 
                        if isvector(lambda)
                            p = -B' * lambda;                                           % Primer vector 
                            p_norm = norm(p, 2);                                        % Primer vector magnitude
                            phat = p / p_norm;                                          % Unit vector of p        
                            u_star = umax * phat;
                        else
                            % If lambda is a matrix of lambda historic values, where each row represents a timestep 
                            [numRows, ~] = size(lambda);
                            u_star = zeros(numRows, 2);
                            for i = 1:numRows
                                lambda_i = lambda(i, :)';
                                p_i = - B' * lambda_i;            
                                phat_i = p_i / norm(p_i, 2);                        
                                u_star(i, :) = (umax * phat_i)';
                            end
                        end
                    end 
                
                    % [ dxdt ]    State Dynamics: Min Time -- Bounded Control   
                    function dxdt = minTimeStateDynamics(t, x, u, mu) % #CHKD
                        % State dynamics for minimum energy transfer with control as perturbation
                        % x: state vector [position; velocity] as a column vector
                        % u: control input [u1; u2] as a column vector
                        % mu: gravitational parameter
                        
                        r = x(1:2); % Position vector as a column vector
                        v = x(3:4); % Velocity vector as a column vector
                        r_mag = norm(r); % norm of the position vector
                        
                        I2 = eye(2);
                        O2 = zeros(2,2);
                        B = [O2;I2];
                    
                        f0 = [v; - (mu/r_mag^3) * r];   % Unperturbed 2B dynamics     
                        
                        dxdt = f0 + B * u;              % Calculate control-perturbed dynamics
                    end
                    
                    % [ Psi ]      Shooting Function: Min Time -- Bounded Control 
                    function Psi = minTimeShootingFunction(Z, t0, npoints, x0, xtar_t0, a_tar, nu_tar_t0, mu, umax, optionsODE)
                    
                        % Lambda0 : initial value for costate vector guessed or not
                        % tf : final simulation time, guessed or not 
                        % t0 : initial time is known 
                        % x0 : initial spacecraft state is known 
                        % umax : max control input magnitude  
                        % xtar : desired final state, xf_target
                        % mu : gravitational parameter 
                        % a : semimajor axis of target body's orbit
                        % Z : contains both lambda0 and tf
                        % nu_tar_t0 : initial true anomaly of target body 
                            
                        lambda0 = Z(1:4);                                             % Extract initial costate vector from Z 
                        tf = Z(end);                                                  % Extract final time from Z vector
                    
                        Psi = zeros(5,1);                                             % Psi function contains 2 BC in this case 
                        
                        tspan = linspace(t0, tf, npoints);                            % Timespan for the simulation       
                        X0 = [x0; lambda0];                                           % Initial augmented state vector [state; costate]        
                        
                        % Propagate augmented state dynamics    
                        [T, X] = ode45(@(t, X) AppliedControlAstronauticsLibrary.minTimeCombinedDynamics(t, X, mu, umax), tspan, X0, optionsODE); 
                        lambda_tf = X(end, 5:8)';                                     % Get lambda vector at final propagation time
                        xf = X(end, 1:4)';                                            % Get state vector at final propagation time
                        
                        ustar_tf = AppliedControlAstronauticsLibrary.minTimeConstrainedOptimalControl(lambda_tf, umax); % Calculate optimal control for final lambda value obtained
                        H_tf = AppliedControlAstronauticsLibrary.minTimeHamiltonian(ustar_tf, lambda_tf, xf, mu);       % Calculate Hamiltonian at final propagation time 
                    
                        % Propagate dynamics of the target point with given initial conditions 
                        [Ttar, Xtar] = ode45 (@(t, Xtar) AppliedControlAstronauticsLibrary.dynamicsCircularPlanarOrbit(t, a_tar, mu, nu_tar_t0), tspan, xtar_t0, optionsODE);
                        xtar_tf = Xtar(end,:)';                                        % Get the target's final state vector
                    
                        % Calculate dxdt of the target at time tf
                        dXtardt_tf = AppliedControlAstronauticsLibrary.dynamicsCircularPlanarOrbit(tf, a_tar, mu,nu_tar_t0);     
                    
                        % Calculate shooting function 
                        Psi(1:4) = xf - xtar_tf;                      % Difference between target's final state and spacecraft's
                        Psi(5) = H_tf - lambda_tf' * dXtardt_tf;
                    
                    end
                    
                    % [ dXdt ]    Combined Dynamics: Min Time -- Bounded Control  
                    function dXdt = minTimeCombinedDynamics(t, X, mu, umax)
                        % Extract state and costate from the augmented vector
                        x = X(1:4);         % State vector is 4-dimensional
                        lambda = X(5:end);  % Costate vector starts at the 5th element
                    
                        % Calculate optimal control 
                        u_star = AppliedControlAstronauticsLibrary.minTimeConstrainedOptimalControl(lambda, umax);
                    
                        % Calculate dynamics of state and costate
                        dxdt = AppliedControlAstronauticsLibrary.minTimeStateDynamics(t, x, u_star, mu);
                        dlambda_dt = AppliedControlAstronauticsLibrary.minTimeCostateDynamics(t, lambda, x);
                    
                        % Combine derivatives into a single vector
                        dXdt = [dxdt; dlambda_dt];
                    end
                
                    % [ H ]         Hamiltonian Function: Min Time -- Bounded Control  
                    function H = minTimeHamiltonian(u, lambda, state, mu)
                        
                        I2 = eye(2);
                        O2 = zeros(2,2);
                        B = [O2; I2]; 
                    
                        if isvector(u)
                            u_two_norm = norm(u);                    % Calculate ||u|| for each row directly    
                            
                            r = state(1:2);                          % Position (r) and velocity (v) histories
                            v = state(3:4);           
                            r_mag = norm(r);             
                            r_mag_cubed = r_mag^3;
                        
                            f0 = [v; (-mu / r_mag_cubed) * r];                % Unperturbed dynamics f0: [v; -mu*r/r^3]
                            f = f0 + B * u;                                   % Perturbed dynamics f: considering B*u where B = [0; I]
                            lambda_dot_f = dot(lambda,f);                     % Calculate lambda dot f 
                    
                            H = u_two_norm + lambda_dot_f;                    % Sum the two components to get H for each row/time step
                    
                        else 
                            u_two_norm = sqrt(sum(u.^2, 2));                   % Calculate ||u|| for each row directly    
                            
                            r_hist = state(:, 1:2);                            % Position (r) and velocity (v) histories
                            v_hist = state(:, 3:4); 
                                
                            r_mag = sqrt(sum(r_hist.^2, 2));                   % Squared magnitude of r for each row 
                            r_mag_cubed = r_mag.^3;
                        
                            f0 = [v_hist, (-mu ./ r_mag_cubed) .* r_hist];     % Unperturbed dynamics f0: [v; -mu*r/r^3]
                            f = f0 + [zeros(size(u)), u];                      % Perturbed dynamics f: considering B*u where B = [0; I]
                            lambda_dot_f = sum(lambda .* f, 2);                % Calculate lambda*f for each row
                            H = u_two_norm + lambda_dot_f;                     % Sum the two components to get H for each row/time step
                        end 
                    end
            
                    
                %% - - Minimum Fuel - Constrained Control
                        
                % [ d𝛌/dt ]    Costate Dynamics: Min Fuel -- Bounded Control 
                function dlambda_dt = minFuelCostateDynamics(t, lambda, x)
                    % Costate dynamics for minimum fuel transfer
                    % lambda: costate vector as a column vector
                    % x: state vector [position; velocity] as a column vector
                
                    r = x(1:2);             % Position vector  
                    r_mag = norm(r);
                    I2 = eye(2);
                    O2 = zeros(2);  
                    C22 = -1 * (I2/(r_mag^3) - 3*(r*r')/(r_mag^5));
                
                    C = [O2, I2; C22, O2];
                    dlambda_dt = -(C' * lambda); % Ensure lambda is a column vector
                end
            
                % [ u* ]        Optimal Control: Min Fuel -- Bounded Control 
                function u_star = minFuelConstrainedOptimalControl(lambda, umax, rho)
                    % Optimal constrained control for minimum fuel optimization using tanh as a smoother approximation
                    % rho is a smoothing parameter 
                    % umax is the max control input possible 
                    % lambda is the costate column vector or matrix with lambda history 
                
                    I2 = eye(2);
                    O2 = zeros(2,2);
                    B = [O2; I2]; 
                
                    % Calculate p vector 
                    if isvector(lambda)
                        p = -B' * lambda;                                           % Primer vector 
                        p_norm = norm(p, 2);                                        % Primer vector magnitude
                        phat = p / p_norm;                                          % Unit vector of p
                        gamma = (umax / 2) * (1 + tanh((p_norm - 1) / rho));        % Magnitude of control
                        u_star = gamma * phat;
                    else
                        % If lambda is a matrix of lambda historic values, where each row represents a timestep 
                        [numRows, ~] = size(lambda);
                        u_star = zeros(numRows, 2);
                        for i = 1:numRows
                            lambda_i = lambda(i, :)';
                            p_i = - B' * lambda_i;
                            p_norm_i = norm(p_i, 2);
                            phat_i = p_i / p_norm_i;            
                            gamma_i = (umax / 2) * (1 + tanh((p_norm_i - 1) / rho)); % Magnitude of control
                            u_star(i, :) = (gamma_i * phat_i)';
                        end
                    end
                end
                
                % [ dxdt ]     State Dynamics: Min Fuel -- Bounded Control  
                function dxdt = minFuelStateDynamics(t, x, u, mu)
                    % State dynamics for minimum energy transfer with control as perturbation
                    % x: state vector [position; velocity] as a column vector
                    % u: control input [u1; u2] as a column vector
                    % mu: gravitational parameter
                
                    r = x(1:2); % Position vector as a column vector
                    v = x(3:4); % Velocity vector as a column vector
                    r_mag = norm(r); % Cubed norm of the position vector
                
                    I2 = eye(2);
                    O2 = zeros(2,2);
                    B = [O2;I2];
                
                    f0 = [v; - (mu/r_mag^3) * r];   %Unperturbed 2B dynamics     
                
                    dxdt = f0 + B * u;              % Calculate control-perturbed dynamics
                end
                
                % [ Psi ]      Shooting Function: Min Fuel -- Bounded Control 
                function Psi = minFuelShootingFunction(lambda0, t0, tf, npoints, x0, xf, mu, umax, rho, optionsODE)
                    tspan = linspace(t0, tf, npoints);
                
                    % Initial augmented state vector [state; costate]
                    X0 = [x0; lambda0];
                
                    % Use the separated combinedDynamics function
                    [T, X] = ode45(@(t, X) AppliedControlAstronauticsLibrary.minFuelCombinedDynamics(t, X, mu, umax, rho), tspan, X0, optionsODE);
                
                    final_state = X(end, 1:4);
                    Psi = final_state' - xf;
                end
            
                % [ dXdt ]    Combined Dynamics: Min Fuel -- Bounded Control  
                function dXdt = minFuelCombinedDynamics(t, X, mu, umax, rho)
                    % Extract state and costate from the augmented vector
                    x = X(1:4);         % State vector is 4-dimensional
                    lambda = X(5:end);  % Costate vector starts at the 5th element
                
                    % Calculate optimal control 
                    u_star = AppliedControlAstronauticsLibrary.minFuelConstrainedOptimalControl(lambda, umax, rho);
                
                    % Calculate dynamics of state and costate
                    dxdt = AppliedControlAstronauticsLibrary.minFuelStateDynamics(t, x, u_star, mu);
                    dlambda_dt = AppliedControlAstronauticsLibrary.minFuelCostateDynamics(t, lambda, x);
                
                    % Combine derivatives into a single vector
                    dXdt = [dxdt; dlambda_dt];
                end
            
                % [ H ]         Hamiltonian Function: Min Fuel -- Bounded Control  
                function H = minFuelHamiltonian(u_hist, lambda_hist, state_hist, mu)
                    % Calculate ||u|| for each row directly
                    u_two_norm = sqrt(sum(u_hist.^2, 2));
                
                    % Position (r) and velocity (v) histories
                    r_hist = state_hist(:, 1:2);
                    v_hist = state_hist(:, 3:4);
                
                    % Calculate the squared magnitude of r for each row and its cube root for gravitational force computation
                    r_mag = sqrt(sum(r_hist.^2, 2));
                    r_mag_cubed = r_mag.^3;
                
                    f0 = [v_hist, (-mu ./ r_mag_cubed) .* r_hist];     % Unperturbed dynamics f0: [v; -mu*r/r^3]
                    f = f0 + [zeros(size(u_hist)), u_hist];            % Perturbed dynamics f: considering B*u where B = [0; I]
                    lambda_dot_f = sum(lambda_hist .* f, 2);           % Calculate lambda*f for each row
                
                    H = u_two_norm + lambda_dot_f;                      % Sum the two components to get H for each row/time step
                end


            %% 2.2) Perturbing Accelerations (2BP)
        
                % [aj2_vec_eci] J2 Perturbation -- ECI frame
                function [aj2_vec_eci] = calc_J2_accel_cartesian(r_vec_eci) 
                    % Calculates the J2 perturbation acceleration components in cartesian
                    % coordinates given the position vector 
                    % Constants
                    mu = 398600.0;  % Earth's gravitational parameter (mu) in km^3/s^2
                    J2 = 0.0010826; % J2 coefficient    
                    ro = 6378.1;    % Earth's equatorial radius [km]
                
                    % Extract position components from the state vector
                    x = r_vec_eci(1);       % Pos X [km]
                    y = r_vec_eci(2);       % Pos Y [km]
                    z = r_vec_eci(3);       % Pos Z [km]
                
                    % Calculate the distance from the center of the Earth
                    r = sqrt(x^2 + y^2 + z^2);
                    
                    % Calculate the J2 perturbation acceleration components
                    z2_r2 = (z^2) / (r^2);
                    factor = -1.5 * J2 * mu * (ro^2) / (r^5);
                
                    aj2_x = factor * x * (1 - 5 * z2_r2);
                    aj2_y = factor * y * (1 - 5 * z2_r2);
                    aj2_z = factor * z * (3 - 5 * z2_r2);
                
                    % Assemble the derivative of the state vector including the J2 perturbation
                    aj2_vec_eci = [aj2_x ; aj2_y ; aj2_z];
                end
            
                % [aSRP_vec_eci] Solar Radiation Pressure Perturbation -- ECI frame
                function [aSRP_vec_eci] = calc_SRP_accel_cartesian(t, r_vec_eci)
                    
                    % Parameters and Constants to be used
                    A_m_ratio = 5.4e-6;      % Area to mass ratio for S/C km2/kg= 5.4e-6;  
                    a = 149597898;           % Average Earth-Sun distance in km (1 astronomical unit)
                    mu = 132712440017.99;    % Gravitational constant of the Sun in km^3/s^2
                    n = sqrt(mu / a^3);      % Mean motion of Earth around Sun in rad/s
                    r_earth_sun = a;         % Assuming circular orbit for Earth around Sun
                    G0 = 1.02e14;            % Solar flux constant kg*km/s2
                
                    % Time-dependent angles for Earth's position in its orbit
                    theta = n * t; % Angle in radian
                    
                    % Unit vectors for the ECI coordinate system
                    x_hat = [1; 0; 0];
                    y_hat = [0; 1; 0];
                
                    % Compute the position of the spacecraft relative to the Sun
                    aSRP_num = A_m_ratio * G0 * (r_vec_eci + r_earth_sun*(cos(theta)*x_hat - sin(theta)*y_hat));
                    
                    aSRP_den = norm(r_vec_eci + r_earth_sun * (cos(theta) * x_hat - sin(theta) * y_hat))^3;
                
                    aSRP_vec_eci = aSRP_num/aSRP_den;        
                end

            %% 2.3) Planar Circular Orbits EOMs

                % [ x ] State Vector: Planar Circular Orbit
                function [x] = stateCircularPlanarOrbit(a, mu, nu)    
                % Calculates the state vector in x-y coordinates in a circular planar orbit
                    % nu : true anomaly angle as measured from x+ counterclockwise
                    % a : semimajor axis value for orbit radius 
                
                    vcirc = sqrt(mu/a);                      % Target's speed around circular orbit
                
                    rx = cos(nu)*a;
                    ry = sin(nu)*a;
                    pos = [rx;ry];
                
                    vx = - sin(nu) * vcirc;
                    vy =   cos(nu) * vcirc;
                    vel = [vx;vy];
                    
                    x = [pos;vel];
                end 
                
                % [ dxdt ] State Dynamics: Planar Circular Orbit
                function [dxdt] = dynamicsCircularPlanarOrbit(t, a, mu, nu0)    % TBCKD
                % Calculates the state dynamics in a circular planar orbit dxdt
                    % mu : gravitational parameter 
                    % a : semi-major axis
                    % nu0 : initial true anomaly 
                    
                    n = sqrt(mu/a^3);                        % Target mean motion 
                    vcirc = sqrt(mu/a);                      % Target's speed around circular orbit
                
                    vx =  - a * sin(nu0 + n * t);
                    vy =    a * cos(nu0 + n * t);
                    vel = [vx; vy];
                    
                    ax = - vcirc * cos(nu0 + n * t);
                    ay = - vcirc * sin(nu0 + n * t);
                    accel = [ax; ay];
                
                    dxdt = n * [vel; accel];
                end 

            %% 2.4) 2BP Dynamics + J2 Perturbation - STM
        
                % - - [ dXdt ] Combined Dynamics State + STM - 2BP + J2
                function [X_dot] = augmentedStateDynamics2BPJ2(t, X, mu, rE, J2)
                % X: augmented state vector that contains both state and flattened STM 
                    % mu = 398600;    % Earth's gravitational parameter
                    % rE = 6378.1;    % Earth's radius [km]
                    % J2 = 0.0010826; % J2 coefficient 
                
                % ---- f0 propagation using [v, -mu*r_vec/r3] ----
                    % Extract position components from the state vector
                        r_vec = X(1:3);        % [km]
                        x = r_vec(1);          % [km]
                        y = r_vec(2);          % [km]
                        z = r_vec(3);          % [km]
                        r = norm(r_vec);       % [km]
                
                    % Extract velocity vector from the state vector
                        v_vec_eci = X(4:6);        % [km/s]
                
                    % Calculate the gravitational acceleration components 2BP nonlinear based
                    % f0 rate of change of state vector 
                        x_dot = [v_vec_eci ; -mu * x / r^3; -mu * y / r^3 ; -mu * z / r^3];     
                
                % ---- STM DOT CALCULATION ONLY ----
                    % Split the state vector and STM 
                        x_vec = X(1:6);                   % State vector [x y z vx vy vz]
                        Phi_vec = X(7:end);               % Get the Phi matrix embedded in the aug state vector
                        Phi_mtx = reshape(Phi_vec, 6, 6); % Reshape Phi to a matrix
                        
                    % Compute the A matrix (STM matrix) at current state value
                        A = AppliedControlAstronauticsLibrary.A_2BPJ2(x_vec, mu, rE, J2);
                        
                    % Compute the STM time derivative based on A calculated before dPhi/dt =  A * Phi
                        dPhi_dt_mtx = A * Phi_mtx;
                        dPhi_dt_vec = reshape(dPhi_dt_mtx, 6^2, 1);
                        
                        X_dot = [x_dot; dPhi_dt_vec];    % Return the augmented state vector derivative
                end 
                    
                % - -  [ A ] Matrix A -- 2BP + J2
                function A = A_2BPJ2(X,mu,rE,J2) 
                    % mu = 398600;    % Earth's gravitational parameter
                    % rE = 6378.1;    % Earth's radius [km]
                    % J2 = 0.0010826; % J2 coefficient       
                
                    % - Retrieve state, STM values and define other values - 
                    x_vec = X(1:6);                 % Extract state vector 
                    r_vec = x_vec(1:3);             % Extract position vector                      
                    v_vec = x_vec(4:6);             % Extract velocity vector     
                    r_mag = norm(r_vec);            % Find magnitude of position vector 
                    I3 = eye(3);                    % 3x3 identity matrix
                    O3 = zeros(3,3);                % 3x3 zero matrix 
                    B = [O3;I3];                    % B matrix to be used in f(x,t) = xdot
                    x = r_vec(1);                   % Extract x component
                    y = r_vec(2);                   % Extract y component
                    z = r_vec(3);                   % Extract z component    
                    n1_hat = [1; 0; 0];             % Define n1 hat vector 
                    n2_hat = [0; 1; 0];             % Define n2 hat vector 
                    n3_hat = [0; 0; 1];             % Define n3 hat vector 
                    
                % - - Compute A Matrix of the System - -
                % Gradient/Jacobian of the 2B dynamics wrt to state vector - No perturbation 𝜹f0/𝜹x
                % Compute the term that is common in the Jacobian matrix
                % r_vec is a column vector as expected at this point 
                    jacobianTerm =  - mu * ((I3/r_mag^3) - ((3/r_mag^5)*(r_vec*r_vec')));
                    
                    % Assemble the Jacobian matrix
                    df0_dx = [O3, I3;
                            jacobianTerm, O3];   
                        
                    % Calculate the common scalar factor for J2 perturbation
                    factorJ2 = -(3 * mu * J2 * rE^2) / (2 * r_mag^5);       
                
                    % Calculate each d_aj2_dx
                    daj2x_dx = factorJ2 * (((5*x / r_mag^2) * ((7*z^2 /r_mag^2) - 1)*r_vec') - ((10*x*z) /r_mag^2) * n3_hat' + (1 - (5 * z^2 / r_mag^2) * n1_hat'));
                    daj2y_dx = factorJ2 * (((5*y / r_mag^2) * ((7*z^2 /r_mag^2) - 1)*r_vec') - ((10*y*z) /r_mag^2) * n3_hat' + (1 - (5 * z^2 / r_mag^2) * n2_hat'));
                    daj2z_dx = factorJ2 * (((5*z / r_mag^2) * ((7*z^2 /r_mag^2) - 3)*r_vec') + (3 - 15*z^2/r_mag^2) * n3_hat');    
                    
                    % Gradient of a_J2 perturbation wrt to state vector
                    daj2_dx = [
                        daj2x_dx, 0, 0, 0;
                        daj2y_dx, 0, 0, 0;
                        daj2z_dx, 0, 0, 0;
                        ];
                
                    % Form A matrix = Jacobian of EoM wrt to state x at time k    
                    A = df0_dx + B * daj2_dx;
                end

            %% 2.5) 2BP Propagators + Perturbations - Various State Representations
            
                %  [ dxdt ]           2BP + J2 -- Cartesian Coordinates
                function [x_dot] = dynamics_2BP_cartesian_J2(t, X, mu, J2, radPlanet)
                        % DYNAMICS_2BP_CARTESIAN_J2 Calculates the state vector derivative for a spacecraft in a
                        % perturbed two-body problem in Cartesian coordinates considering J2 perturbation.
                        %
                        % Usage:
                        %   x_dot = AppliedControlAstronauticsLibrary.dynamics_2BP_cartesian_J2(t, X)
                        %
                        % Where:
                        %   t is the current time (unused as the problem is autonomous)
                        %   X is the state vector [x; y; z; vx; vy; vz]
                        %   x_dot is the derivative of the state vector
                        %
                        % The function computes the gravitational acceleration including the J2 perturbation
                        % from Earth's oblateness and returns the derivative of the position and velocity
                        % of the spacecraft.
                    
                        % Constants
                        % mu = 398600.0;  % Earth's gravitational parameter (mu) in km^3/s^2
                        % J2 = 0.0010826; % J2 coefficient
                        % ro = 6378.1;    % Earth's equatorial radius [km]
                        
                        % Extract position components from the state vector
                        x = X(1);       % Pos X [km]
                        y = X(2);       % Pos Y [km]
                        z = X(3);       % Pos Z [km]
                        
                        % Extract velocity components from the state vector
                        vx = X(4);      % Vel X [km/s]
                        vy = X(5);      % Vel Y [km/s]
                        vz = X(6);      % Vel Z [km/s]
                        
                        % Calculate the distance from the center of the Earth
                        r = sqrt(x^2 + y^2 + z^2);
                        
                        % Calculate the gravitational acceleration components
                        dvxdt = -mu * x / r^3;     dvydt = -mu * y / r^3;    dvzdt = -mu * z / r^3;
                        
                        % Calculate the J2 perturbation acceleration components
                        z2_r2 = (z^2) / (r^2);
                        factor = -1.5 * J2 * mu * (radPlanet^2) / (r^5);
                    
                        aj2_x = factor * (x / r) * (1 - 5 * z2_r2);
                        aj2_y = factor * (y / r) * (1 - 5 * z2_r2);
                        aj2_z = factor * (z / r) * (3 - 5 * z2_r2);
                    
                        % Include the J2 perturbation in the acceleration components
                        dvxdt = dvxdt + aj2_x;
                        dvydt = dvydt + aj2_y;
                        dvzdt = dvzdt + aj2_z;
                        
                        % Assemble the derivative of the state vector including the J2 perturbation
                        x_dot = [vx; vy; vz; dvxdt; dvydt; dvzdt];
                end
                            
                %  [ dxdt, r_eci ] 2BP + J2 -- Milankovitch Coordinates
                function [x_dot,r_vec_eci] = dynamics_2BP_milankovitch_J2(t, X)            
                    % DYNAMICS_2BP_MILANKOVITCH_J2 Propagates the state of an orbiting object under
                    % the influence of J2 perturbations using Milankovitch orbital elements.
                    % This function computes the time derivative of the state vector and the
                    % position vector in the ECI frame for a given set of Milankovitch elements.
                    %
                    % Inputs:
                    %   t - Time variable (
                    % not used in this function, included for ode45 compatibility).
                    %   X - State vector in Milankovitch elements:
                    %       X(1:3) - Specific angular momentum vector (h_vec) in ECI frame.
                    %       X(4:6) - Eccentricity vector (e_vec) in ECI frame.
                    %       X(7)   - True longitude (L), in radians.
                    %   mu - Standard gravitational parameter of the central body.
                    %   libCall - Library call for additional functions like 'OrbitalElementsToDCM'.
                    %
                    % Outputs:
                    %   x_dot - Time derivative of the state vector.
                    %   r_vec_eci - Position vector in the ECI frame.
                    %
                    % Example call:
                    %   [x_dot, r_vec_eci] = AppliedControlAstronauticsLibrary.dynamics_2BP_milankovitch_J2(t, X, mu, AAE590Functions);
                    % Where:
                    %   - t is the time (scalar)
                    %   - X is the current state vector (7x1 vector)
                    %   - mu is the gravitational parameter (scalar)
                    % The function returns the time derivative of the state vector (x_dot) and the
                    
                    libCall = KeplerianOrbitalMechanicsLibrary();
                    
                    % Constants
                    mu = 398600.0;  % Earth's gravitational parameter (mu) in km^3/s^2
                    
                    % Extract State Vector Elements for Milankovitch elements    
                    h_vec_eci = X(1:3);
                    hz = h_vec_eci(3);
                    e_vec_eci = X(4:6);    
                    L = X(7);               %True longitude [rads]
                
                    % ECI vectors
                    z_vec = [0;0;1];        % Vector pointing +z direction in ECI            
                    x_vec = [1;0;0];        % Vector pointing +x direction in ECI            
                    
                    % Magnitudes of h and e vectors
                    e_mag = norm(e_vec_eci);     % Magnitude of e vector 
                    e_vec_hat = e_vec_eci/e_mag; % Unit eccentricity vector
                    h_mag = norm(h_vec_eci);     % Magnitude of h vector 
                    h_hat_eci = h_vec_eci/h_mag;
                    
                    p = norm(h_vec_eci)^2/mu;              % Semilatus rectum
                    i = acosd(hz/h_mag);                   % Recover inclination from Milankovitch X0     
                    n_vec_eci = cross(z_vec,h_hat_eci);    % Find Line of Nodes unit vector 
                    a = p / (1 - e_mag^2);                 % Calculate the semimajor axis
                
                    % Normalize the line of nodes vector to get the LoN unit vector
                    n_vec_hat = n_vec_eci / norm(n_vec_eci);
                
                    RAAN = acosd(dot(x_vec,n_vec_hat));         % RAAN angle in degrees is angle between LINE OF NODES and X+
                    omega = acosd(dot(n_vec_hat,e_vec_hat));    % Argument of periapsis angle
                    nu = rad2deg(L) - RAAN - omega;             % True anomaly angle
                    omega_plus_nu = rad2deg(L) - RAAN;  % Calculate longitude of periapsis nu+omega
                
                    if n_vec_hat(2) < 0 % Correct the quadrant for RAAN
                        RAAN = 360 - RAAN;
                    end
                    
                    % Calculate DCM313 at current point using angles found above
                    DCM313 = libCall.OrbitalElementsToDCM(RAAN,i,omega_plus_nu); 
                
                    % Compute the position vector in ECI coordinates with new DCM313 
                    % We start from the position vector in rotating frame of ref
                    r_mag = (a*(1-e_mag^2)) / (1+(e_mag*cosd(nu)));    
                    r_vec_rot = [r_mag;0;0];
                
                    % Now convert r vector in RTN (rotating) to ECI frame
                    r_vec_eci  = libCall.vec_rot_to_eci(DCM313,r_vec_rot)';
                
                    % Calculate the J2 perturbation acceleration in ECI coordinates
                    aj2_vec_eci = AppliedControlAstronauticsLibrary.calc_J2_accel_cartesian(r_vec_eci);
                
                    % Calculate the velocity vector in rotation frame (for ease) for Gaussian planetary elements, matrix
                    % B
                    
                    % Find flight path angle as a function of e and nu first 
                    fpa = atand((e_mag*sind(nu)/(1+e_mag*cosd(nu))));
                    
                    % Find speed magnitude as a funtion for mu, r, a
                    v_mag = sqrt(2*(mu/r_mag)-(mu/a));
                
                    % Velocity as vector RTN (rotating) frame      
                    v_vec_rot = [v_mag*sind(fpa),v_mag*cosd(fpa),0]; 
                
                    % Velocity vector from RTN (rotating) to ECI frame 
                    v_vec_eci  = libCall.vec_rot_to_eci(DCM313,v_vec_rot)';
                    
                    % Unperturbed Component for rate of change of state vector 
                    % f_0(x) computation    
                    f0_milankovitch = [0; 0; 0; 0; 0; 0; h_mag/r_mag^2]; 
                                    
                    % Define the B matrix using Gaussian planetary equations for Milankovitch
                    
                    % Calculate skew-symmetric matrix for r_vec
                    r_tilde = [  0      -r_vec_eci(3)  r_vec_eci(2);
                                r_vec_eci(3)  0      -r_vec_eci(1);
                            -r_vec_eci(2)  r_vec_eci(1)  0     ];
                    
                    % Calculate skew-symmetric matrix for h_vec
                    h_tilde = [  0      -h_vec_eci(3)  h_vec_eci(2);
                                h_vec_eci(3)  0      -h_vec_eci(1);
                            -h_vec_eci(2)  h_vec_eci(1)  0     ];
                
                    % Calculate skew-symmetric matrix for v_vec
                    v_tilde = [  0      -v_vec_eci(3)  v_vec_eci(2);
                                v_vec_eci(3)  0      -v_vec_eci(1);
                            -v_vec_eci(2)  v_vec_eci(1)  0     ];
                    
                    % Calculate dot products
                    z_dot_r = dot(z_vec, r_vec_eci);
                    z_dot_h = dot(z_vec, h_vec_eci);
                    scalar_factor = (z_dot_r / (h_mag * (h_mag + z_dot_h)));
                
                    % Construct the B matrix
                    B = [r_tilde; ...
                        (1/mu) * (cross(v_tilde,r_tilde)- h_tilde); ...
                        scalar_factor * h_vec_eci'];
                
                    % Compute the perturbed rate of change of the equinoctial elements
                    x_dot = f0_milankovitch + B * aj2_vec_eci;   
                        
                end     
                
                %  [ dxdt, r_eci ] 2BP + J2 -- Equinoctial Coordinates
                function [x_dot,r_vec_eci] = dynamics_2BP_equinoctial_J2(t, X)            
                    % Constants
                    mu = 398600.0;       % Earth's gravitational parameter (mu) in km^3/s^2
                    r0 = 6378.14;        % Earth's radius
                    J2 = 1.0826e-3;      %J2 perturbation coefficient
                
                    % Extract State Vector Elements for modified equinoctial parameters
                    % Ref: "A Survey of orbital elements"
                    p = X(1);
                    f = X(2);
                    g = X(3);
                    h = X(4);
                    k = X(5);
                    L = X(6);       %   [rads]
                    
                    % Unperturbed Component for rate of change of state vector
                    % f_0(x) computation
                    
                    q = 1 + f*cos(L) + g*sin(L);        % L is in [rad]    
                    s_sq = 1 + h^2 + k^2;
                    L_dot = sqrt(mu*p) * (q/p)^2;       %in rad/s, uses mean motion
                    
                    f0_equinoctial = [0; 0; 0; 0; 0; L_dot]; % Only L_dot is non-zero for unperturbed motion
                
                    % Recover COE angles from modified equinoctial state    
                    RAAN = atan2d(k, h);                % RAAN [deg]
                    i = 2 * atand(sqrt(h^2 + k^2));     % Inclination [deg]
                    omega_plus_nu = rad2deg(L) - RAAN;  % [deg]       
                            
                    % Compute the position vector in ECI coordinates with new DCM313    
                    r = p / q;                     % (Walker, 1985. pg4) 
                
                
                    % Calculate DCM313 Matrix - RTN (rotating) to ECI
                    DCM313 =  libCall.OrbitalElementsToDCM(RAAN,i,omega_plus_nu);
                
                    % Compute the position vector in ECI coordinates with new DCM313
                    r_vec_rot = [r; 0; 0];  % [km]
                    r_vec_eci = libCall.vec_rot_to_eci(DCM313,r_vec_rot)';
                
                    % Calculate the J2 perturbation in the RTN frame (rotating) components
                    % as per document shared in Brightspace 
                    C = - (mu*J2*r0^2)/(r^4);
                    aj2R = 1.5 * C * (1 - (12 * (h * sin(L) - k * cos(L))^2) / (1 + h^2 + k^2)^2);
                    aj2T = 12 * C * ((h * sin(L) - k * cos(L)) * (h * cos(L) + k * sin(L))) / ((1 + h^2 + k^2)^2);
                    aj2N = 6 * C * ((1 - h^2 - k^2) * (h * sin(L) - k * cos(L))) / ((1 + h^2 + k^2)^2);
                    
                    % Perturbation J2 vector in RTN direction
                    aj2_vec_rtn = [aj2R; aj2T; aj2N];
                
                    % Define the B matrix from the image provided
                    B = sqrt(p/mu) * ...
                        [0,        2*p/q,                    0;
                        sind(L),  ((q+1)*cosd(L)+f)/q,     -g*(h*sind(L)-k*cosd(L))/q;
                        -cosd(L), ((q+1)*sind(L)+g)/q,     f*(h*sind(L)-k*cosd(L))/q;
                        0,        0,                       s_sq*cosd(L)/(2*q);
                        0,        0,                       s_sq*sind(L)/(2*q);
                        0,        0,                       (h*sind(L) - k*cosd(L))/q];
                
                    % Compute the perturbed rate of change of the equinoctial elements
                    x_dot = f0_equinoctial + B * aj2_vec_rtn;
                end     
                    
                %  [ dxdt, r_eci ] 2BP + J2 -- Keplerian Coordinates
                function [x_dot,r_vec_eci] = dynamics_2BP_keplerian_J2(t,X)
                
                    libCall = KeplerianOrbitalMechanicsLibrary();
                    
                    % Constants
                    mu = 398600.0;  % Earth's gravitational parameter (mu) in km^3/s^2
                        
                    % Extract State Vector Elements
                    a = X(1);
                    e = X(2);
                    i = X(3);
                    RAAN = X(4);
                    omega = X(5);
                    M = X(6);    
                    n = sqrt(mu/a^3);  % mean motion [rad/s]
                    
                    % Compute the unperturbed rate of change of the Keplerian elements
                    f0_coe = [0; 0; 0; 0; 0; n];    %M_dot is n
                
                    % Compute semi-latus, specific angular momentum and semi-minor axis 
                    b = libCall.b_ae(a,e);                 % semi-minor axis[km]
                    h = libCall.h_uae(mu,a,e);             % Specific angular momentum
                    p = libCall.p_ae(a,e);                 % Semilatus Rectum
                
                    % Compute the Eccentric Anomaly (E) using the provided M value
                    E = libCall.E_Me(M, e);  %Based on N-Rhapson iterative approach
                    
                    % Compute true anomaly (ν) from Eccentric Anomaly (E) and the radius 
                    nu = libCall.ta_eE(e,E);  
                    r = libCall.r_aeta(a,e,nu);  
                    
                    % Calculate New DCM313 Matrix - RTN (rotating) to ECI
                    DCM313 =  libCall.OrbitalElementsToDCM(RAAN,i,omega+nu);
                    DCM313inv = inv(DCM313);
                
                    % Compute the position vector in ECI coordinates with new DCM313
                    r_vec_rot = [r; 0; 0];  % [km]
                    r_vec_eci = libCall.vec_rot_to_eci(DCM313,r_vec_rot)';
                    
                    % Calculate the J2 perturbation acceleration in ECI coordinates
                    aj2_vec_eci = AppliedControlAstronauticsLibrary.calc_J2_accel_cartesian(r_vec_eci);
                
                    % Convert J2 acceleration from ECI back to RTN (rotating) 
                    aj2_vec_rot = libCall.vec_eci_to_rot(DCM313inv, aj2_vec_eci);
                
                    % Pre-compute common terms for B(X) matrix
                    sin_nu = sind(nu);
                    cos_nu = cosd(nu);
                    sin_i = sind(i);
                    tan_i = tand(i);
                    
                    % B matrix construction
                    B = (1/h) * ...
                        [2*a^2*e*sin_nu,               2*a^2*p/r,                   0;
                        p*sin_nu,                     (p+r)*cos_nu + r*e,          0;
                        0,                            0,                           r*cosd(nu + omega);
                        0,                            0,                           r*sind(nu + omega)/sin_i;
                        -p*cos_nu/e,                  (p+r)*sin_nu/e,              -r*sind(nu + omega)/tan_i;
                        b*p*cos_nu/(a*e) - 2*b*r/a,  -b*(p+r)*sin_nu/(a*e),                        0];
                
                % Compute the perturbed rate of change of the Keplerian elements
                    x_dot = f0_coe + B * aj2_vec_rot';
                end
                
                %  [ dxdt ]           2BP + J2 + SRP -- Cartesian Coordinates
                function [x_dot] = dynamics_2BP_cartesian_J2_SRP(t,X)
                    % DYNAMICS_2BP_CARTESIAN_J2 Calculates the state vector derivative for a spacecraft in a
                    % perturbed two-body problem in Cartesian coordinates considering J2 and SRP perturbation.
                    %
                    % Usage:
                    %   x_dot = AppliedControlAstronauticsLibrary.dynamics_2BP_cartesian_J2(t, X)
                    %
                    % Where:
                    %   t is the current time (unused as the problem is autonomous)
                    %   X is the state vector [x; y; z; vx; vy; vz]
                    %   x_dot is the derivative of the state vector
                    %
                    % The function computes the gravitational acceleration including the J2 perturbation
                    % from Earth's oblateness and returns the derivative of the position and velocity
                    % of the spacecraft.
                
                    % Constants
                    mu = 398600.0;  % Earth's gravitational parameter (mu) in km^3/s^2
                    
                    % Extract position components from the state vector
                    r_vec_eci = X(1:3);        % [km]
                    x = r_vec_eci(1);          % [km]
                    y = r_vec_eci(2);          % [km]
                    z = r_vec_eci(3);          % [km]
                    r = norm(r_vec_eci);       % [km]
                    
                    % Extract velocity vector from the state vector
                    v_vec_eci = X(4:6);        % [km]
                    
                    % Calculate the gravitational acceleration components 2BP based
                    % f0 rate of change of state vector 
                    f0 = [v_vec_eci ; -mu * x / r^3; -mu * y / r^3 ; -mu * z / r^3];
                        
                    aj2_vec_eci = AppliedControlAstronauticsLibrary.calc_J2_accel_cartesian(r_vec_eci);
                    
                    % Calculate cannon ball perturbation in cartesian
                    aSRP_vec_eci = AppliedControlAstronauticsLibrary.calc_SRP_accel_cartesian(t, r_vec_eci);
                    
                    % B matrix 
                    B = [zeros(3,3);eye(3)];
                    
                    % Assemble the derivative of the state vector including the J2 perturbation    
                    x_dot = f0 + B * (aj2_vec_eci + aSRP_vec_eci);    
                end
            
                %  [ dxdt, r_eci ] 2BP + J2 + SRP -- Milankovitch Coordinates
                function [x_dot,r_vec_eci] = dynamics_2BP_milankovitch_J2_SRP(t, X)                       
                    % DYNAMICS_2BP_MILANKOVITCH_J2 Propagates the state of an orbiting object under
                    % the influence of J2 and SRP perturbations using Milankovitch orbital elements.
                    % This function computes the time derivative of the state vector and the
                    % position vector in the ECI frame for a given set of Milankovitch elements.
                    %
                    % Inputs:
                    %   t - Time variable (
                    % not used in this function, included for ode45 compatibility).
                    %   X - State vector in Milankovitch elements:
                    %       X(1:3) - Specific angular momentum vector (h_vec) in ECI frame.
                    %       X(4:6) - Eccentricity vector (e_vec) in ECI frame.
                    %       X(7)   - True longitude (L), in radians.
                    %   mu - Standard gravitational parameter of the central body.
                    %   libCall - Library call for additional functions like 'OrbitalElementsToDCM'.
                    %
                    % Outputs:
                    %   x_dot - Time derivative of the state vector.
                    %   r_vec_eci - Position vector in the ECI frame.
                    %
                    % Example call:
                    %   [x_dot, r_vec_eci] = dynamics_2BP_milankovitch_J2(t, X, mu, AAE590Functions);
                    % Where:
                    %   - t is the time (scalar)
                    %   - X is the current state vector (7x1 vector)
                    %   - mu is the gravitational parameter (scalar)
                    % The function returns the time derivative of the state vector (x_dot) and the
                    
                    libCall = KeplerianOrbitalMechanicsLibrary();
                        
                    % Constants
                    mu = 398600.0;  % Earth's gravitational parameter (mu) in km^3/s^2
                    
                    % Extract State Vector Elements for Milankovitch elements    
                    h_vec_eci = X(1:3);
                    hz = h_vec_eci(3);
                    e_vec_eci = X(4:6);    
                    L = X(7);               %True longitude [rads]
                
                    % ECI vectors
                    z_vec = [0;0;1];        % Vector pointing +z direction in ECI            
                    x_vec = [1;0;0];        % Vector pointing +x direction in ECI            
                    
                    % Magnitudes of h and e vectors
                    e_mag = norm(e_vec_eci);     % Magnitude of e vector 
                    e_vec_hat = e_vec_eci/e_mag; % Unit eccentricity vector
                    h_mag = norm(h_vec_eci);     % Magnitude of h vector 
                    h_hat_eci = h_vec_eci/h_mag;
                    
                    p = norm(h_vec_eci)^2/mu;              % Semilatus rectum
                    i = acosd(hz/h_mag);                   % Recover inclination from Milankovitch X0     
                    n_vec_eci = cross(z_vec,h_hat_eci);    % Find Line of Nodes unit vector 
                    a = p / (1 - e_mag^2);                 % Calculate the semimajor axis
                
                    % Normalize the line of nodes vector to get the LoN unit vector
                    n_vec_hat = n_vec_eci / norm(n_vec_eci);
                
                    RAAN = acosd(dot(x_vec,n_vec_hat));         % RAAN angle in degrees is angle between LoN and X+
                    omega = acosd(dot(n_vec_hat,e_vec_hat));    % Argument of periapsis angle
                    nu = rad2deg(L) - RAAN - omega;                      % True anomaly angle
                    omega_plus_nu = rad2deg(L) - RAAN;  % Calculate longitude of periapsis nu+omega
                
                    if n_vec_hat(2) < 0 % Correct the quadrant for RAAN
                        RAAN = 360 - RAAN;
                    end
                    
                    % Calculate DCM313 at current point using angles found above
                    DCM313 = libCall.OrbitalElementsToDCM(RAAN,i,omega_plus_nu); 
                
                    % Compute the position vector in ECI coordinates with new DCM313 
                    % We start from the position vector in rotating frame of ref
                    r_mag = (a*(1-e_mag^2)) / (1+(e_mag*cosd(nu)));    
                    r_vec_rot = [r_mag;0;0];
                
                    % Now convert r vector in RTN (rotating) to ECI frame
                    r_vec_eci  = libCall.vec_rot_to_eci(DCM313,r_vec_rot)';
                
                    % Calculate the J2 perturbation acceleration in ECI coordinates
                    aj2_vec_eci = AppliedControlAstronauticsLibrary.calc_J2_accel_cartesian(r_vec_eci);
                
                    % Calculate SRP cannonball perturbation in ECI
                    aSRP_vec_eci = AppliedControlAstronauticsLibrary.calc_SRP_accel_cartesian(t, r_vec_eci);
                
                    % Calculate the velocity vector in rotation frame (for ease) for Gaussian planetary elements, matrix
                    % B
                    
                    % Find flight path angle as a function of e and nu first 
                    fpa = atand((e_mag*sind(nu)/(1+e_mag*cosd(nu))));
                    
                    % Find speed magnitude as a funtion for mu, r, a
                    v_mag = sqrt(2*(mu/r_mag)-(mu/a));
                
                    % Velocity as vector RTN (rotating) frame      
                    v_vec_rot = [v_mag*sind(fpa),v_mag*cosd(fpa),0]; 
                
                    % Velocity vector from RTN (rotating) to ECI frame 
                    v_vec_eci  = libCall.vec_rot_to_eci(DCM313,v_vec_rot)';
                    
                    % Unperturbed Component for rate of change of state vector 
                    % f_0(x) computation    
                    f0_milankovitch = [0; 0; 0; 0; 0; 0; h_mag/r_mag^2]; 
                                    
                    % Define the B matrix using Gaussian planetary equations for Milankovitch
                    
                    % Calculate skew-symmetric matrix for r_vec
                    r_tilde = [  0      -r_vec_eci(3)  r_vec_eci(2);
                                r_vec_eci(3)  0      -r_vec_eci(1);
                            -r_vec_eci(2)  r_vec_eci(1)  0     ];
                    
                    % Calculate skew-symmetric matrix for h_vec
                    h_tilde = [  0      -h_vec_eci(3)  h_vec_eci(2);
                                h_vec_eci(3)  0      -h_vec_eci(1);
                            -h_vec_eci(2)  h_vec_eci(1)  0     ];
                
                    % Calculate skew-symmetric matrix for v_vec
                    v_tilde = [  0      -v_vec_eci(3)  v_vec_eci(2);
                                v_vec_eci(3)  0      -v_vec_eci(1);
                            -v_vec_eci(2)  v_vec_eci(1)  0     ];
                    
                    % Calculate dot products
                    z_dot_r = dot(z_vec, r_vec_eci);
                    z_dot_h = dot(z_vec, h_vec_eci);
                    scalar_factor = (z_dot_r / (h_mag * (h_mag + z_dot_h)));
                
                    % Construct the B matrix
                    B = [r_tilde; ...
                        (1/mu) * (cross(v_tilde,r_tilde)- h_tilde); ...
                        scalar_factor * h_vec_eci'];
                
                    % Compute the perturbed rate of change of the equinoctial elements
                    x_dot = f0_milankovitch + B * (aj2_vec_eci + aSRP_vec_eci);
                
                end     
            
                %  [ dxdt, r_eci ] 2BP + J2 + SRP  -- Equinoctial Coordinates
                function [x_dot,r_vec_eci] = dynamics_2BP_equinoctial_J2_SRP(t, X)
                    
                    libCall = KeplerianOrbitalMechanicsLibrary();
                    
                    % Constants
                    mu = 398600.0;       % Earth's gravitational parameter (mu) in km^3/s^2
                    r0 = 6378.14;        % Earth's radius
                    J2 = 1.0826e-3;      %J2 perturbation coefficient
                
                    % Extract State Vector Elements for modified equinoctial parameters
                    % Ref: "A Survey of orbital elements"
                    p = X(1);
                    f = X(2);
                    g = X(3);
                    h = X(4);
                    k = X(5);
                    L = X(6);       %   [rads]
                    
                    % Unperturbed Component for rate of change of state vector
                    % f_0(x) computation
                    
                    q = 1 + f*cos(L) + g*sin(L);        % L is in [rad]    
                    s_sq = 1 + h^2 + k^2;
                    L_dot = sqrt(mu*p) * (q/p)^2;       %in rad/s, uses mean motion
                    
                    f0_equinoctial = [0; 0; 0; 0; 0; L_dot]; % Only L_dot is non-zero for unperturbed motion
                
                    % Recover COE angles from modified equinoctial state    
                    RAAN = atan2d(k, h);                % RAAN [deg]
                    i = 2 * atand(sqrt(h^2 + k^2));     % Inclination [deg]
                    omega_plus_nu = rad2deg(L) - RAAN;  % [deg]       
                            
                    % Compute the position vector in ECI coordinates with new DCM313    
                    r = p / q;                     % (Walker, 1985. pg4) 
                
                    % Calculate the J2 perturbation in the RTN frame (rotating) components
                    % as per document shared in Brightspace - directly in RTN frame
                    C = - (mu*J2*r0^2)/(r^4);
                    aj2R = 1.5 * C * (1 - (12 * (h * sin(L) - k * cos(L))^2) / (1 + h^2 + k^2)^2);
                    aj2T = 12 * C * ((h * sin(L) - k * cos(L)) * (h * cos(L) + k * sin(L))) / ((1 + h^2 + k^2)^2);
                    aj2N = 6 * C * ((1 - h^2 - k^2) * (h * sin(L) - k * cos(L))) / ((1 + h^2 + k^2)^2);
                    
                    % Perturbation J2 vector in RTN direction 
                    aj2_vec_rtn = [aj2R; aj2T; aj2N];
                    
                    % Calculate DCM313 Matrix - RTN (rotating) to ECI
                    DCM313 =  libCall.OrbitalElementsToDCM(RAAN,i,omega_plus_nu);
                    DCM313inv = inv(DCM313);
                
                    % Compute the position vector in ECI coordinates with new DCM313
                    r_vec_rot = [r; 0; 0];  % [km]
                    r_vec_eci = libCall.vec_rot_to_eci(DCM313,r_vec_rot)';
                
                    % Calculate SRP cannonball perturbation in ECI
                    aSRP_vec_eci = AppliedControlAstronauticsLibrary.calc_SRP_accel_cartesian(t, r_vec_eci);
                
                    % Convert SRP Cannoball acceleration from ECI back to RTN (rotating) 
                    aSRP_vec_rtn = libCall.vec_eci_to_rot(DCM313inv, aSRP_vec_eci);    
                
                    % Define the B matrix from the image provided
                    B = sqrt(p/mu) * ...
                        [0,        2*p/q,                    0;
                        sind(L),  ((q+1)*cosd(L)+f)/q,     -g*(h*sind(L)-k*cosd(L))/q;
                        -cosd(L), ((q+1)*sind(L)+g)/q,     f*(h*sind(L)-k*cosd(L))/q;
                        0,        0,                       s_sq*cosd(L)/(2*q);
                        0,        0,                       s_sq*sind(L)/(2*q);
                        0,        0,                       (h*sind(L) - k*cosd(L))/q];
                
                    % Compute the perturbed rate of change of the equinoctial elements
                    x_dot = f0_equinoctial + B * (aj2_vec_rtn + aSRP_vec_rtn);
                    x_dot = f0_equinoctial + B * (aj2_vec_rtn);
                    
                end     
                
                %  [ dxdt, r_eci ] 2BP + J2 + SRP -- Keplerian Coordinates
                function [x_dot,r_vec_eci] = dynamics_2BP_keplerian_J2_SRP(t,X)
        
                    libCall = KeplerianOrbitalMechanicsLibrary();
                    
                    % Constants
                    mu = 398600.0;  % Earth's gravitational parameter (mu) in km^3/s^2
                        
                    % Extract State Vector Elements
                    a = X(1);
                    e = X(2);
                    i = X(3);
                    RAAN = X(4);
                    omega = X(5);
                    M = X(6);    
                    n = sqrt(mu/a^3);  % mean motion [rad/s]
                    
                    % Compute the unperturbed rate of change of the Keplerian elements
                    f0_coe = [0; 0; 0; 0; 0; n];    %M_dot is n
                
                    % Compute semi-latus, specific angular momentum and semi-minor axis 
                    b = libCall.b_ae(a,e);                 % semi-minor axis[km]
                    h = libCall.h_uae(mu,a,e);             % Specific angular momentum
                    p = libCall.p_ae(a,e);                 % Semilatus Rectum
                
                    % Compute the Eccentric Anomaly (E) using the provided M value
                    E = libCall.E_Me(M, e);  %Based on N-Rhapson iterative approach
                    
                    % Compute true anomaly (ν) from Eccentric Anomaly (E) and the radius 
                    nu = libCall.ta_eE(e,E);  
                    r = libCall.r_aeta(a,e,nu);  
                    
                    % Calculate New DCM313 Matrix - RTN (rotating) to ECI
                    DCM313 =  libCall.OrbitalElementsToDCM(RAAN,i,omega + nu);
                    DCM313inv = inv(DCM313);
                
                    % Compute the position vector in ECI coordinates with new DCM313
                    r_vec_rot = [r; 0; 0];  % [km]
                    r_vec_eci = libCall.vec_rot_to_eci(DCM313,r_vec_rot)';
                    
                    % Calculate the J2 perturbation acceleration in ECI coordinates
                    aj2_vec_eci = AppliedControlAstronauticsLibrary.calc_J2_accel_cartesian(r_vec_eci);
                
                    % Convert J2 acceleration from ECI back to RTN (rotating) 
                    aj2_vec_rot = libCall.vec_eci_to_rot(DCM313inv, aj2_vec_eci);
                
                    % Calculate SRP cannonball perturbation in cartesian
                    aSRP_vec_eci = AppliedControlAstronauticsLibrary.calc_SRP_accel_cartesian(t, r_vec_eci);
                
                    % Convert SRP Cannoball acceleration from ECI back to RTN (rotating) 
                    aSRP_vec_rot = libCall.vec_eci_to_rot(DCM313inv, aSRP_vec_eci);
                    
                    % Pre-compute common terms for B(X) matrix
                    sin_nu = sind(nu);
                    cos_nu = cosd(nu);
                    sin_i = sind(i);
                    tan_i = tand(i);
                    
                    % B matrix construction
                    B = (1/h) * ...
                        [2*a^2*e*sin_nu,               2*a^2*p/r,                   0;
                        p*sin_nu,                     (p+r)*cos_nu + r*e,          0;
                        0,                            0,                           r*cosd(nu + omega);
                        0,                            0,                           r*sind(nu + omega)/sin_i;
                        -p*cos_nu/e,                  (p+r)*sin_nu/e,              -r*sind(nu + omega)/tan_i;
                        b*p*cos_nu/(a*e) - 2*b*r/a,  -b*(p+r)*sin_nu/(a*e),                        0];
                
                % Compute the perturbed rate of change of the Keplerian elements
                    x_dot = f0_coe + B * (aj2_vec_rot' + aSRP_vec_rot');
                end
            %% 2.6) Clohessy Wiltshire Hill Dynamics 

                % CWH Non Linear Equation Dynamics
                function dxdt = CWH_dynamics(t, x, n, u)
                    % CWH_DYNAMICS: Computes the time derivative of the state for the
                    % Clohessy-Wiltshire-Hill (CWH) equations.
                    %
                    % Inputs:
                    %   t   - Time (not used explicitly in the equations, but required for ODE solvers)
                    %   x   - State vector [x, y, z, vx, vy, vz]
                    %   n   - Mean motion of the chief satellite (constant)
                    %   u   - Control inputs [ux, uy, uz] (optional, default is [0, 0, 0])
                    %
                    % Outputs:
                    %   dxdt - Time derivative of the state vector [vx, ax, vy, ay, vz, az]
                    
                    % Check if control input is provided, otherwise assume zero control input
                    if nargin < 4
                        u = [0; 0; 0];  % Default control input [ux, uy, uz]
                    end
                    
                    % State vector components
                    x_pos = x(1);  % x-position
                    y_pos = x(2);  % y-position
                    z_pos = x(3);  % z-position
                    vx = x(4);     % x-velocity   
                    vy = x(5);     % y-velocity    
                    vz = x(6);     % z-velocity
                    
                    % CWH equations of motion (nonlinear)
                    ax = 3*n^2*x_pos + 2*n*vy + u(1);        % Acceleration in x
                    ay = -2*n*vx + u(2);                     % Acceleration in y
                    az = -n^2*z_pos + u(3);                  % Acceleration in z
                    
                    % Time derivative of the state vector
                    dxdt = [vx; vy; vz; ax; ay; az];
                end
            
                % CWH STM
                function STM = clohessy_wiltshire_stm(nc, tk, tk_1)
                    % CLOHESSY_WILTSHIRE_STM: Computes the State Transition Matrix (STM) 
                    % for the Clohessy-Wiltshire equations based on mean motion nc and 
                    % time steps tk and tk_1.
                    %
                    % Inputs:
                    %   nc  - Mean motion of the inspector satellite
                    %   tk  - Time at the current step
                    %   tk_1 - Time at the previous step
                    %
                    % Outputs:
                    %   STM - 4x4 State Transition Matrix for the Clohessy-Wiltshire equations
                
                    % Calculate psi, the phase angle between both SC
                    psi = nc * (tk - tk_1);
                    
                    % Define each submatrix in the STM
                    
                    % Phi_rr: 2x2 submatrix
                    Phi_rr = [4 - 3 * cos(psi), 0;
                            6 * (sin(psi) - psi), 1];
                    
                    % Phi_rv: 2x2 submatrix
                    Phi_rv = [1 / nc * sin(psi), 2 / nc * (1 - cos(psi));
                            2 / nc * (cos(psi) - 1), 4 / nc * sin(psi) - 3 / nc * psi];
                    
                    % Phi_vr: 2x2 submatrix
                    Phi_vr = [3 * nc * sin(psi), 0;
                            6 * nc * (cos(psi) - 1), 0];
                    
                    % Phi_vv: 2x2 submatrix
                    Phi_vv = [cos(psi), 2 * sin(psi);
                            -2 * sin(psi), -3 + 4 * cos(psi)];
                    
                    % Combine submatrices to form the 4x4 STM
                    STM = [Phi_rr, Phi_rv;
                        Phi_vr, Phi_vv];
                end
            
            %% 2.7) 2BP - Converter Functions
                %% - - 2.7.1) State Representations
            
                    % [x_eci] Keplerian to ECI Converter Function 
                    function X_eci_matrix = convertKeplerianToECI(X_coe, mu, libCall)
                        % CONVERTKEPLERIANTOECI Convert Keplerian orbital elements to ECI coordinates.
                        %
                        % Usage:
                        %   X_pos_keplerian = convertKeplerianToECI(T_coe, X_coe, calc)
                        %
                        % Where:
                        %   T_coe - Time array for Keplerian elements.
                        %   X_coe - Matrix of Keplerian elements at each time step.
                        %   libCall - Instance of AAE590Functions class with necessary orbital mechanics functions.
                        %
                        % Returns:
                        %   X_pos_keplerian - Matrix of positions in ECI coordinates.
                        
                        % Initialize the matrix to store position in ECI coordinates
                        X_eci_matrix = zeros(length(X_coe), 6);
                    
                        % Loop through each time step to convert Keplerian elements to ECI coordinates
                        for idx = 1:length(X_coe)
                            % Extract the Keplerian elements
                            a = X_coe(idx, 1);
                            e = X_coe(idx, 2);
                            i = X_coe(idx, 3);
                            RAAN = X_coe(idx, 4);
                            omega = X_coe(idx, 5);
                            M = X_coe(idx, 6);
                    
                            % Compute Eccentric Anomaly (E) and True Anomaly (nu)
                            E = E_Me(M, e);    % rads
                            nu = 2*atan(sqrt((1+e)/(1-e))*tan(E/2));            
                            
                            % Compute the orbital radius (r) based on True Anomaly (nu)
                            r = (a*(1-e^2)) / (1+(e*cos(nu)));
                    
                            % Generate the Direction Cosine Matrix (DCM) for r-θ-h (rotating) to ECI frame
                            DCM313_ith = libCall.OrbitalElementsToDCM(rad2deg(RAAN), rad2deg(i), rad2deg(omega + nu)); 
                    
                            % Compute the position vector in ECI coordinates
                            r_vec_rot = [r; 0; 0];  % Position vector in the orbital plane
                            r_vec_eci = libCall.vec_rot_to_eci(DCM313_ith, r_vec_rot); % Convert to ECI
                    
                            % Find flight path angle 
                            fpa = atan((e*sin(nu)/(1+e*cos(nu))));
                            
                            % Find speed magnitude as a funtion for mu, r, a
                            v = sqrt(2*(mu/r)-(mu/a));
                        
                            % Velocity as vector r-θ-h (rotating) frame      
                            v_vec_rot = [v*sin(fpa),v*cos(fpa),0]; 
                        
                            % Velocity vector from r-θ-h (rotating) to ECI frame 
                            v_vec_eci  = libCall.vec_rot_to_eci(DCM313_ith,v_vec_rot)';
                            
                            % Store the computed ECI coordinates (position and velocity)
                            X_eci_matrix(idx, 1:3) = r_vec_eci'; % Position
                            X_eci_matrix(idx, 4:6) = v_vec_eci'; % Velocity   
                        end
                    end
                        
                    % [x_eci, x_coe]  Modified Equinoctial to ECI + COE 
                    function [X_eci_matrix, COE] = convertEquinoctialToECI(X_eqn, mu, libCall)                    
                        % CONVERT EQUINOCTIAL TO ECI Convert Modified Equinoctial orbital elements to ECI coordinates.
                        %
                        % Usage:
                        %   [X_pos_equinoctial, COE_Equinoctial] = convertEquinoctialToECI(T_eqn, X_eqn, calc)
                        %
                        % Where:
                        %   T_eqn - Time array for Equinoctial elements.
                        %   X_eqn - Matrix of Equinoctial elements at each time step.
                        %   libCall - Instance of AAE590Functions class with necessary orbital mechanics functions.
                        %
                        % Returns:
                        %   X_pos_equinoctial - Matrix of positions in ECI coordinates.
                        %   COE_Equinoctial - Matrix of Classical Orbital Elements [a, e, i, RAAN].
                        
                        % Initialize the matrix to store position and velocity in ECI coordinates and COE
                        X_eci_matrix = zeros(length(X_eqn), 6);
                        COE = zeros(length(X_eqn), 5);  % [a, e, i, RAAN, omega + nu]
                    
                        % Loop through each time step to convert Equinoctial elements to ECI coordinates
                        for idx = 1:length(X_eqn)
                            % Extract the Equinoctial elements
                            p = X_eqn(idx,1);       % [km]
                            f = X_eqn(idx,2);       % [adim]
                            g = X_eqn(idx,3);       % [adim]
                            h = X_eqn(idx,4);       % [adim]
                            k = X_eqn(idx,5);       % [adim]
                            L = X_eqn(idx,6);       % [rad]
                    
                            % Compute classical orbital elements from modified equinoctial elements
                            q = 1 + f*cos(L) + g*sin(L);        % [adim]
                            RAAN_ith = atan2(k, h);             % Right Ascension of the Ascending Node in [rads]
                            i_ith = 2 * atan(sqrt(h^2 + k^2));  % Inclination in radians
                            e = sqrt(f^2 + g^2);                % Eccentricity
                            a = p / (1 - e^2);                  % Semi-major axis assuming elliptical orbit
                            omega_plus_nu = L - RAAN_ith;       % Argument of latitude (arg of peri + true anomaly) aka true longitude
                    
                            % Calculate the Direction Cosine Matrix (DCM) for r-θ-h (rotating) to ECI frame
                            DCM313 = libCall.OrbitalElementsToDCM_Rads(RAAN_ith, i_ith, omega_plus_nu); 
                    
                            % Compute the position vector in ECI coordinates
                            r = p / q;
                            r_vec_rot = [r; 0; 0];  % Position vector in the orbital plane
                            r_vec_eci = libCall.vec_rot_to_eci(DCM313, r_vec_rot); % Convert to ECI   
                                        
                            % Compute velocity in ECI coordinates
                            s_sq = 1 + h^2 + k^2;
                            q = 1 + f*cos(L) + g*sin(L);
                        
                            
                            v_vec_eci = [0; 0; 0]; % Velocity vector in ECI coordinates
                            
                            % Store the computed ECI coordinates and velocity
                            X_eci_matrix(idx, 1:3) = r_vec_eci;
                            X_eci_matrix(idx, 4:6) = v_vec_eci;
                            
                            % Store the COE elements in a matrix also 
                            COE(idx, :) = [a, e, rad2deg(i_ith), rad2deg(RAAN_ith),rad2deg(omega_plus_nu)];
                        end
                    end
                    
                    % [x_eci, x_coe] Milankovitch Orbital Elements to Earth Centered Inertial + Keplerian  
                    function [X_eci_matrix, COE] = convertMilankovitchToECI(X, mu, libCall)               
                    % This function converts a matrix of state vectors in Milankovitch elements
                        % to Earth-Centered Inertial (ECI) position vectors and corresponding
                        % Common Orbital Elements (COE).
                        %
                        % Parameters:
                        %   X        - A matrix of state vectors, each row is a state vector
                        %              in Milankovitch elements (nx7 matrix where n is the number
                        %              of state vectors).
                        %   mu       - The standard gravitational parameter (mu) of the central body.
                        %   libCall  - An object that provides the function `OrbitalElementsToDCM`
                        %              which computes the Direction Cosine Matrix (DCM) from
                        %              orbital elements.
                        %
                        % Returns:
                        %   r_vec_eci_matrix - A matrix where each row is an ECI position vector
                        %                      corresponding to a state vector from `X`.
                        %   COE_matrix       - A matrix where each row contains the COEs
                        %                      [a, e, i, RAAN, omega] for each state vector.
                    
                        % Initialize matrices to store results
                        num_states = size(X,1);
                        X_eci_matrix = zeros(num_states, 6);
                        COE = zeros(num_states, 5);
                        
                        z_vec = [0;0;1];        % Vector pointing +z direction in ECI            
                        x_vec = [1;0;0];        % Vector pointing +x direction in ECI            
                    
                        for idx = 1:num_states
                            % Extract the Milankovitch elements from the current state vector
                            h_vec_eci = X(idx, 1:3);
                            e_vec_eci = X(idx, 4:6);        
                            L = X(idx, 7);              % True longitude in radians
                            
                    
                            % Compute magnitudes and unit vectors for h and e
                            h = norm(h_vec_eci);
                            e = norm(e_vec_eci);
                            h_hat = h_vec_eci / h;
                            e_hat = e_vec_eci / e;
                    
                            hz = h_vec_eci(3);
                    
                            % Compute semilatus rectum and semimajor axis
                            p = h^2 / mu;
                            a = p / (1 - e^2);
                                                        
                            i = acos(hz/h);                        % Recover inclination [rad]
                            n_vec_eci = cross(z_vec,h_hat);    % Find Line of Nodes unit vector            
                            n_hat = n_vec_eci / norm(n_vec_eci); % Find LoN unit vector
                        
                            % The Euler angles are recovered from the vector relationships 
                            RAAN = atan2(norm(cross(x_vec,n_hat)),dot(x_vec,n_hat));
                    
                            % omega = acos(dot(n_hat,e_vec_hat));    % Argument of periapsis angle [rad]        
                            omega = atan2(norm(cross(e_hat,n_hat)),dot(e_hat,n_hat));
                    
                            nu = L - RAAN - omega;             % True anomaly angle [rad]    
                            
                            % Calculate DCM313 at current point using Euler angles recovered above
                            DCM313 = libCall.OrbitalElementsToDCM_Rads(RAAN,i,omega + nu);  
                        
                            % Compute the position vector in ECI coordinates with new DCM313 
                            % We start from the position vector in rotating frame of reference
                            r = (a*(1-e^2)) / (1+(e*cos(nu)));    
                            r_vec_rot = [r;0;0];
                        
                            % Now convert r vector in r-θ-h (rotating) to ECI frame
                            r_vec_eci  = libCall.vec_rot_to_eci(DCM313,r_vec_rot)';
                                        
                            % Find flight path angle as a function of e and nu first 
                            fpa = atan((e*sin(nu)/(1+e*cos(nu))));
                            
                            % Find speed magnitude as a funtion for mu, r, a
                            v_mag = sqrt(2*(mu/r)-(mu/a));
                        
                            % Velocity as vector r-θ-h (rotating) frame      
                            v_vec_rot = [v_mag*sin(fpa),v_mag*cos(fpa),0]; 
                        
                            % Velocity vector from r-θ-h (rotating) to ECI frame 
                            v_vec_eci  = libCall.vec_rot_to_eci(DCM313,v_vec_rot)';
                    
                    
                            % Store the results in the matrices
                            X_eci_matrix(idx, 1:3) = r_vec_eci';
                            X_eci_matrix(idx, 4:6) = v_vec_eci';
                            COE(idx, :) = [a, e, i, RAAN, omega];
                        end
                    end
                    
                    % [x_equinoctial] Keplerian Orbital Elements to Equinoctial State Vector 
                    function x0_equinoctial = keplerianToEquinoctial(a, e, i, RAAN, omega, nu)
                        % Convert Keplerian elements to Equinoctial elements
                        %
                        % Parameters:
                        %   a     - Semi-major axis
                        %   e     - Eccentricity
                        %   i     - Inclination (in rads)
                        %   RAAN  - Right Ascension of the Ascending Node (in rads)
                        %   omega - Argument of Periapsis (in rads)
                        %   nu    - True Anomaly (in rads)
                        %
                        % Returns:
                        %   x0_equinoctial - The state vector in Equinoctial elements [p, f, g, h, k, L]
                        
                        % Compute the semilatus rectum
                        p = a * (1 - e^2);
                        
                        % Compute the Equinoctial elements f and g (related to eccentricity)
                        f = e * cos(omega + RAAN);
                        g = e * sin(omega + RAAN);
                        
                        % Compute the Equinoctial elements h and k (related to inclination)
                        h = tan(i / 2) * cos(RAAN);
                        k = tan(i / 2) * sin(RAAN);
                        
                        % Compute the true longitude (L) in radians for Modified Equinoctial Elements
                        L = omega + RAAN + nu;
                        
                        % Construct the Equinoctial state vector
                        x0_equinoctial = [p; f; g; h; k; L];
                    end
 

end
    
end
