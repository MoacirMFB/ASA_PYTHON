%% Virtual Thrust Optimization (v1.1)
% Goal: maximize miss distance at CA by optimizing piecewise-constant virtual thrust
% This is a minimal version to test the optimization framework and proxy bounds.
% Decision variables are dimensionless normalized accelerations y_k, with A_k = amax * y_k.
% Constraints:
%   1) Total budget: Σ ||y_k|| Δt <= tau_budget, where tau_budget = ΔVmax/amax
%   2) Per-segment bound: ||y_k|| <= 1


clear; clc; close all;

%% Settings
reltol = 1e-9;
abstol = 1e-9;

%% Instantiate libraries
int = InterceptionLibrary();   
plt = PlotterLibrary();        
kep = KeplerianOrbitalMechanicsLibrary(); 

Earth = CelestialBody("Earth");         

%% User inputs: asteroid + impactor bounds
rhoAst = 2400;                 % [kg/m^3] asteroid bulk density
D_ast  = 100;                  % [m]     asteroid diameter
beta   = 1.0;                  % [-]     momentum enhancement factor

mA = (4/3)*pi*(D_ast/2)^3 * rhoAst;   % [kg] asteroid mass (sphere)

%% Derived bounds: per-impact Δv and cadence a_max
% dtMin = 24*3600;     % [s]
% cosg  = 1.0;         % [-]
% ms_list   = (1:5)'  * 1000;   % [kg]
% vrel_list = (5:10)' * 1000;   % [m/s]
% fprintf('\n=== Proxy bounds (beta=%.1f) ===\n', beta);
% fprintf('mA=%.3e kg | dtMin=%.1f hr | cos(g)=%.2f\n', mA, dtMin/3600, cosg);
% fprintf('m_s[kg]  v_rel[km/s]   dV_per[m/s]     a_max[m/s^2]\n');
% for i = 1:numel(ms_list)
%     for j = 1:numel(vrel_list)
%         ms   = ms_list(i);
%         vrel = vrel_list(j);
%         dVper = compute_dV_per_impact(mA, ms, vrel, beta, cosg);
%         amax  = compute_amax_from_cadence(mA, ms, vrel, beta, cosg, dtMin);
%         fprintf('%7.0f  %11.1f  %12.6e  %12.6e\n', ms, vrel/1000, dVper, amax);
%     end
% end
% 
% % Example: compute dVmax for N=20 impacts with max mass/speed
% N = 20;     % number of impacts
% dVmax = compute_dVmax_from_impactors(mA, ms_list(end)*ones(N,1), vrel_list(end)*ones(N,1), beta, cosg*ones(N,1));
% fprintf('\nN=%d, ms=%.0f kg, vrel=%.1f km/s -> dVmax=%.6e m/s\n', N, ms_list(end), vrel_list(end)/1000, dVmax);


%% Environment and trajectory propagation (forward)
env = int.makeEnv('years', 4, 'step_min', 10, 'odeOpt', odeset('RelTol',reltol,'AbsTol',abstol));  % Create environment 
[env.t_Earth, env.X_Earth_hist] = int.propagateEarth(env);  % propagate earth with environment settings

asteroid = int.astCatalog('Apophis');               % load asteroid
asteroid = int.propagateAsteroids(asteroid, env);   % propagate asteroid with env settings


%% Plot Earth and asteroid trajectories
% bodies = int.makeBodiesForPlot(asteroid, env);
% plt.plot_2BP_trajectories(bodies, "Sun", env.muSun_km, false, 'StatesProvided', true, 'ShowStartEnd', false, 'Title', "Earth + Asteroid");


%% Prepare impact/MBI initial conditions (backward setup)
forceImpact = true;                    % false for natural CA
leadTime   = 3;                        % look back up to X years  
monthsBack     = 0:1:12 * leadTime;    % list of MBIs to prepare
t0 = 24;                               % [months] before CA to set as initial time for optimization 

[asteroid, ICs_at_MBI_Earth_Ast] = int.prepareMBI(asteroid, env, monthsBack, forceImpact); % prepare MBI initial conditions for Earth and asteroid at specified months before CA

ICs = ICs_at_MBI_Earth_Ast{1};               % 1x58 struct with months and x0 for A and E
xA_t0 = ICs(find([ICs.month] == t0, 1, 'first')).stateAst(:);     % 6x1 state vector of asteroid at t0 months before CA (force col)
xE_t0 = ICs(find([ICs.month] == t0, 1, 'first')).stateEarth(:);   % 6x1 state vector of earth at t0 months before CA (force col)


%% Optimization problem setup (SQP, minimal v1)
% Goal: maximize heliocentric miss distance at the close-approach epoch (t_f = 0).
% Decision variables (piecewise-constant over M segments):
%   y_k ∈ R^3  [-] normalized acceleration on segment k (A_k = amax * y_k)
% Constraints:
%   ||y_k|| <= 1  (per-segment magnitude bound)
%   Σ ||y_k|| Δt <= ΔVmax/amax  (total normalized thrust budget)

% --- Choose proxy bounds from a representative impactor configuration ---
ms_use   = 1000;      % [kg] example impactor mass (5 tons)
Vrel_use = 10e3;      % [m/s] example relative impact speed (10 km/s)
N_imp    = 20;        % number of impactors (for ΔV budget)
cosg     = 1;   
dtDay    = 10;
dtMin = dtDay * 24 * 3600;      % [s]

amax_mps2  = compute_amax_from_cadence(mA, ms_use, Vrel_use, beta, cosg, dtMin);   % [m/s^2]
dVmax_mps  = compute_dVmax_from_impactors(mA, ms_use*ones(N_imp,1), Vrel_use*ones(N_imp,1), beta, cosg*ones(N_imp,1)); % [m/s]

amax_kmps2 = amax_mps2 / 1000;   % [km/s^2]
dVmax_kmps = dVmax_mps  / 1000;  % [km/s]
% Normalized thrust-time budget (seconds): tau_budget = ΔVmax/amax
% This is the maximum time you can apply ||y||=1 under the ΔV budget.
tau_budget = dVmax_kmps / amax_kmps2;   % [s]

% --- Time convention (v1) ---
% By construction, MBI states are defined relative to the close approach:
%   t_f = 0 at CA/impact epoch, and t0 is negative (months before CA).
sec_per_month = 30 * 86400;

tf_sec = 0.0;
t0_sec = -t0 * sec_per_month;

% Earth state at CA (month = 0 MBI). 
xE_tf = ICs(find([ICs.month] == 0, 1, 'first')).stateEarth(:);   % Earth state at CA (month=0) as col vec
if isempty(xE_tf)
    error('Earth state at CA not found: ICs does not contain month=0. Fix getStatesAtMBI dt==0 branch or include 0 in monthsBack.');
end

xA_0 = xA_t0(:);   % 6x1

% Fixed Earth state at tf (for v1 objective)
rE_tf = xE_tf(1:3);   % 3x1

% --- Discretization ---
M = 50;                              % number of control segments (small for v1)
T = tf_sec - t0_sec;                % [s] (positive time of flight)
dt_seg = T / M;
% Feasible warm start: ensure sum(||y_k||)*dt_seg <= tau_budget
% If we use constant ||y|| = y0_mag over the horizon, tau_used ≈ y0_mag*T.
y0_mag_max = tau_budget / T;

% --- Initial guess for normalized acceleration controls 
% Small tangential acceleration as a warm start (dimensionless)
vhat0 = xA_0(4:6)/norm(xA_0(4:6));
y0_mag = 0.5 * min(0.01, y0_mag_max);   % 50% margin inside feasibility
Y0 = repmat(y0_mag * vhat0.', M, 1);     % [M x 3]
z0 = Y0.'; z0 = z0(:);                  % stack as [3M x 1]

% Component-wise bounds for y (tighter for stability); norm bound enforced in nonlinear constraints
lb = -1.2 * ones(3*M,1);
ub =  1.2 * ones(3*M,1);

% --- Solve with interior-point (feasibility mode) ---
opts = optimoptions('fmincon', ...
    'Algorithm','interior-point', ...
    'Display','iter', ...
    'MaxIterations',400, ...
    'MaxFunctionEvaluations',2e5, ...
    'StepTolerance',1e-10, ...
    'OptimalityTolerance',1e-6, ...
    'FiniteDifferenceStepSize', 1e-4, ...
    'ConstraintTolerance', 1e-6, ...
    'EnableFeasibilityMode', true, ...
    'SubproblemAlgorithm','cg', ...
    'FiniteDifferenceType','central', ...
    'HessianApproximation','lbfgs', ...
    'OutputFcn', []);   % keep disabled: avoids re-propagating inside OutputFcn

muSun = env.muSun_km;   % [km^3/s^2]
odeOpt = odeset('RelTol', reltol, 'AbsTol', abstol);

obj = @(z) obj_miss_distance_scaled(z, xA_0, rE_tf, t0_sec, tf_sec, M, amax_kmps2, muSun, kep, odeOpt);
nonlcon = @(z) nlcon_budget_and_yk(z, M, dt_seg, dVmax_kmps/amax_kmps2);

[zopt, fopt, exitflag, out] = fmincon(obj, z0, [], [], [], [], lb, ub, nonlcon, opts);

fprintf('\nSQP done. exitflag=%d | fopt=%.6e\n', exitflag, fopt);

% Decode solution and report achieved miss
Y_opt = decode_accel(zopt, M);  % [M x 3] (dimensionless)
A_opt = amax_kmps2 * Y_opt;     % [km/s^2]
xA_tf_opt = propagate_piecewise_accel(xA_0, t0_sec, tf_sec, A_opt, muSun, kep, odeOpt);
R_tf = xA_tf_opt(1:3) - rE_tf;
fprintf('Achieved |R(tf)| = %.6f km\n', norm(R_tf));

%% Local functions

function Y = decode_accel(z, M)
% decode_accel  Unpack decision vector into per-segment normalized acceleration vectors
% Output Y is [M x 3], dimensionless
    Y = reshape(z, 3, M).';
end


function J = obj_miss_distance_scaled(z, x0, rE_tf, t0, tf, M, amax, mu, kep, odeOpt)
% obj_miss_distance_scaled  Minimize negative squared miss distance (scaled controls)
% Decision variables z store y_k (dimensionless), where A_k = amax * y_k.

    Y = decode_accel(z, M);          % [M x 3] dimensionless
    A = amax * Y;                   % [M x 3] km/s^2

    xf = propagate_piecewise_accel(x0, t0, tf, A, mu, kep, odeOpt);

    R = xf(1:3) - rE_tf(:);
    J = -0.5 * dot(R, R);
end


function [c, ceq] = nlcon_budget_and_yk(z, M, dt_seg, tau_budget)
% nlcon_budget_and_yk  Constraints in terms of y_k (dimensionless)
%   1) Total budget: Σ ||y_k|| Δt <= tau_budget, where tau_budget = ΔVmax/amax
%   2) Per-segment bound: ||y_k|| <= 1

    Y = decode_accel(z, M);          % [M x 3] dimensionless
    ynorm = vecnorm(Y, 2, 2);        % [M x 1]

    tau_used = sum(ynorm) * dt_seg;  % [s]
    c_budget = tau_used - tau_budget;

    c_seg = ynorm - 1.0;

    c = [c_budget; c_seg];
    ceq = [];
end


function stop = outfun_print_miss(z, optimValues, state)
% outfun_print_miss  Output function for fmincon to print miss distance during optimization (normalized controls)
%
% Signature matches fmincon OutputFcn: (x, optimValues, state) -> stop (bool)
%
% Uses a persistent struct P to store constant parameters for efficiency
% Prints miss distance and objective function value at initialization and each iteration
%
% Note: re-propagates asteroid trajectory each call, which can be computationally expensive

    stop = false;

    persistent P

    switch state
        case 'init'
            % Retrieve parameters from base workspace once at start of optimization
            % This avoids passing many parameters through nested functions
            P.xA0   = evalin('base','xA_0');
            P.rE_tf = evalin('base','rE_tf');
            P.t0    = evalin('base','t0_sec');
            P.tf    = evalin('base','tf_sec');
            P.M     = evalin('base','M');
            P.mu    = evalin('base','muSun');
            P.kep   = evalin('base','kep');
            P.odeOpt= evalin('base','odeOpt');
            P.dt_seg = evalin('base','dt_seg');
            P.amax   = evalin('base','amax_kmps2');
            P.dVmax  = evalin('base','dVmax_kmps');
            P.tau_budget = evalin('base','dVmax_kmps/amax_kmps2');

            % Decode controls and propagate asteroid trajectory
            Y = decode_accel(z, P.M);
            A = P.amax * Y;
            xf = propagate_piecewise_accel(P.xA0, P.t0, P.tf, A, P.mu, P.kep, P.odeOpt);

            % Compute miss distance at final time
            R = xf(1:3) - P.rE_tf(:);
            miss_km = norm(R);

            [c, ~] = nlcon_budget_and_yk(z, P.M, P.dt_seg, P.tau_budget);
            maxViol = max(c);

            tau_used = sum(vecnorm(Y,2,2)) * P.dt_seg;

            fprintf('\n[init] miss = %.6f km | f = %.6e | max(c)=%.3e | tau=%.3e/%.3e s\n', miss_km, optimValues.fval, maxViol, tau_used, P.tau_budget);

        case 'iter'
            % At each iteration, re-propagate and print miss distance and objective value
            Y = decode_accel(z, P.M);
            A = P.amax * Y;
            xf = propagate_piecewise_accel(P.xA0, P.t0, P.tf, A, P.mu, P.kep, P.odeOpt);
            R = xf(1:3) - P.rE_tf(:);
            miss_km = norm(R);

            [c, ~] = nlcon_budget_and_yk(z, P.M, P.dt_seg, P.tau_budget);
            maxViol = max(c);

            tau_used = sum(vecnorm(Y,2,2)) * P.dt_seg;

            fprintf('[iter %4d] miss = %.6f km | f = %.6e | max(c)=%.3e | tau=%.3e/%.3e s\n', optimValues.iteration, miss_km, optimValues.fval, maxViol, tau_used, P.tau_budget);

        case 'done'
            % No action needed on completion
    end
end


function xf = propagate_piecewise_accel(x0, t0, tf, A, mu, kep, odeOpt)
% propagate_piecewise_accel  Single ODE call with piecewise-constant accel lookup
% x0 [6x1] in km, km/s. A is [M x 3] in km/s^2.
%
% The control is constant on each segment k:
%   t in [t0+(k-1)dt, t0+k dt) -> a_ctrl = A(k,:)

    M = size(A,1);
    T = tf - t0;
    dt_seg = T / M;

    % Integrate forward from t0 (negative) to tf (=0)
    tspan = [t0 tf];

    odefun = @(tt, xx) dyn_controlled_2bp_piecewise(tt, xx, t0, dt_seg, A, mu, kep);
    [~, X] = ode113(odefun, tspan, x0, odeOpt);

    xf = X(end,:).';
end


function xdot = dyn_controlled_2bp(t, x, a_ctrl, mu, kep)
% dyn_controlled_2bp  Dynamics of two-body problem with added control acceleration
%
% Wraps kep.dynamics_2BP_cartesian to compute gravitational acceleration,
% then adds constant control acceleration vector to velocity derivatives.
%
% Inputs:
%   t      - current time (unused here)
%   x      - state vector [6x1], [position; velocity] in km and km/s
%   a_ctrl - control acceleration vector [3x1] in km/s^2
%   mu     - gravitational parameter [km^3/s^2]
%   kep    - KeplerianOrbitalMechanicsLibrary object
%
% Output:
%   xdot   - time derivative of state vector [6x1]

    % Compute gravitational dynamics from Keplerian library
    xdot = kep.dynamics_2BP_cartesian(t, x, mu);

    % Add control acceleration to velocity derivative components
    xdot(4:6) = xdot(4:6) + a_ctrl;
end

function dVmax = compute_dVmax_from_impactors(mA, ms, vrel, beta, cosgamma)
% compute_dVmax_from_impactors
%   ΔV_max [m/s] upper bound on asteroid ΔV from N impacts
%   Formula: ΔV_max ≈ (β/mA) * Σ (m_s * v_rel * cosγ)
%   Inputs:
%     mA       [kg]    asteroid mass (scalar)
%     ms       [kg]    impactor masses (Nx1 or 1xN)
%     vrel     [m/s]   relative impact speeds (same size as ms)
%     beta     [-]     momentum enhancement factor (default 1)
%     cosgamma [-]     alignment cosine(s) (default ones)
%   Output:
%     dVmax    [m/s]   total achievable ΔV upper bound (nonnegative)

    % Default beta to 1.0 if not provided or empty
    if nargin < 4 || isempty(beta)
        beta = 1.0;
    end
    % Default cosgamma to vector of ones if not provided or empty
    if nargin < 5 || isempty(cosgamma)
        cosgamma = ones(size(ms));
    end

    ms = ms(:);         
    vrel = vrel(:);
    cosgamma = cosgamma(:);

    % Compute total ΔV as sum of individual momentum contributions scaled by beta/mA
    dVmax = (beta / mA) * sum(ms .* vrel .* cosgamma);

    % Clamp to nonnegative values to avoid negative ΔV due to numerical issues
    dVmax = max(dVmax, 0.0);
end


function dV_per = compute_dV_per_impact(mA, ms, vrel, beta, cosgamma)
% compute_dV_per_impact
%   Δv_per [m/s] upper bound on asteroid ΔV from one impact
%   Formula: Δv_per ≈ β*(m_s/mA)*v_rel*cosγ
%   Inputs:
%     mA       [kg]    asteroid mass (scalar)
%     ms       [kg]    impactor mass (scalar)
%     vrel     [m/s]   relative impact speed (scalar)
%     beta     [-]     momentum enhancement factor (default 1)
%     cosgamma [-]     alignment cosine (default 1)
%   Output:
%     dV_per   [m/s]   per-impact ΔV upper bound (nonnegative)

    % Default beta to 1.0 if not provided or empty
    if nargin < 4 || isempty(beta)
        beta = 1.0;
    end
    % Default cosgamma to 1.0 if not provided or empty
    if nargin < 5 || isempty(cosgamma)
        cosgamma = 1.0;
    end

    % Compute per-impact ΔV using momentum transfer formula
    dV_per = beta * (ms / mA) * vrel * cosgamma;

    % Clamp to nonnegative values
    dV_per = max(dV_per, 0.0);
end


function amax = compute_amax_from_cadence(mA, ms, vrel, beta, cosgamma, dt_min)
% compute_amax_from_cadence
%   a_max [m/s^2] proxy max acceleration from impact cadence
%   Formula: a_max ≈ Δv_per / dt_min
%   Inputs:
%     mA       [kg]    asteroid mass (scalar)
%     ms       [kg]    impactor mass (scalar)
%     vrel     [m/s]   relative impact speed (scalar)
%     beta     [-]     momentum enhancement factor (default 1)
%     cosgamma [-]     alignment cosine (default 1)
%     dt_min   [s]     minimum time between impacts
%   Output:
%     amax     [m/s^2] cadence-based acceleration proxy

    % Default beta to 1.0 if not provided or empty
    if nargin < 4 || isempty(beta)
        beta = 1.0;
    end
    % Default cosgamma to 1.0 if not provided or empty
    if nargin < 5 || isempty(cosgamma)
        cosgamma = 1.0;
    end

    % Compute per-impact ΔV
    dV_per = compute_dV_per_impact(mA, ms, vrel, beta, cosgamma);

    % Approximate max acceleration as ΔV per minimum time between impacts
    amax = dV_per / dt_min;
end
function xdot = dyn_controlled_2bp_piecewise(t, x, t0, dt_seg, A, mu, kep)
% dyn_controlled_2bp_piecewise  2BP + piecewise-constant acceleration lookup
%
% Segment index:
%   k = floor((t - t0)/dt_seg) + 1, clamped to [1, M]

    M = size(A,1);
    k = floor((t - t0) / dt_seg) + 1;
    if k < 1
        k = 1;
    elseif k > M
        k = M;
    end

    a_ctrl = A(k,:).';

    xdot = kep.dynamics_2BP_cartesian(t, x, mu);
    xdot(4:6) = xdot(4:6) + a_ctrl;
end