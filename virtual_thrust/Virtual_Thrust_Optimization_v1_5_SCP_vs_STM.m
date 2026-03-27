%% Virtual Thrust Optimization (v1.3) — File Overview
% -------------------------------------------------------------------------
% PURPOSE
%   Optimize a continuous “virtual thrust” (control acceleration) profile to
%   maximize the heliocentric miss distance between an asteroid and Earth at
%   the close-approach (CA) epoch.

clear; clc; close all;

cvx_clear

%% User settings: execution / numerics
reltol = 1e-12;
abstol = 1e-12;

RUN_SCP = true;           % true = run SCP section, false = skip SCP and only run STM benchmark
USE_OPT_DV_DIR = true;   % true = use Conway optimal dV direction for SCP warm start, false = use tangential warm start


cvx_precision default
cvx_solver mosek

%% User settings: scenario / asteroid
asteroid_name = 'Apophis';
rhoAst = 2400;                 % [kg/m^3] asteroid bulk density
D_ast  = 100;                  % [m] asteroid diameter
beta   = 1.0;                  % [-] momentum enhancement factor
forceImpact = true;            % false for natural CA
leadTime = 3;                  % [years] look back up to this many years
monthsBack = 0:1:12*leadTime;  % list of MB trahiIs to prepare
t0 = 24;                       % [months] before CA to set as initial time for optimization

%% User settings: interceptor / optimization
mass_sc  = 1000;      % [kg] example impactor mass
Vrel_use = 10e3;      % [m/s] example relative impact speed (10 km/s)
N_imp    = 5;         % number of impactors (for ΔV budget)
cosg     = 1;         % cosγ alignment factor (1 = max aligned, 0 = perpendicular, -1 = anti-aligned)
dtDay    = 10;        % made-up min cadence of impacts in days
N = 750;              % number of ZOH intervals for SCP
Kmax = 50;            % max SCP outer iterations
Delta_u0 = 0.75;      % initial trust-region radius per interval
eta_good  = 0.25;     % accept step if rho >= eta_good and miss improves
eta_great = 0.75;     % expand trust region if rho >= eta_great
shrink    = 0.5;      % trust-region shrink factor when rejecting
expand    = 1.5;      % trust-region expansion factor when very successful

%% Instantiate libraries
int = InterceptionLibrary();   
plt = PlotterLibrary();        
kep = KeplerianOrbitalMechanicsLibrary(); 
aca = AppliedControlAstronauticsLibrary();

Earth = CelestialBody("Earth");   
RE_km = Earth.radius.km;
Sun = CelestialBody("Sun");

muSun_km = Sun.mu.km;

%% User inputs: asteroid + impactor bounds

mA = (4/3)*pi*(D_ast/2)^3 * rhoAst;   % [kg] asteroid mass (sphere)


%% Environment and trajectory propagation (forward)
env = int.makeEnv('years', 4, 'step_min', 10, 'odeOpt', odeset('RelTol',reltol,'AbsTol',abstol));  % Create environment 
[env.t_Earth, env.X_Earth_hist] = int.propagateEarth(env);  % propagate earth with environment settings

muSun = env.muSun_km;   % [km^3/s^2]

asteroid = int.astCatalog(asteroid_name);               % load asteroid
asteroid = int.propagateAsteroids(asteroid, env);   % propagate asteroid with env settings

odeOpt = odeset('RelTol', reltol, 'AbsTol', abstol);

%% Plot Earth and asteroid trajectories
bodies = int.makeBodiesForPlot(asteroid, env);
plt.plot_2BP_trajectories(bodies, "Sun", env.muSun_km, false, 'StatesProvided', true, 'ShowStartEnd', true, 'Title', "Earth + Asteroid");
hFig_orbit = gcf;

%% Prepare impact/MBI initial conditions (backward setup)

[asteroid, ICs_at_MBI_Earth_Ast] = int.prepareMBI(asteroid, env, monthsBack, forceImpact); % prepare MBI initial conditions for Earth and asteroid at specified months before CA

ICs = ICs_at_MBI_Earth_Ast{1};               % 1x58 struct with months and x0 for A and E
xA_t0 = ICs(find([ICs.month] == t0, 1, 'first')).stateAst(:);     % 6x1 state vector of asteroid at t0 months before CA (force col)
xE_t0 = ICs(find([ICs.month] == t0, 1, 'first')).stateEarth(:);   % 6x1 state vector of earth at t0 months before CA (force col)


%% Optimization problem setup (SCP + CVX)

% --- Choose proxy bounds from a representative impactor configuration ---
dtMin = dtDay * 24 * 3600;      % [s] impact cadence in seconds

% [m/s^2] proxy max acceleration from impact cadence
amax_mps2  = compute_amax_from_cadence(mA, mass_sc, Vrel_use, beta, cosg, dtMin);   
amax_kmps2 = amax_mps2 / 1000;   % [km/s^2]

% [m/s] max ΔV from N_imp impactors with given properties
dVmax_mps  = compute_dVmax_from_impactors(mA, mass_sc*ones(N_imp,1), Vrel_use*ones(N_imp,1), beta, cosg*ones(N_imp,1)); 
dVmax_kmps = dVmax_mps  / 1000;  % [km/s]

% Single-interceptor impulsive ΔV benchmark using the same representative interceptor
% mass/speed already defined above.
dV1_mps  = compute_dV_per_impact(mA, mass_sc, Vrel_use, beta, cosg);   % per-impact ΔV upper bound [m/s] 
dV1_kmps = dV1_mps / 1000;                                            % per-impact ΔV upper bound [km/s] 

% Normalized thrust-time budget (seconds): tau_budget = ΔVmax/amax
% This is the maximum time you can apply ||y||=1 under the ΔV budget.
tau_budget = dVmax_kmps / amax_kmps2;   % [s]

% --- Time convention (v1) ---
% By construction, MBI states are defined relative to the close approach:
%   t_f = 0 at CA/impact epoch, and t0 is negative (months before CA).
sec_per_month = 30 * 86400;

tf_sec = 0.0;                       % final time is time zero (CA)
t0_sec = -t0 * sec_per_month;       % initial time is user defined 

% Earth state at CA (month = 0 MBI). 
xE_tf = ICs(find([ICs.month] == 0, 1, 'first')).stateEarth(:);   % Earth state at CA (month=0) as col vec
xA_0 = xA_t0(:);   % 6x1

% Fixed Earth state at tf (for v1 objective)
rE_tf = xE_tf(1:3);   % 3x1

% Rebuild the asteroid's initial Cartesian state from the catalog COEs so
% it can be printed alongside the benchmark STM terms for parity checks.
coe_debug = asteroid{1}.coe;
coe_debug = [coe_debug(1) * env.AU2km, coe_debug(2:6)];
X0_debug = kep.coe_to_cartesian(coe_debug, muSun_km, 'useTA', true);

%% Conway max-theoretical STM benchmark (always runs)
bench_conway_single = conway_max_theoretical_deflection_stm(xA_0, rE_tf, dV1_kmps, tf_sec, t0_sec, kep, muSun_km, odeOpt);

fprintf('\nConway max theoretical deflection (single interceptor) = %.6f km\n', ...
    bench_conway_single.dr_max_km);
print_conway_diagnostics(asteroid_name, X0_debug, xA_0, bench_conway_single, t0);

% --- Discretization (ZOH control for SCP) ---
T = tf_sec - t0_sec;                 % total time span from t0 to tf (s)
dt_seg = T / N;                      % duration of each ZOH segment (s)

disp("Interval duration in hours is: ")
dt_seg / 3600

% Time grid for discrete states
% x_k corresponds to time t_k, with k=1..N+1 
% u_k is held constant on [t_k, t_{k+1})
t_grid = linspace(t0_sec, tf_sec, N+1).';   % initialize time grid (s)

% Feasible warm start: ensure sum(||u_j||)*dt_seg <= tau_budget
% If we use constant ||u|| = u0_mag over the horizon, tau_used ≈ u0_mag*T.
u0_mag_max = tau_budget / T;    % max constant control magnitude that meets the budget if applied over the whole horizon

% --- Initial guess for normalized acceleration controls 
% Small constant-direction acceleration as a warm start (dimensionless)
u0_mag = 0.5 * min(0.01, u0_mag_max);   % initial guess control magnitude (dimensionless), scaled down to 50% from max

if USE_OPT_DV_DIR
    uhat0 = bench_conway_single.e_opt(:) / norm(bench_conway_single.e_opt(:));   % Conway optimal impulsive dV direction
else
    uhat0 = xA_0(4:6) / norm(xA_0(4:6));                                          % tangential warm start
end

U0 = repmat(u0_mag * uhat0.', N, 1);    % initial guess for ZOH controls (N x 3)

% Component-wise bounds for u (tighter for stability); norm bound enforced in nonlinear constraints
lb = -1.2 * ones(N,3);  
ub =  1.2 * ones(N,3);




%% SCP
if RUN_SCP

%% SCP loop (CVX with discrete linearized dynamics)
% ZOH control u_k on each interval [t_k, t_{k+1}).
% At each SCP iteration, linearize/discretize dynamics about the nominal
% (x_k^nom, u_k^nom) to build discrete constraints:
%   x_{k+1} = A_k x_k + B_k u_k + c_k

% --- SCP settings ---
Delta_u = Delta_u0;      % trust-region radius per interval (bounds ||u_k - u_k^nom||)

% Nominal interval controls
U_nom = U0;              % nominal ZOH control guess

fprintf('\n=== SCP start (discrete dynamics) ===\n');

% Dimensions
szx = 6;                 % state dimension [r;v]
szu = 3;                 % control dimension

% Control injection matrix for acceleration (dimensionless u -> km/s^2)
O3 = zeros(3,3); I3 = eye(3);
Bctrl = [O3; I3] * amax_kmps2;    % maps u -> accel in vdot

for it = 1:Kmax

    % ----- Nonlinear rollout of nominal (ZOH) -----
    [X_nom] = rollout_zoh_2bp_control(xA_0, t_grid, U_nom, muSun, kep, amax_kmps2, odeOpt);  % [(N+1)x6]
    x_tf_nom = X_nom(end,:).';        % terminal state under nominal controls
    R_nom = x_tf_nom(1:3) - rE_tf;    % terminal miss vector (km)
    miss_nom = norm(R_nom);           % nominal miss distance (km)

    % ----- Build discrete linearized dynamics (A_k, B_k, c_k) -----
    Ak = zeros(szx, szx, N);          % discrete A_k matrices
    Bk = zeros(szx, szu, N);          % discrete B_k matrices
    ck = zeros(szx, N);               % discrete c_k vectors

    % Loop over intervals to compute (Ak, Bk, ck) for linearized discrete dynamics:
    for k = 1:N
        xk_nom = X_nom(k,:).';                % nominal state at step k
        uk_nom = U_nom(k,:).';                % nominal control on interval k

        % Continuous-time Jacobian A(x) for 2BP (control-independent)
        Acont = kep.jacobian_2BP_cartesian(0, xk_nom, muSun);    % continuous ∂f/∂x at (xk_nom)

        % Continuous-time affine term c so that: xdot = Acont*x + Bctrl*u + c
        f0 = kep.dynamics_2BP_cartesian(0, xk_nom, muSun);       % continuous dynamics f(x,0)
        f0(4:6) = f0(4:6) + amax_kmps2 * uk_nom;                 % add control accel to vdot
        ccont = f0 - Acont*xk_nom - Bctrl*uk_nom;                % affine term so f = A x + B u + c at nominal

        % Discretize over [t_k, t_{k+1}] using ACA helper (augmented integration)
        tk   = t_grid(k);                                        % interval start time (s)
        tk1  = t_grid(k+1);                                      % interval end time (s)

        % Augmented IC: [x; vec(I); vec(0); vec(0)]
        Phi0 = eye(szx);                                         % STM IC for discretization
        Y0aug = [xk_nom; Phi0(:); zeros(szx*szu,1); zeros(szx,1)]; % aug IC: state, STM, B-map, c-map

        [Ak(:,:,k), Bk(:,:,k), ck(:,k), ~] = aca.linearDiscreteTimeMatrices(tk, tk1, Y0aug, Acont, Bctrl, ccont, uk_nom, szx, szu, odeOpt); % returns (Ak,Bk,ck) for x_{k+1}=Ak x_k + Bk u_k + ck
    end

    % ----- Solve convex subproblem in CVX -----
    % Linear objective: maximize first-order increase in 1/2||R||^2 ~ R_nom^T * delta_r
    rnom_tf = x_tf_nom(1:3);
    
    cvx_begin quiet
        variables x(N+1, szx) u(N, szu)       % decision vars: state and ZOH control
        expressions tau_used

        tau_used = sum( norms(u,2,2) ) * dt_seg;   % approx ∫||u|| dt with ZOH

        maximize( R_nom' * (x(N+1,1:3)' - rnom_tf) ) % linearized objective about nominal terminal r

        subject to
            % initial condition
            x(1,:)' == xA_0;                     % initial condition

            % discrete linearized dynamics
            for k = 1:N
                x(k+1,:)' == Ak(:,:,k) * x(k,:)' + Bk(:,:,k) * u(k,:)' + ck(:,k); % linearized discrete dynamics constraint
            end

            % control bounds
            norms(u,2,2) <= 1;                   % per-interval magnitude bound

            % budget
            tau_used <= tau_budget;              % total budget bound

            % trust region
            norms(u - U_nom, 2, 2) <= Delta_u;   % trust region on control update

    cvx_end

    U_new = u;  % candidate control from convex subproblem

    % ----- Nonlinear evaluation of candidate -----
    X_new = rollout_zoh_2bp_control(xA_0, t_grid, U_new, muSun, kep, amax_kmps2, odeOpt);   % nonlinear rollout of candidate control
    x_tf_new = X_new(end,:).';         % terminal state under candidate controls
    R_new = x_tf_new(1:3) - rE_tf;     % candidate miss vector
    miss_new = norm(R_new);            % candidate miss distance

    % Predicted improvement proxy from convex objective
    pred = R_nom' * (x(N+1,1:3)' - rnom_tf); % predicted (linear) improvement proxy
    act  = 0.5*(miss_new^2 - miss_nom^2);    % actual improvement in 1/2||R||^2
    rho  = act / max(pred, 1e-12);           % trust-region ratio (actual/predicted)

    fprintf('SCP %2d | miss_nom=%.3f km -> miss_new=%.3f km | rho=%.3f | Delta_u=%.3f | cvx=%s\n', ...
        it, miss_nom, miss_new, rho, Delta_u, cvx_status);

    % Trust-region accept/reject
    if rho > eta_good && miss_new >= miss_nom       % accept if improvement is good and actual miss is better than nominal
        U_nom = U_new;                              % update nominal control to candidate
        if rho > eta_great                          % expand trust region if very successful
            Delta_u = min(1.0, expand*Delta_u);     % cap max trust region to 1.0 (which is the hard control bound)
        end
    else
        Delta_u = max(1e-3, shrink*Delta_u);        % reject and shrink trust region, but keep nominal control unchanged
    end

    if Delta_u < 5e-3                            % stop if trust region is too small
        break;
    end

end

% Final nonlinear rollout
Uopt = U_nom;   % final optimized ZOH control
X_opt = rollout_zoh_2bp_control(xA_0, t_grid, Uopt, muSun, kep, amax_kmps2, odeOpt);
xA_tf_opt = X_opt(end,:).';
R_tf = xA_tf_opt(1:3) - rE_tf;
fprintf('\nSCP done. Achieved |R(tf)| = %.6f km\n', norm(R_tf));

%% Overlay optimized acceleration profile on the asteroid orbit
% Plot acceleration vectors along the optimized asteroid trajectory on top of
% the existing Earth+asteroid orbit figure.
figure(hFig_orbit);
ax_orbit = gca;
hold(ax_orbit, 'on');

% Use state samples at the ZOH grid points and the corresponding interval controls.
R_nodes = X_opt(1:end-1, 1:3);              % [N x 3] asteroid positions at interval starts
A_nodes_plot = amax_kmps2 * Uopt;           % [N x 3] physical acceleration [km/s^2]
A_norm_plot = vecnorm(A_nodes_plot, 2, 2);  % [N x 1]
t_nodes_months = t_grid(1:end-1) / sec_per_month;

% Downsample nonzero-control arrows to keep the plot readable.
idx_nonzero = find(A_norm_plot > 0);
idx_plot = idx_nonzero(1:8:end);

Rq = R_nodes(idx_plot, :);
Aq = A_nodes_plot(idx_plot, :);
Aq_norm = A_norm_plot(idx_plot);
tq_months = t_nodes_months(idx_plot);

if ~isempty(Aq_norm)
    % Scale arrow lengths relative to orbit size while preserving magnitude ratios.
    r_scale = max(vecnorm(R_nodes, 2, 2));
    a_scale = max(Aq_norm);
    arrow_scale = 0.16 * r_scale / a_scale;
    Aq_vis = arrow_scale * Aq;

    % Color arrows by time.
    cmap = turbo(256);
    tmin = min(t_nodes_months);
    tmax = max(t_nodes_months);
    if abs(tmax - tmin) < eps
        cidx = ones(size(tq_months));
    else
        cidx = 1 + round((size(cmap,1)-1) * (tq_months - tmin) / (tmax - tmin));
    end
    cidx = max(1, min(size(cmap,1), cidx));

    for i = 1:numel(cidx)
        quiver3(ax_orbit, ...
            Rq(i,1), Rq(i,2), Rq(i,3), ...
            Aq_vis(i,1), Aq_vis(i,2), Aq_vis(i,3), ...
            0, ...
            'Color', cmap(cidx(i),:), ...
            'LineWidth', 1.8, ...
            'MaxHeadSize', 1.6, ...
            'HandleVisibility', 'off');
    end

    % Mark sampled locations using the same time colormap.
    scatter3(ax_orbit, Rq(:,1), Rq(:,2), Rq(:,3), 18, tq_months, 'filled', ...
        'DisplayName', 'Control samples');

    colormap(ax_orbit, cmap);
    cb = colorbar(ax_orbit);
    cb.Label.String = 'Time [months] (CA at 0)';

    title(ax_orbit, 'Earth + Asteroid + Optimized acceleration profile');
    legend(ax_orbit, 'show');
end

%% Plot optimized control input (ZOH intervals)
Aopt = amax_kmps2 * Uopt;     % [N x 3]
Aopt_mmps2 = 1e6 * Aopt;      % [N x 3] convert km/s^2 -> mm/s^2

% Compute accumulated ΔV from optimized ZOH acceleration
Aopt_norm_kmps2 = vecnorm(Aopt, 2, 2);                 % [N x 1] ||a|| in km/s^2
DV_cum_kmps = [0; cumsum(Aopt_norm_kmps2) * dt_seg];    % [N+1 x 1] accumulated ΔV in km/s
DV_cum_mps  = 1000 * DV_cum_kmps;                       % [N+1 x 1] accumulated ΔV in m/s
DVmax_mps_plot = dVmax_mps;                              % [m/s] max allowable ΔV

% Time axes for stairs (months)
t_months = t_grid / sec_per_month;      % [N+1]

% Build stair vectors (repeat last control for plotting)
U_stair = [Uopt; Uopt(end,:)];
A_stair = [Aopt_mmps2; Aopt_mmps2(end,:)];   % mm/s^2

figure('Name','Optimized control input (SCP discrete)','Color','w');
subplot(3,1,1);
stairs(t_months, U_stair(:,1), 'LineWidth', 1.1); hold on;
stairs(t_months, U_stair(:,2), 'LineWidth', 1.1);
stairs(t_months, U_stair(:,3), 'LineWidth', 1.1);
stairs(t_months, vecnorm(U_stair,2,2), '--', 'Color', [0 0 0], 'LineWidth', 2.2);
yline(1.0, ':', 'Color', [0.85 0.1 0.1], 'LineWidth', 2.0);
xlabel('Time [months] (CA at 0)');
ylabel('u_k [-]');
title('Normalized ZOH control u_k');
grid on;
legend('u_x','u_y','u_z','||u||','||u||=1','Location','best');

subplot(3,1,2);
stairs(t_months, A_stair(:,1)); hold on;
stairs(t_months, A_stair(:,2));
stairs(t_months, A_stair(:,3));
stairs(t_months, vecnorm(A_stair,2,2), '--');
yline(1e6*amax_kmps2, ':');
xlabel('Time [months] (CA at 0)');
ylabel('a_k [mm/s^2]');
title('Physical ZOH acceleration a_k = a_{max} u_k (mm/s^2)');
grid on;
legend('a_x','a_y','a_z','||a||','||a||=a_{max}','Location','best');

subplot(3,1,3);
stairs(t_months, DV_cum_mps, 'LineWidth', 1.2); hold on;
yline(DVmax_mps_plot, ':', 'LineWidth', 1.2);
xlabel('Time [months] (CA at 0)');
ylabel('\Delta V_{cum} [m/s]');
title('Accumulated \Delta V over time (ZOH)');
grid on;
legend('\Delta V_{cum}(t)', '\Delta V_{max}', 'Location', 'best');

else
    fprintf('\nRUN_SCP = false: skipping SCP optimization and SCP plots.\n');
end




%% Local functions
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

function Xhist = rollout_zoh_2bp_control(x0, t_grid, U, mu, kep, amax, odeOpt)
% rollout_zoh_2bp_control  Rollout nonlinear 2BP with ZOH control on each interval.
% Inputs:
%   x0     [6x1]
%   t_grid [N+1 x 1]
%   U      [N x 3] (dimensionless)
% Output:
%   Xhist  [N+1 x 6] states at each grid time

    N = size(U,1);
    Xhist = zeros(N+1, 6);
    Xhist(1,:) = x0(:).';

    xk = x0(:);
    for k = 1:N
        tk  = t_grid(k);
        tk1 = t_grid(k+1);
        uk  = U(k,:).';

        odefun = @(t,x) dyn_2bp_zoh(t, x, uk, amax, mu, kep);
        [~, Xseg] = ode45(odefun, [tk tk1], xk, odeOpt);
        xk = Xseg(end,:).';
        Xhist(k+1,:) = xk.';
    end
end

function xdot = dyn_2bp_zoh(t, x, uk, amax, mu, kep)
% dyn_2bp_zoh  2BP + constant accel amax*uk over an interval.
    xdot = kep.dynamics_2BP_cartesian(t, x, mu);
    xdot(4:6) = xdot(4:6) + amax * uk;
end

function bench = conway_max_theoretical_deflection_stm(xA_0, rE_tf, dVmax_kmps, tf_sec, t0_sec, kep, muSun_km, odeOpt)
% conway_max_theoretical_deflection_stm
%   Compute Conway's maximum theoretical close-approach deflection for a
%   fixed impulsive ΔV magnitude using the 2BP state transition matrix.
%
%   The perturbation model is linearized about the nominal asteroid arc from
%   the intervention epoch t0 to the close-approach epoch tf. For a small
%   impulsive velocity change applied at t0,
%
%       delta_r(tf) = R * delta_v0
%
%   where R is the position-from-velocity STM block. For a fixed
%   ||delta_v0|| = dVmax_kmps, the maximum achievable linearized deflection
%   magnitude is obtained from the dominant eigenpair of R'R:
%
%       max ||delta_r(tf)|| = sqrt(lambda_max(R'R)) * dVmax_kmps
%
%   Inputs:
%     xA_0        [6x1] asteroid state at 
%     rE_tf       [3x1] Earth position at tf (used only for optional sign check)
%     dVmax_kmps  [scalar] fixed impulsive ΔV magnitude [km/s]
%     tf_sec      [scalar] final time in the local convention (CA at 0)
%     t0_sec      [scalar] initial time in the local convention
%     kep         library object with propagateSTM_2BP(...)
%     muSun_km    [scalar] solar gravitational parameter [km^3/s^2]
%     odeOpt      ODE options used for STM propagation
%
%   Output struct fields:
%     bench.Phi_tf              full 6x6 STM from t0 to tf
%     bench.Phi_rr              3x3 position-w.r.t.-position block
%     bench.Phi_rv              3x3 position-w.r.t.-velocity block
%     bench.Phi_vr              3x3 velocity-w.r.t.-position block
%     bench.Phi_vv              3x3 velocity-w.r.t.-velocity block
%     bench.M                   Phi_rv' * Phi_rv matrix
%     bench.lambda              eigenvalues of R'R (ascending order)
%     bench.lambda_max          largest eigenvalue of R'R
%     bench.e_opt               dominant eigenvector of R'R
%     bench.dV_opt_kmps         optimal impulsive ΔV vector [km/s]
%     bench.dr_opt_km           resulting linearized deflection vector [km]
%     bench.dr_max_km           maximum linearized deflection magnitude [km]
%     bench.rA_tf_lin_km        linearized asteroid position at tf [km]
%     bench.miss_vec_lin_km     linearized miss vector relative to Earth [km]
%     bench.miss_lin_km         linearized miss distance relative to Earth [km]
%
%   Notes:
%     - This is Conway's impulsive linear benchmark, not the continuous-
%       thrust optimization problem.
%     - The sign of the optimal eigenvector is arbitrary for the deflection
%       magnitude. Here we choose the sign that increases the Earth-relative
%       miss distance at tf.
%
% Duration from intervention epoch to close approach
T = tf_sec - t0_sec;

% --- STM propagation over time arc from t0 to tf ---
tspan_STM = [0, T];
[Phi_tf, ~] = kep.propagateSTM_2BP(xA_0(:), muSun_km, tspan_STM, odeOpt);

% Extract STM blocks using explicit notation:
%   Phi_rr = d r_f / d r_0,   Phi_rv = d r_f / d v_0
%   Phi_vr = d v_f / d r_0,   Phi_vv = d v_f / d v_0
Phi_rr = Phi_tf(1:3, 1:3);
Phi_rv = Phi_tf(1:3, 4:6);
Phi_vr = Phi_tf(4:6, 1:3);
Phi_vv = Phi_tf(4:6, 4:6);

% Conway uses the position-w.r.t.-velocity block:
%   delta_r(tf) = Phi_rv * delta_v0
M = Phi_rv.' * Phi_rv;                 % matrix for eigenanalysis to find optimal direction of delta_v0 for max delta_r(tf)
[V, D] = eig(M);                       % eigen-decomposition of M; columns of V are eigenvectors, D is diagonal with eigenvalues
lambda = diag(D);                      % eigenvalues of M (R'R), sorted in ascending order
[lambda_max, idx] = max(lambda);       % largest eigenvalue gives max gain direction for delta_v0
e_opt = V(:, idx);                     % optimal direction for delta_v0 to maximize linearized ||delta_r(tf)|| for fixed ||delta_v0||
vhat_opt = e_opt / norm(e_opt);        % normalize to get unit direction

% Candidate optimal impulse and corresponding linearized deflection
DVAst_kms = dVmax_kmps * vhat_opt;      % optimal delta_v0 in km/s
dr_STM = Phi_rv * DVAst_kms(:);         % resulting linearized delta_r(tf) from optimal delta_v0

% Package outputs
bench = struct();
bench.Phi_tf          = Phi_tf;
bench.Phi_rr          = Phi_rr;
bench.Phi_rv          = Phi_rv;
bench.Phi_vr          = Phi_vr;
bench.Phi_vv          = Phi_vv;
bench.M               = M;
bench.lambda          = lambda;
bench.lambda_max      = lambda_max;
bench.e_opt           = DVAst_kms / max(norm(DVAst_kms), eps);
bench.dV_opt_kmps     = DVAst_kms;
bench.dr_opt_km       = dr_STM;
bench.dr_max_km       = norm(dr_STM);
bench.rA_tf_lin_km    = xA_0(1:3) + dr_STM;
bench.miss_vec_lin_km = bench.rA_tf_lin_km - rE_tf(:);
bench.miss_lin_km     = norm(bench.miss_vec_lin_km);
end

function print_conway_diagnostics(ast_name, X0, xA_0, bench, t0_months)
% print_conway_diagnostics  Print the key states and STM block used in the Conway benchmark.

fprintf('\n=== Conway benchmark diagnostics ===\n');
fprintf('Asteroid: %s\n', string(ast_name));

fprintf('Asteroid initial Cartesian state X0 [km, km/s]:\n');
fprintf('%.15g ', X0(:));
fprintf('\n');

fprintf('\nAsteroid state at t0 = -%g months [km, km/s]:\n', t0_months);
fprintf('%.15g ', xA_0(:));
fprintf('\n');

fprintf('\nPhi_rv [km / (km/s)]:\n');
for i = 1:size(bench.Phi_rv, 1)
    fprintf('%.15g ', bench.Phi_rv(i, :));
    fprintf('\n');
end

fprintf('\nConway max theoretical deflection: %.12f km\n', bench.dr_max_km);
end
