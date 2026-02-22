classdef CR3BPLibrary
    % CR3BPLIBRARY  Applied astrodynamics & control utilities for CR3BP work.
    %
    %   Toolbox-style class focused on Circular Restricted Three-Body Problem
    %   (CR3BP) modeling, estimation, targeting, continuation, and plotting.
    %   Provides high-level routines for trajectory design around libration
    %   regions (e.g., halo/lyapunov families) and transfer construction.
    %
    % FEATURES
    %   • Dynamics
    %       - dynamicsCR3BP: 6-state synodic-frame equations of motion
    %       - augmentedDynamicsCR3BP: state + STM propagation (6 + 36)
    %   • Integrators
    %       - integrateCR3BP: plain state propagation (ode89 wrapper)
    %       - stateWithCovariancedDynamics / integrateCR3BPwithCovariance:
    %         EKF-style covariance propagation (Pdot = A P + P A' + G Qs G')
    %   • Plotters
    %       - plotCR3BPOrbit: single/multi-IC propagation or state-history
    %         plotting; optional JC/period colormaps, textured primaries,
    %         arrows, legends, dark mode
    %   • Targeters (single-case differential correction)
    %       - finalStateTargeter_StopAtXf: match [y z vx vy vz] at x = x_f
    %       - finalPositionTargeter_StopAtXf: match [y z] at x = x_f
    %       - targeterPerpendicularXZ_FixedX / _FixedZ / _FixedVy:
    %         enforce perpendicular crossing of xz-plane (y=0, v_x=v_z=0)
    %   • Continuation (families of periodic orbits)
    %       - findPerpendicularXZPeriodicOrbits_FixedX
    %       - findPerpendicularXZPeriodicOrbits_FixedZ
    %       - findPerpendicularXZPeriodicOrbits_FixedVy
    %   • State extraction
    %       - getStateAtTau: interpolate states at fractional period marks
    %   • Initial-guess helpers
    %       - estimateDeltaV0_LPOtoMoon: ΔV magnitude/direction via
    %         pseudo-potential / JC heuristics
    %   • Converters (frames & coordinates)
    %       - sunToMoonSynodic: Sun→Moon vectors in synodic & inertial frames
    %       - bci_to_syn: body-centered inertial → synodic (pos/vel[,period])
    %
    % UNITS & FRAMES
    %   • Default CR3BP normalization: n = 1, distances nondimensionalized
    %     by primary separation; time by 1/n; velocities by (distance/time).
    %   • Some helpers accept/return dimensional data when documented; keep
    %     consistency across a workflow.
    %   • Frames: Synodic (rotating barycentric), BCI (body-centered inertial),
    %     and Sun-centered inertial where noted.
    %
    % DEPENDENCIES
    %   KeplerianOrbitalMechanicsLibrary   – general two-body utilities
    %   AttitudeDeterminationLibrary       – DCM/quaternion utilities (used in converters)
    %
    % EXAMPLES
    %   cr = CR3BPLibrary();
    %   x0 = [1.02 0 0 0 0.15 0]; tspan = [0 3.2];
    %   [T,Y] = cr.integrateCR3BP(x0, 0.01215058, 1, tspan, odeset('RelTol',1e-12,'AbsTol',1e-12));
    %   f = cr.plotCR3BPOrbit([x0 3.2], zeros(5,2), 0.01215058, 1, odeset, 0, 'plotTitle',"CR3BP Demo");
    %
    % NOTES
    %   - Differential correctors assume smooth crossings/events and may
    %     require good initial guesses; check exit flags and iteration logs.
    %   - Continuation routines append results and report success/failure per
    %     step; verify families before downstream use.
    %   - Covariance propagation uses user-supplied Qs and G; ensure proper
    %     scaling in normalized units.
    %
    % AUTHOR
    %   Moacir Fonseca Becker
    %   Purdue University
    %
    % LAST MODIFIED
    %   08/13/2025


    properties (Access = private)
        orb   KeplerianOrbitalMechanicsLibrary
        adc   AttitudeDeterminationLibrary
    end

    methods
        function obj = CR3BPLibrary(orbInstance, adcInstance)
            %CR3BPLibrary  Construct a new CR3BPLibrary object

            %—— Handle orbInstance input ——
            if nargin >= 1 && isa(orbInstance, 'KeplerianOrbitalMechanicsLibrary')
                obj.orb = orbInstance;
            else
                obj.orb = KeplerianOrbitalMechanicsLibrary();
            end

            %—— Handle adcInstance input ——
            if nargin == 2 && isa(adcInstance, 'AttitudeDeterminationLibrary')
                obj.adc = adcInstance;
            else
                obj.adc = AttitudeDeterminationLibrary();
            end
        end


    end



    methods
        %% 3) CR3BP FUNCTIONS

        %% 3.1) Dynamics

        % CR3BP Only
        function dXdt = dynamicsCR3BP(obj,t, X, mu, n)
            % CR3BP (Circular Restricted Three Body Problem) Dynamics
            % Function
            % Inputs:
            %   X - State vector [x, y, z, vx, vy, vz] mu - Mass parameter
            %   (ratio of secondary mass to total system mass) (optional) n
            %   - Mean motion (average angular velocity)
            % Outputs:
            %   dxdt - Derivative of the state vector wrt to time

            % Unpack the state vector into position and velocity components
            % x, y, z - Position coordinates vx, vy, vz - Velocity
            % components

            % Calculate distances 'd' and 'r' for gravitational effect
            % calculations 'd' - Distance from S/C to the secondary body
            % 'r' - Distance from S/C to the primary body

            % Compute the derivatives of position (velocity components)
            % dxdt, dydt, dzdt - Derivatives of x, y, z (same as vx, vy,
            % vz)

            % Compute the derivatives of velocity using CR3BP equations
            % dvxdt, dvydt, dvzdt - Accelerations in x, y, z directions

            if nargin < 5  % If the number of inputs is less than 3
                n = 1;    % Use default value for n
            end

            % Unpack the state vector
            x = X(1);
            y = X(2);
            z = X(3);
            vx = X(4);
            vy = X(5);
            vz = X(6);

            % Distance magnitudes from S/C to each primary
            d = sqrt((x + mu)^2 + y^2 + z^2);     % Distance r13
            r = sqrt((x - 1 + mu)^2 + y^2 + z^2); % Distance r23

            % CR3BP Differential Equations
            dxdt = vx;
            dydt = vy;
            dzdt = vz;
            dvxdt = 2*n*vy + n^2*x - (1 - mu)*(x + mu)/d^3 - mu*(x - 1 + mu)/r^3;
            dvydt = -2*n*vx + n^2*y - (1 - mu)*y/d^3 - mu*y/r^3; % Corrected terms
            dvzdt = -((1 - mu)*z/d^3) - mu*z/r^3;

            % Return the rate of change
            dXdt = [dxdt; dydt; dzdt; dvxdt; dvydt; dvzdt];
        end

        % CR3BP + STM
        function [dXaug_dt] = augmentedDynamicsCR3BP(obj,t, Xaug, mu, n)
            % dynamics_CR3BP_STM computes the time derivative of the state
            % and state transition matrix (STM)
            % for the Circular Restricted Three-Body Problem (CR3BP).
            %
            % Inputs:
            %   t - Time variable (not used in the function, but required
            %   for ode89) Xaug - Augmented state vector that includes the
            %   current state vector and
            %          the reshaped STM vector [x, y, z, vx, vy, vz,
            %          STM_vector]
            %   mu - (CR3BP mass ratio of the secondary body) n - Mean
            %   motion (angular velocity, default is 1 if not specified)
            %
            % Outputs:
            %   Xaug_dot - Time derivative of the augmented state vector,
            %   which includes
            %              the derivative of the state vector and the
            %              derivative of the reshaped STM vector.

            if (nargin < 5) || (isempty(n))   % If the number of inputs is less than 3
                n = 1;                        % Use default value for n
            end

            % ---- CR3BP DYNAMICS ----
            % Extract position components from the state vector
            state_vec = Xaug(1:6);                    % State vector [x y z vx vy vz]
            x = state_vec(1);          % [nondim]
            y = state_vec(2);          % [nondim]
            z = state_vec(3);          % [nondim]

            dState_dt =  obj.dynamicsCR3BP(t,state_vec, mu, n);

            % ---- STM DYNAMICS ---- Split the state vector and STM
            Phi_vec = Xaug(7:end);                % Get the Phi matrix embedded in the aug state vector
            Phi_mtx = reshape(Phi_vec, 6, 6);     % Reshape Phi to a 6X6 matrix

            % Compute the Jacobian A matrix at current state value
            A =  obj.dynamicsJacobianA(x,y,z,mu);

            % Compute the STM time derivative based on A calculated before
            % dPhi/dt =  A * Phi
            dPhi_dt_mtx = A * Phi_mtx;
            dPhi_dt_vec = reshape(dPhi_dt_mtx, 6^2, 1);

            dXaug_dt = [dState_dt; dPhi_dt_vec];    % Return the augmented state vector derivative
        end






        %% ===============================================================
        %% 3.2) Integrators

        % Integrate CR3BP
        function [T, Y] = integrateCR3BP(obj,x0, mu, n, tSpan, optionsODE)
            % Integrate the Circular Restricted Three-Body Problem (CR3BP)
            % dynamics using MATLAB's ode89 solver.
            %
            % Inputs:
            %   x0         - Initial state vector for the orbit, can
            %   include the time duration as the 7th element [x, y, z, vx,
            %   vy, vz, (duration)] mu         - Gravitational parameter
            %   for the CR3BP n          - Mean motion (average angular
            %   velocity) in the CR3BP system tSpan      - Timespan for
            %   integration [start, end] optionsODE - ODE solver options
            %
            % Outputs:
            %   T - Time vector from the ODE solver Y - State vectors from
            %   the ODE solver

            % Ensure x0 contains only state variables, not time duration
            if length(x0) > 6
                x0 = x0(1:6); % If x0 has 7 elements, consider only the first 6 as the state vector
            end

            % Define the dynamics function as an anonymous function
            dynamicsFunc = @(t, X) obj.dynamicsCR3BP(obj,X, mu, n);

            % Call ode89 with the dynamics function and other arguments
            [T, Y] = ode89(dynamicsFunc, tSpan, x0, optionsODE);
        end





        % Define the augmented dynamics function
        function dXdt = stateWithCovariancedDynamics(obj, t, X, mu, n, Qs, G)
            % augmentedDynamics Computes the derivative of the
            % augmented state vector
            %
            % Inputs:
            %   t - Current time (not used as CR3BP is autonomous) X -
            %   Current augmented state vector [42x1]
            %
            % Outputs:
            %   dXdt - Derivative of the augmented state vector [42x1]

            % Unpack state and covariance from augmented vector
            state = X(1:6);                   % [x; y; z; vx; vy; vz]
            P = reshape(X(7:end), 6, 6);      % [6x6] Covariance matrix

            % Compute state derivatives using CR3BP dynamics
            t = 0;  % Dummy time value
            state_dot = obj.dynamicsCR3BP(t, state, mu, n); % [6x1]

            % Compute Jacobian matrix A for CR3BP dynamics
            F = obj.dynamicsJacobianA(state(1),state(2),state(3), mu);     % [6x6]

            % Compute Pdot for covariance propagation: Pdot = A*P +
            % P*A' + G*Qs*G'
            Pdot = F * P + P * F' + G * Qs * G';       % [6x6]

            % Flatten Pdot to [36x1] for integration
            dPdt = Pdot(:);

            % Combine state derivatives and covariance derivatives
            dXdt = [state_dot; dPdt];                 % [42x1]
        end

        % Integrate CR3BP Dynamics with State Estimate Covariance
        function [T, Y] = integrateCR3BPwithCovariance(obj, X0, mu, n, tspan, Qs, G, optionsODE)
            % integrateCR3BPwithCovariance Integrates the Circular
            % Restricted Three-Body Problem (CR3BP) dynamics along with the
            % state estimate covariance matrix using MATLAB's ode89 solver.
            %
            % This function integrates both the state vector and the
            % covariance matrix over the specified timespan, outputting the
            % augmented state and covariance in the Y output. This is
            % essential for Extended Kalman Filter (EKF) implementations
            % where covariance propagation is required alongside state
            % dynamics.
            %
            % Inputs:
            %   obj         - Instance of the class containing this method.
            %   X0          - Initial augmented state vector [42x1] where:
            %                   - First 6 elements are the initial state
            %                   [x, y, z, vx, vy, vz]. - Next 36 elements
            %                   are the initial covariance matrix P
            %                   (flattened column-wise).
            %   mu          - Gravitational parameter for the CR3BP
            %   (dimensionless). n           - Mean motion (average angular
            %   velocity) in the CR3BP system (dimensionless). tSpan
            %   - Timespan for integration [t0, tf] in nondimensional units
            %   (tau). Qs          - Process noise covariance matrix for
            %   EKF [6x6].
            %                   Represents the uncertainty in the process
            %                   model.
            %   G           - Process noise gain matrix [6x6].
            %                   Maps the process noise into the state
            %                   space.
            %   optionsODE  - ODE solver options (e.g., ode89 options)
            %   created using odeset.
            %
            % Outputs:
            %   T - Time vector from the ODE solver in nondimensional units
            %   (tau). Y - Augmented state vectors from the ODE solver,
            %   where each row contains:
            %       [x, y, z, ves bx, vy, vz, P(:)']

            % Validate Inputs
            if length(X0) ~= 42
                error('Initial state vector X0 must be a 42x1 vector: [6 state variables; 36 covariance matrix elements].');
            end
            if size(Qs,1) ~=6 || size(Qs,2) ~=6
                error('Process noise covariance matrix Qs must be a 6x6 matrix.');
            end
            if size(G,1) ~=6 || size(G,2) ~=6
                error('Process noise gain matrix G must be a 6x6 matrix.');
            end

            % Define the dynamics function as an anonymous function
            dynamicsFunc = @(t, X) obj.stateWithCovariancedDynamics(t, X, mu, n, Qs, G);

            % Call ode89 with the dynamics function and other arguments
            [T, Y] = ode89(dynamicsFunc, tspan, X0, optionsODE);
        end

        %% ===============================================================
        %% 3.3) Plotters
        % Single or Multiple ICs (CR3BP)
        function f = plotCR3BPOrbit(obj,ICs, Lpoints, mu, n, optionsODE, Flag, varargin)
            % plotCR3BPOrbit Plots the trajectory for the Circular
            % Restricted Three Body Problem (CR3BP) given initial
            % conditions or precomputed states, marks the libration points,
            % and optionally colors the orbits based on their Jacobi
            % constant values or provided color.
            %  Last Update: 17-04-2024
            % Usage:
            %   f = obj.plotCR3BPOrbit(ICs, Lpoints, mu, n, optionsODE,
            %   Flag, varargin)
            %
            % Inputs:
            %   ICs - An initial state vector (col or row) of 7
            %   elements or a matrix where each row contains the state
            %   vector and final time for a particular orbit, or
            %   precomputed states for direct plotting. Lpoints - A 5x2
            %   matrix containing the x and y coordinates of the five
            %   libration points, L1 through L5. A row should be set to
            %   [0, 0] if a particular libration point is not to be
            %   plotted. mu - Gravitational parameter for the CR3BP,
            %   representing the mass ratio of the two primary bodies.
            %   n - Mean motion (average angular velocity) in the CR3BP
            %   system, typically set to 1 for normalized units.
            %   optionsODE - MATLAB ODE solver options structure,
            %   typically created with odeset. Flag - A boolean flag
            %   indicating whether to color the orbits based on their
            %   Jacobi Constant (JC) or period values. If 1 JC, if 2
            %   period values. varargin - Additional optional
            %   parameters
            %
            % Plots CR3BP trajectories from either ICs (propagated on-the-fly) or
            % pre-computed state histories. Supports colouring by Jacobi constant
            % (Flag==1) or orbital period (Flag==2). For Flag==0, the routine leaves
            % any existing colourbar untouched and does not create a new one.
            %---------------------------------------------------------------------
            % LAST UPDATE: 08-Jul-2025 – added smart colourbar handling & robust axes
            %---------------------------------------------------------------------
            % OPTIONAL NAME-VALUE PAIRS
            %   'plotTitle'             – custom title (default "CR3BP Propagation")
            %   'figureHandle'          – plot into existing figure
            %   'arrows'                – true/false, draw velocity arrows
            %   'blackBackground'       – black theme
            %   'plotRealisticPrimaries'– textured Earth/Moon
            %   'showLegends'           – toggle legend
            %   'states'                – if true, ICs is a state history
            %---------------------------------------------------------------------

            % Create an input parser
            p = inputParser;

            % Define default values
            defaultArrows = false;
            defaultFigHandle = [];
            defaultColor = 'b';
            defaultLineWidth = 2;
            blackBackground = false;
            defaultPlotRealisticPrimaries = true;
            defaultShowLegends = true;             % Default to true
            defaultPlotTitle = 'CR3BP Propagation'; % NEW default title


            % Add parameters to input parser
            addParameter(p, 'arrows', defaultArrows, @islogical);
            addParameter(p, 'figureHandle', defaultFigHandle, @(x) isempty(x) || (ishandle(x) && strcmp(get(x, 'Type'), 'figure')));
            addParameter(p, 'states', false, @islogical); % Handling 'states' flag
            addParameter(p, 'color', defaultColor, @(x) (ischar(x) || isstring(x) || (isnumeric(x) && numel(x)==3)));
            addParameter(p, 'LineWidth', defaultLineWidth, @isnumeric); % Line width input for plotting
            addParameter(p, 'LineStyle', '-', @(x) ischar(x) || (isstring(x) && isscalar(x)));
            addParameter(p, 'blackBackground', blackBackground, @islogical);  % Added 'background' parameter
            addParameter(p, 'plotRealisticPrimaries', defaultPlotRealisticPrimaries, @islogical);
            addParameter(p, 'showLegends', defaultShowLegends, @islogical);
            addParameter(p, 'plotTitle', defaultPlotTitle, @(x) ischar(x) || isstring(x)); % NEW parameter


            % Parse input arguments
            parse(p, varargin{:});

            % Normalize user color to 1×3 RGB (works for 'r','red',"#FF0000",[1 0 0],[1;0;0])
            plotColorRaw = p.Results.color;
            if isstring(plotColorRaw); plotColorRaw = char(plotColorRaw); end
            plotColorRGB = validatecolor(plotColorRaw);   % 1×3 numeric


            % Extract values from the parser
            arrows = p.Results.arrows;
            figHandle = p.Results.figureHandle;
            plotStatesDirectly = p.Results.states;   % Use directly the parsed result
            plotColor = p.Results.color;             % Extracted color parameter
            LineWidth = p.Results.LineWidth;         % Extracted line width
            LineStyle = p.Results.LineStyle;
            blackBackground = p.Results.blackBackground;  % Extract 'background' parameter
            plotRealisticPrimaries = p.Results.plotRealisticPrimaries;
            showLegends = p.Results.showLegends;
            plotTitle = char(p.Results.plotTitle);   % Ensure it is char for MATLAB title


            % - - - - Preliminary Setup - - - - -
            t_star = 375704.306539867;                  % [s]
            t_star = t_star/3600;                       % [hrs]

            % Load textures if primaries are to be plotted
            if plotRealisticPrimaries
                earthTexture = imread('earth.jpg');
                moonTexture = imread('moon.tif');
            end

            % - - - - Sanity Check on the ICs Vector or Matrix Provided - -
            % - - -
            if ~plotStatesDirectly
                % Check if ICs is a vector and has exactly 7 elements
                if isvector(ICs) && length(ICs) == 7
                    if iscolumn(ICs)
                        ICs = ICs';                 % Convert column vector to row vector
                    end
                elseif ismatrix(ICs) && size(ICs, 2) == 7
                    % No action needed for matrix with correct format
                else
                    error('ICs must be a 7-element vector or a matrix with 7 columns. Each row represents [x, y, z, vx, vy, vz, Tmax].');
                end
            end

            % Check and clean ICs from NaN rows
            validICs = all(~isnan(ICs), 2);  % Logical index of rows without NaN
            ICs = ICs(validICs, :);          % Remove rows with NaN

            % - - - - Figure Initial Setup - - - - -

            if isempty(figHandle)
                f = figure('Visible','on', 'Position', [100, 100, 800, 600]);
                hold on
            else
                f = figure(figHandle);
                set(f, 'Position', [100, 100, 800, 600]);
                hold on;
            end


            % --- Set the background color based on the 'background'
            % parameter ---
            % Set background and axis colors
            if blackBackground
                % Set to black background
                set(gca, 'Color', 'k', 'XColor', 'w', 'YColor', 'w', 'ZColor', 'w', 'FontSize', 14);
                set(gcf, 'Color', 'k');
                xlabel('X (nd)', 'FontSize', 18, 'Color', 'w');
                ylabel('Y (nd)', 'FontSize', 18, 'Color', 'w');
                zlabel('Z (nd)', 'FontSize', 18, 'Color', 'w');
                title(plotTitle,      'FontSize', 18, 'Color', 'w');

            else
                % Set to white background (default)
                set(gca, 'Color', 'w', 'XColor', 'k', 'YColor', 'k', 'ZColor', 'k', 'FontSize', 14);
                set(gcf, 'Color', 'w');
                xlabel('X (nd)', 'FontSize', 18, 'Color', 'k');
                ylabel('Y (nd)', 'FontSize', 18, 'Color', 'k');
                zlabel('Z (nd)', 'FontSize', 18, 'Color', 'k');
                title(plotTitle,      'FontSize', 18, 'Color', 'k');

            end


            %  - - - - Jacobi Constants or Period Color Mapping to Orbits
            %  when Flag is 1 or 2 - - -
            if Flag == 1  % If JC plot requested
                [JC_Array, ~, ~] = obj.jacobiConstantCR3BP(ICs, mu, n);  % Calculate JC for each row of states
                cmap = spring(size(ICs, 1));  % Create a colormap with a color for each orbit

                if size(ICs, 1) > 1  % If multiple orbits
                    JC_min = min(JC_Array);  % Find min JC
                    JC_max = max(JC_Array);  % Find max JC

                    % Handle the case when all Jacobi constants are
                    % equal
                    if JC_max == JC_min
                        deltaJC = max(1e-6 * abs(JC_min), eps);
                        JC_max = JC_min + deltaJC;
                        colorIndices = ceil(size(cmap, 1) / 2) * ones(size(JC_Array));
                    else
                        colorIndices = round(((JC_Array - JC_min) / (JC_max - JC_min)) * (size(cmap, 1) - 1)) + 1;
                    end

                    colors = cmap(colorIndices, :);  % Get color values
                else  % If single orbit
                    JC_min = JC_Array;
                    deltaJC = max(1e-6 * abs(JC_min), eps);
                    JC_max = JC_min + deltaJC;
                    colors = cmap(ceil(size(cmap, 1) / 2), :);  % Use middle color
                end

                % Set the color axis limits using caxis
                caxis([JC_min, JC_max]);

            elseif Flag == 2                    % Period color map
                if plotStatesDirectly
                    error('I cannot know the period and add it to the plot if your input was a history of states and not an initial condition')
                end
                periods = ICs(:, 7);                                  % Get all the periods from the ICs
                if size(ICs, 1) > 1                                   % If there are multiple orbits
                    cmap = winter(length(periods));
                    minVal = min(periods);
                    maxVal = max(periods);
                    colorIndices = round(((periods - minVal) / (maxVal - minVal)) * (size(cmap, 1) - 1)) + 1;
                    colors = cmap(colorIndices, :);
                    clim(t_star*[minVal maxVal]);                        % Set the colormap axis limits based on periods
                else
                    cmap = spring(1);                                    % Generate a colormap for one item
                    maxVal = periods + 0.00000001;                       % Fake max period for color scale
                    colors = cmap(1, :);                                 % Use the only available color
                    clim(t_star*[periods maxVal]);                       % Set limits with fake max
                end

            else
                % Flag == 0 → single solid color for all orbits (numeric RGB)
                colors = repmat(plotColorRGB, size(ICs,1), 1);  % N×3
            end

            % Initialize legend entries only if legends are to be shown
            if showLegends && ~plotStatesDirectly
                legendEntries = cell(size(ICs, 1), 1);
            end



            % --- UPDATE COLORBAR OR SKIP --------------------------------------
            if Flag == 1 || Flag == 2          % only touch the bar when we really need one
                % Does a colorbar already exist in this figure?
                existingCB = findall(gcf,'Type','colorbar');

                if isempty(existingCB)         % none → make a brand-new one and set colormap
                    colormap(gca, cmap);       % safe: only affects these axes
                    cb = colorbar;             % create bar
                    if Flag == 1
                        title(cb,'Jacobi Constant');
                    else
                        title(cb,'Period (hours)');
                    end
                    % style for dark background
                    if blackBackground
                        set(cb,'Color','w');
                        cb.Title.Color = 'w';
                    end
                    set(cb,'FontSize',18);     % keep this on the new bar only
                else                           % a bar is already there → leave it alone
                    cb = existingCB(1);        % (optional) handle, but no edits
                end
            end



            % - - - - PROPAGATE AND PLOT EACH IC - - -

            if ~plotStatesDirectly

                for i = 1:size(ICs, 1)
                    x0_nd = ICs(i, 1:6);            % Initial state conditions
                    Tmax_nd = ICs(i, end);          % Last element is Max adimensional time for simulation
                    tspan_nd = [0, Tmax_nd];        % Non-dimensional timespan for sim

                    % % Change color assignment to use colormap
                    % if Flag > 0
                    %        color = colors(i, :);     % Corrected indexing
                    % else
                    %       defaultColors = lines(size(ICs, 1)); % Generate enough colors for each orbit
                    %       color = defaultColors(i, :);         % Use modulo to cycle through colors
                    % end

                    color = colors(i,:);   % colors was set correctly above for ANY Flag


                    % Propagate the ODE
                    [t, y] = ode89(@(t, X) obj.dynamicsCR3BP(t,X, mu, n), tspan_nd, x0_nd, optionsODE);

                    % Plot the trajectory of the S/C
                    plot3(y(:, 1), y(:, 2), y(:, 3), 'Color', color, 'LineWidth', LineWidth, 'LineStyle', LineStyle); 
                    

                    % If requested, calc and plot vector of initial velocity

                    if arrows
                        ax = gca;
                        xl = ax.XLim; yl = ax.YLim; zl = ax.ZLim;
                        plotSize = norm([diff(xl), diff(yl), diff(zl)]);
                        arrowScale = 0.10 * plotSize;  % arrows about 10% of the overall axes size

                        pos_vector = ICs(i,1:3); vel_vector = ICs(i,4:6);
                        % Use the computed arrowScale as the "scale" argument:
                        quiver3( ...
                            pos_vector(1), pos_vector(2), pos_vector(3), ...
                            vel_vector(1), vel_vector(2), vel_vector(3), ...
                            arrowScale, 'Color', 'k', 'MaxHeadSize', 3, 'linewidth', 2);
                    end


                    % Generate label for the legend if legends are to
                    % be shown
                    if showLegends
                        legendEntries{i} = sprintf('IC-%d: [%.2f, %.2f, %.2f, %.2f, %.2f, %.2f]', i, x0_nd);
                    end

                end
            else
                % Plot already computed state trajectories
                plot3(ICs(:,1), ICs(:,2), ICs(:,3), 'Color', plotColorRGB, 'LineWidth', LineWidth, 'LineStyle', LineStyle);

                drawnow limitrate

            end

            % ---------- EARTH & MOON -------------------------------------------
            EarthPosition = [- mu, 0, 0];     % Earth at origin
            MoonPosition = [1 - mu, 0, 0]; % Moon at (1 - mu, 0, 0)
            EarthRadius = 6378.1 / 384400; % Earth's radius in nondimensional units
            MoonRadius = 1737.4 / 384400;  % Moon's radius in nondimensional units

            if plotRealisticPrimaries
                % Plot Earth with texture
                [Xe, Ye, Ze] = sphere(50);
                EarthHandle = surf(EarthPosition(1) + EarthRadius*Xe, EarthPosition(2) + EarthRadius*Ye, EarthPosition(3) + EarthRadius*Ze, 'EdgeColor', 'none', 'FaceColor', 'texturemap', 'CData', earthTexture, 'DisplayName', 'Earth');
                set(EarthHandle, 'FaceAlpha', 0.8, 'FaceLighting', 'gouraud');

                % Plot the Moon with texture
                [Xm, Ym, Zm] = sphere(50);
                MoonHandle = surf(MoonPosition(1) + MoonRadius*Xm, MoonPosition(2) + MoonRadius*Ym, MoonPosition(3) + MoonRadius*Zm, 'EdgeColor', 'none', 'FaceColor', 'texturemap', 'CData', moonTexture, 'DisplayName', 'Moon');
                set(MoonHandle, 'FaceAlpha', 1);
            else
                scatter3(MoonPosition(1), MoonPosition(2), 0, 40, 'color', [.5 .5 .5]);
                scatter3(EarthPosition(1), EarthPosition(2), 0, 40, 'b');
            end


            % ---------- LIBRATION POINTS ----------------------------------------
            if blackBackground
                lpMarkerColor = 'w';
            else
                lpMarkerColor = 'k';
            end

            for i = 1:size(Lpoints, 1)
                Lpoint = Lpoints(i, :);
                if Lpoint(1) ~= 0
                    % Plot libration point using a dot marker
                    % scatter3(Lpoint(1), Lpoint(2), 0, 40, 'w',
                    % 'filled', 'DisplayName', sprintf('L%d', i));
                    scatter3(Lpoint(1), Lpoint(2), 0, 40, 'r', 'filled', 'DisplayName', sprintf('L%d', i));

                    % Add label above the libration point
                    % text(Lpoint(1), Lpoint(2), 0, sprintf('L%d', i),
                    % 'Color', 'white', 'VerticalAlignment', 'bottom',
                    % 'HorizontalAlignment', 'center', 'FontSize', 10,
                    % 'FontWeight', 'bold');
                    text(Lpoint(1), Lpoint(2), 0, sprintf('L%d', i), 'Color', lpMarkerColor, 'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'center', 'FontSize', 10, 'FontWeight', 'bold');
                end
            end

            % ---------- LEGENDS -------------------------------------------
            if showLegends && ~plotStatesDirectly
                legend(legendEntries, 'Location', 'bestoutside', 'TextColor', 'black', 'Color', 'white');
            end

            % ---------- FINAL TOUCHES -------------------------------------------
            % xlim([-1.5 1.5]);   ylim([-1 1.5]);     zlim([-1 1]);

            rotate3d on;
            % axis equal;
            set(gca,'DataAspectRatio',[1 1 1])
            cameratoolbar('Show');
            cameratoolbar('SetMode','orbit');

        end



        % Zero Velocity Curves ZVCs
        function h = plotZVC(obj, mu, C0, xlims, ylims, N, XLi, r0)
            %PLOTZVC_CR3BP  Plot Zero-Velocity Curves (ZVC) for a given Jacobi value in the CR3BP.
            %
            %   h = plotZVC_CR3BP(obj, mu, C0, xlims, ylims, N, XLi, r0)
            %
            % Description
            % ----------
            % Draws the ZVC boundary (where v = 0) for the nondimensional CR3BP with
            % mean motion n = 1. The forbidden region { 2U* - C < 0 } is shaded;
            % the boundary where 2U* - C = 0 is plotted as a contour line.
            %
            % Inputs
            % ------
            % obj    : (unused hook for class/library methods; pass your CR3BP object)
            % mu     : mass ratio, m2/(m1 + m2)  (scalar)
            % C0     : Jacobi constant for which to draw the ZVC (scalar)
            % xlims  : 1x2 limits for x-axis, e.g., [-1.6 1.6]
            % ylims  : 1x2 limits for y-axis, e.g., [-1.6 1.6]
            % N      : grid size per axis (e.g., 400–800). If empty or omitted, N=600.
            % XLi    : (optional) libration point states to annotate.
            %          Accepts either:
            %            - kx2 matrix of [x y] coordinates, or
            %            - kx6 matrix of CR3BP states [x y z vx vy vz] (first two cols used).
            %          Pass [] to skip.
            % r0     : (optional) initial position to mark.
            %          Accepts 1x3 or 3x1. Only x,y are used. Pass [] to skip.
            %
            % Outputs
            % -------
            % h      : struct of graphics handles:
            %          .fig, .ax, .hForbidden, .hZVC, .hEarth, .hMoon, .hLi, .hIC
            %
            % Notes
            % -----
            % * Units are nondimensional; n = 1 is assumed.
            % * The potential used is U* = (1-mu)/d + mu/r + 0.5*(x^2+y^2),
            %   with d = sqrt((x+mu)^2 + y^2) and r = sqrt((x-1+mu)^2 + y^2).
            % * The ZVC boundary is the level set C_grid(x,y) = C0 where
            %     C_grid = x^2 + y^2 + 2(1-mu)/d + 2mu/r.
            % * r0 may be given as row or column; it is internally reshaped to 3x1.
            %
            % Example
            % -------
            %   XLi = [xL1 yL1; xL2 yL2; xL3 yL3; 0.5-mu  sqrt(3)/2; 0.5-mu -sqrt(3)/2];
            %   plotZVC_CR3BP(cr3bpObj, mu, C_IC, [-1.6 1.6], [-1.6 1.6], 600, XLi, [-0.27 -0.42 0]);
            %
            % See also: contour, contourf

            % ---------- Defaults & input hygiene ----------
            if nargin < 6 || isempty(N), N = 600; end
            if nargin < 7, XLi = []; end
            if nargin < 8, r0  = []; end

            % Accept either kx6 or kx2 for XLi; use first two columns for plotting
            if ~isempty(XLi)
                if size(XLi,2) >= 2
                    XLi = XLi(:,1:2);
                else
                    error('plotZVC_CR3BP:BadXLi','XLi must have at least 2 columns [x y].');
                end
            end

            % Accept row or column for r0; keep only first two components
            if ~isempty(r0)
                r0 = r0(:);                        % make column
                if numel(r0) < 2
                    error('plotZVC_CR3BP:BadR0','r0 must have at least x and y components.');
                end
            end

            % ---------- Grid & potential field ----------
            x = linspace(xlims(1), xlims(2), N);
            y = linspace(ylims(1), ylims(2), N);
            [X,Y] = meshgrid(x,y);

            % distances to primaries (avoid singularities)
            d = sqrt((X+mu).^2 + Y.^2);   d(d < 1e-12) = 1e-12;
            r = sqrt((X-1+mu).^2 + Y.^2); r(r < 1e-12) = 1e-12;

            % ZVC field with v=0: C_grid = x^2 + y^2 + 2(1-mu)/d + 2mu/r
            Cgrid = X.^2 + Y.^2 + 2*(1-mu)./d + 2*mu./r;

            % ---------- Plot ----------
            h.fig = figure('Color','w');
            h.ax  = axes('NextPlot','add'); grid(h.ax,'on'); box(h.ax,'on'); axis(h.ax,'equal');

            % ZVC boundary Cgrid = C0
            h.hZVC = contour(h.ax, X, Y, Cgrid, [C0 C0], 'k', 'LineWidth', 1.8);

            % Primaries in barycentric coords
            h.hEarth = scatter(h.ax, -mu, 0, 70, 'b', 'filled');
            h.hMoon  = scatter(h.ax, 1-mu, 0, 70, 'k', 'filled');

            % Libration points (optional)
            h.hLi = gobjects(0);
            if ~isempty(XLi)
                h.hLi = scatter(h.ax, XLi(:,1), XLi(:,2), 45, 'k', 'filled');
                txt = {'L1','L2','L3','L4','L5'};
                for i = 1:min(5,size(XLi,1))
                    text(h.ax, XLi(i,1)+0.03, XLi(i,2)+0.03, txt{i}, ...
                        'Color','k','FontSize',10,'Interpreter','none');
                end
            end

            % Initial condition marker (optional)
            h.hIC = gobjects(1);
            if ~isempty(r0)
                h.hIC = scatter(h.ax, r0(1), r0(2), 45, 'r', 'filled');
                text(h.ax, r0(1)+0.03, r0(2)-0.06, 'IC', 'Color','r');
            end

            xlim(h.ax, xlims); ylim(h.ax, ylims);
            xlabel(h.ax, 'x (nd)'); ylabel(h.ax, 'y (nd)');
            title(h.ax, sprintf('ZVC for C = %.12f   (CR3BP, nd)', C0));
            legend(h.ax, {'ZVC (v=0)','Earth','Moon','L-points','IC'}, ...
                'Location','bestoutside');

        end



        %% ===============================================================
        %% 3.4) Targeter Functions to Find Transfers


        function [X0_optimized , dV0_opt , dVf , Ttraj , exit_flag , ...
                iter_count   , errHistory , iterLog , trajHistory , X0_hist , Xplane_hist] = ...
                finalStateTargeter_StopAtXf( ...
                obj, X0_start , dV_guess , X_target , mu , n , Tmax , tolODE , tolErr , maxIter)

            % ===============================================================
            %  Final-state targeter ( y , z , vx , vy , vz ) – stop at X = x_f
            % ===============================================================
            %
            %  Purpose:
            %    Iteratively corrects the initial velocity so that, after propagating
            %    in the CR3BP until crossing the x-plane (x = x_f), the final state
            %    matches the target values in:
            %       - y, z, vx, vy, vz
            %  Method:
            %    Uses a differential correction scheme:
            %      1. Propagate state + STM until x = x_f.
            %      2. Compute position/velocity errors vs. target.
            %      3. Build a linear system (K-matrix) linking Δ(initial velocity) to final errors.
            %      4. Solve for correction via pseudoinverse of K.
            %      5. Update initial velocity guess and iterate until convergence.
            %
            %  Inputs:
            %    obj         : Object with CR3BP dynamics methods.
            %    X0_start    : Initial state [x y z vx vy vz] at t = 0.
            %    dV_guess    : Initial guess for ΔV at t = 0.
            %    X_target    : Target final state [x y z vx vy vz] at x = x_f.
            %    mu, n       : CR3BP parameters (mass ratio, mean motion).
            %    Tmax        : Max allowed propagation time.
            %    tolODE      : ODE solver tolerances (RelTol, AbsTol).
            %    tolErr      : Convergence tolerance on error norm.
            %    maxIter     : Max correction iterations.
            %
            %  Outputs:
            %    X0_optimized: Optimized initial state [x y z vx vy vz].
            %    dV0_opt     : Optimal ΔV applied at t = 0.
            %    dVf         : ΔV to match final velocity exactly.
            %    Ttraj       : Propagation time until x = x_f.
            %    exit_flag   :  1 = converged, -1 = no convergence.
            %    iter_count  : Number of iterations used.
            %    errHistory  : Error norm history per iteration.
            %
            % ===============================================================

            % -------- initialization ----------------------------------------
            verbose     = false;
            iter_count  = 0;
            exit_flag   = 0;

            errHistory  = [];     % scalar norm per iteration
            iterLog     = [];     % one row per iteration (see header)
            trajHistory = {};     % cell of [x y z vx vy vz t]
            X0_hist     = [];     % [x0..vz0 t0]
            Xplane_hist = [];     % [x..vz tf hit plane_gap]

            if isrow(X0_start),  X0_start = X0_start.'; end
            X0_start   = X0_start(1:6);
            rf_target  = X_target(1:3);      % desired final position
            vf_target  = X_target(4:6);      % desired final velocity

            X0_optimized      = X0_start;    % initialize optimal X0
            X0_optimized(4:6) = X0_optimized(4:6) + dV_guess(:);

            xf_tar  = rf_target(1);   yf_tar  = rf_target(2);   zf_tar  = rf_target(3);

            % Event opts (+ MaxStep helps not to leap over the root)
            options = odeset('Events',@eventStopAtXf,'RelTol',tolODE,'AbsTol',tolODE, ...
                'MaxStep', Tmax/2000);

            % -------- differential-correction loop --------------------------
            while iter_count < maxIter

                % Log the iterate's starting state (t=0 in 7th col)
                X0_hist(end+1,:) = [X0_optimized(:).'  0];

                % propagate state+STM; capture event outputs
                [T, Xaug, TE, YE, ~] = ode89(@(t,X) obj.augmentedDynamicsCR3BP(t,X,mu,n), ...
                    [0 Tmax] , [X0_optimized ; reshape(eye(6),36,1)] , ...
                    options);

                % Build a [x y z vx vy vz t] history (downsample to ~500 samples)
                if numel(T) > 500
                    idx = round(linspace(1,numel(T),500));
                else
                    idx = 1:numel(T);
                end
                trajHistory{end+1,1} = [ Xaug(idx,1:6) , T(idx) ];

                % extract final state and STM: prefer event, else use last point
                hit     = ~isempty(TE);
                if hit
                    tf     = TE(end);
                    Xaug_f = YE(end,:);        % augmented state at the event
                else
                    tf     = T(end);
                    Xaug_f = Xaug(end,:);
                end
                Xf     = Xaug_f(1:6).';
                Phi_tf = reshape(Xaug_f(7:end),6,6);

                % plane diagnostics
                plane_gap = Xf(1) - xf_tar;    % should be ~0 if hit==true

                % compute y,z pos errors and vx,vy,vz errors (goal - actual form)
                posErr = [ yf_tar - Xf(2) ;
                    zf_tar - Xf(3) ];
                velErr = [ vf_target(1) - Xf(4) ;
                    vf_target(2) - Xf(5) ;
                    vf_target(3) - Xf(6) ];

                % keep nondimensional bookkeeping
                errVec  = [posErr ; velErr];
                errNorm = norm(errVec);
                errHistory(end+1,1) = errNorm;

                % log the plane hit state for this iteration
                Xplane_hist(end+1,:) = [Xf(:).'  tf  double(hit)  plane_gap];

                if verbose
                    % also print physical units for intuition (optional)
                    l_star_km  = 389703;
                    v_star_kms = 1.01754797650856;
                    dy_km   = posErr(1) * l_star_km;
                    dz_km   = posErr(2) * l_star_km;
                    dv_kms  = norm(velErr) * v_star_kms;

                    fprintf('iter %3d  tf=%.6f  hit=%d  plane_gap=%.3e  |pos(y,z)|=%.3e km  |vel|=%.3e km/s\n', ...
                        iter_count, tf, hit, plane_gap, hypot(dy_km,dz_km), dv_kms);
                end

                % --- convergence? ---------------------------------------------------
                if errNorm < tolErr
                    exit_flag  = 1;
                    Ttraj      = tf;
                    dVf        = (vf_target - Xf(4:6));                   % ΔV at final hit
                    dV0_opt    = X0_optimized(4:6) - X0_start(4:6);       % net ΔV at t0
                    % final log row (no ΔVcorr this time)
                    iterLog(end+1,:) = [iter_count, tf, plane_gap, posErr(1), posErr(2), velErr(1), velErr(2), velErr(3), double(hit), errNorm, NaN];
                    return
                end

                % --- build K matrix -------------------------------------------------
                vxf = Xf(4); vyf = Xf(5); vzf = Xf(6);
                % guard tiny vxf to avoid blow-up in divisions
                if abs(vxf) < 1e-8, vxf = sign(vxf + (vxf==0)) * 1e-8; end

                Xdot_tf = obj.dynamicsCR3BP(0,Xf,mu,n);  % [ẋ ẏ ż v̇x v̇y v̇z]
                axf = Xdot_tf(4); ayf = Xdot_tf(5); azf = Xdot_tf(6);

                K = [ Phi_tf(2,4)-Phi_tf(1,4)*(vyf/vxf) , Phi_tf(2,5)-Phi_tf(1,5)*(vyf/vxf) , Phi_tf(2,6)-Phi_tf(1,6)*(vyf/vxf) ;
                    Phi_tf(3,4)-Phi_tf(1,4)*(vzf/vxf) , Phi_tf(3,5)-Phi_tf(1,5)*(vzf/vxf) , Phi_tf(3,6)-Phi_tf(1,6)*(vzf/vxf) ;
                    Phi_tf(4,4)-Phi_tf(1,4)*(axf/vxf) , Phi_tf(4,5)-Phi_tf(1,5)*(axf/vxf) , Phi_tf(4,6)-Phi_tf(1,6)*(axf/vxf) ;
                    Phi_tf(5,4)-Phi_tf(1,4)*(ayf/vxf) , Phi_tf(5,5)-Phi_tf(1,5)*(ayf/vxf) , Phi_tf(5,6)-Phi_tf(1,6)*(ayf/vxf) ;
                    Phi_tf(6,4)-Phi_tf(1,4)*(azf/vxf) , Phi_tf(6,5)-Phi_tf(1,5)*(azf/vxf) , Phi_tf(6,6)-Phi_tf(1,6)*(azf/vxf) ];

                % solve least squares for ΔV correction
                dVcorr = pinv(K) * errVec;     % sign matches goal-actual definition above

                % apply and log
                X0_optimized(4:6) = X0_optimized(4:6) + dVcorr;
                iterLog(end+1,:)  = [iter_count, tf, plane_gap, posErr(1), posErr(2), velErr(1), velErr(2), velErr(3), double(hit), errNorm, norm(dVcorr)];

                iter_count = iter_count + 1;

                if verbose
                    fprintf('iter %2d  ||err||=%.3e   ||ΔVcorr||=%.3e nd\n', iter_count, errNorm, norm(dVcorr));
                end
            end

            % -------- no convergence -----------------------------------------------
            exit_flag    = -1;
            X0_optimized = NaN(6,1); dV0_opt = NaN(1,3);
            dVf          = NaN(1,3); Ttraj   = NaN;

            % ---------- nested event -----------------------------------------------
            function [value,isterminal,direction] = eventStopAtXf(~,Xaug)
                value      = Xaug(1) - xf_tar;  % stop when x = x_target
                isterminal = 1;                 % halt
                direction  = 0;                 % either crossing direction accepted
            end
        end





        % ===============================================================
        %  Final-state targeter  ( y , z , vx , vy , vz )  – stop at X = x f
        % ===============================================================

        % function [X0_optimized , dV0_opt , dVf , Ttraj , exit_flag , ...
        %         iter_count   , errHistory] = ...
        %         finalStateTargeter_StopAtXf( ...
        %         obj, X0_start , dV_guess , X_target , mu , n , Tmax , tolODE , tolErr , maxIter)
        %     % ===============================================================
        %     %  Final-state targeter ( y , z , vx , vy , vz ) – stop at X = x_f
        %     % ===============================================================
        %     %
        %     %  Purpose:
        %     %    Iteratively corrects the initial velocity so that, after propagating
        %     %    in the CR3BP until crossing the x-plane (x = x_f), the final state
        %     %    matches the target values in:
        %     %       - y, z, vx, vy, vz
        %     %  Method:
        %     %    Uses a differential correction scheme:
        %     %      1. Propagate state + STM until x = x_f.
        %     %      2. Compute position/velocity errors vs. target.
        %     %      3. Build a linear system (K-matrix) linking Δ(initial velocity) to final errors.
        %     %      4. Solve for correction via pseudoinverse of K.
        %     %      5. Update initial velocity guess and iterate until convergence.
        %     %
        %     %  Inputs:
        %     %    obj         : Object with CR3BP dynamics methods.
        %     %    X0_start    : Initial state [x y z vx vy vz] at t = 0.
        %     %    dV_guess    : Initial guess for ΔV at t = 0.
        %     %    X_target    : Target final state [x y z vx vy vz] at x = x_f.
        %     %    mu, n       : CR3BP parameters (mass ratio, mean motion).
        %     %    Tmax        : Max allowed propagation time.
        %     %    tolODE      : ODE solver tolerances (RelTol, AbsTol).
        %     %    tolErr      : Convergence tolerance on error norm.
        %     %    maxIter     : Max correction iterations.
        %     %
        %     %  Outputs:
        %     %    X0_optimized: Optimized initial state [x y z vx vy vz].
        %     %    dV0_opt     : Optimal ΔV applied at t = 0.
        %     %    dVf         : ΔV to match final velocity exactly.
        %     %    Ttraj       : Propagation time until x = x_f.
        %     %    exit_flag   :  1 = converged, -1 = no convergence.
        %     %    iter_count  : Number of iterations used.
        %     %    errHistory  : Error norm history per iteration.
        %     %
        %     % ===============================================================
        %
        %
        %
        %     % -------- initialization ----------------------------------------
        %     verbose = true;
        %     iter_count  = 0;
        %     exit_flag   = 0;
        %     errHistory  = [];                             % store norm of error in each loop
        %
        %     if isrow(X0_start),  X0_start = X0_start.'; end
        %     X0_start      = X0_start(1:6);
        %     rf_target     = X_target(1:3);                % final desired position
        %     vf_target     = X_target(4:6);                % final desired velocity
        %     X0_optimized        = X0_start;               % initialize optimal X0
        %     X0_optimized(4:6)   = X0_optimized(4:6) + dV_guess(:);
        %
        %     xf_tar  = rf_target(1);   yf_tar  = rf_target(2);   zf_tar  = rf_target(3);
        %     vxf_tar = vf_target(1);   vyf_tar = vf_target(2);   vzf_tar = vf_target(3);
        %
        %     options = odeset('Events',@eventStopAtXf,'RelTol',tolODE,'AbsTol',tolODE);
        %
        %
        %     % -------- differential-correction loop --------------------------
        %     while iter_count < maxIter
        %
        %         % propagate state and STM
        %         [T , Xaug] = ode89(@(t,X) obj.augmentedDynamicsCR3BP(t,X,mu,n), ...
        %             [0 Tmax] , [X0_optimized ; reshape(eye(6),36,1)] , ...
        %             options);
        %
        %         % extract final state and STM from propagation
        %         Xf     = Xaug(end,1:6).';
        %         Phi_tf = reshape(Xaug(end,7:end),6,6);
        %
        %         % compute y and z position errors
        %         posErr = [ yf_tar - Xf(2) ;
        %             zf_tar - Xf(3) ];
        %
        %         % compute vx, vy, vz errors
        %         velErr = [ vxf_tar - Xf(4) ;
        %             vyf_tar - Xf(5) ;
        %             vzf_tar - Xf(6) ];
        %
        %         % CR3BP scales (km, km/s)
        %         l_star_km  = 389703;
        %         v_star_kms = 1.01754797650856;
        %
        %         % convert to physical units
        %         dy_km   = posErr(1) * l_star_km;
        %         dz_km   = posErr(2) * l_star_km;
        %         dvx_kms = velErr(1) * v_star_kms;
        %         dvy_kms = velErr(2) * v_star_kms;
        %         dvz_kms = velErr(3) * v_star_kms;
        %
        %         posErrNorm_km = hypot(dy_km, dz_km);
        %         velErrNorm_kms = norm([dvx_kms; dvy_kms; dvz_kms]);
        %
        %         % keep nondimensional vector for solver bookkeeping
        %         errVec  = [posErr ; velErr];
        %         errNorm = norm(errVec);
        %         errHistory(end+1,1) = errNorm;
        %
        %         if verbose
        %             fprintf('iter %3d  pos [km]: dy=%.3e  dz=%.3e  (|pos|=%.3e)   vel [km/s]: dvx=%.3e  dvy=%.3e  dvz=%.3e  (|vel|=%.3e)\n', ...
        %                 iter_count, dy_km, dz_km, posErrNorm_km, dvx_kms, dvy_kms, dvz_kms, velErrNorm_kms);
        %         end
        %
        %         % --- convergence? ---------------------------------------------------
        %         if errNorm < tolErr
        %             exit_flag  = 1;
        %             Ttraj      = T(end);
        %             dVf        = -Xf(4:6);
        %             dV0_opt    = X0_optimized(4:6) - X0_start(4:6);
        %             return
        %         end
        %
        %         % --- build K matrix -------------------------------------------------
        %         vxf = Xf(4); vyf = Xf(5); vzf = Xf(6);
        %         a_tf = obj.dynamicsCR3BP(0,Xf,mu,n);   % dynamics at final position [vẋ vẏ vż] in 4:6
        %         axf = a_tf(4); ayf = a_tf(5); azf = a_tf(6);
        %
        %         K = [ Phi_tf(2,4)-Phi_tf(1,4)*(vyf/vxf) , Phi_tf(2,5)-Phi_tf(1,5)*(vyf/vxf) , Phi_tf(2,6)-Phi_tf(1,6)*(vyf/vxf) ;
        %             Phi_tf(3,4)-Phi_tf(1,4)*(vzf/vxf) , Phi_tf(3,5)-Phi_tf(1,5)*(vzf/vxf) , Phi_tf(3,6)-Phi_tf(1,6)*(vzf/vxf) ;
        %             Phi_tf(4,4)-Phi_tf(1,4)*(axf/vxf) , Phi_tf(4,5)-Phi_tf(1,5)*(axf/vxf) , Phi_tf(4,6)-Phi_tf(1,6)*(axf/vxf) ;
        %             Phi_tf(5,4)-Phi_tf(1,4)*(ayf/vxf) , Phi_tf(5,5)-Phi_tf(1,5)*(ayf/vxf) , Phi_tf(5,6)-Phi_tf(1,6)*(ayf/vxf) ;
        %             Phi_tf(6,4)-Phi_tf(1,4)*(azf/vxf) , Phi_tf(6,5)-Phi_tf(1,5)*(azf/vxf) , Phi_tf(6,6)-Phi_tf(1,6)*(azf/vxf) ];
        %
        %         warning('off','MATLAB:rankDeficientMatrix');
        %         dVcorr = pinv(K) * errVec;              % least-squares correction
        %         warning('on','MATLAB:rankDeficientMatrix');
        %
        %         X0_optimized(4:6) = X0_optimized(4:6) + dVcorr;
        %         iter_count        = iter_count + 1;
        %
        %
        %         if verbose
        %             fprintf('iter %2d |err| = %.3e   ‖ΔV‖=%.3e nd\n', ...
        %                 iter_count, errNorm, norm(dVcorr));
        %         end
        %     end
        %
        %     % -------- no convergence -----------------------------------------------
        %     exit_flag   = -1;
        %     X0_optimized= NaN(6,1); dV0_opt = NaN(1,3);
        %     dVf         = NaN(1,3); Ttraj = NaN;
        %
        %     % ---------- nested event -----------------------------------------------
        %     function [value,isterminal,direction] = eventStopAtXf(~,Xaug)
        %         value      = Xaug(1) - xf_tar;  % stop when x = x_target
        %         isterminal = 1;                 % halt
        %         direction  = 0;
        %     end
        % end
        %


        % ===============================================================
        %  Final-Position ( y , z)  targeter – stop at X = xf
        % ===============================================================
        function [X0_optimized, deltaV0opt, deltaVf, Ttraj, exit_flag, iter_count] = ...
                finalPositionTargeter_StopAtXf( ...
                obj,X0_start, dV_guess, rf_target, mu, n, Tmax, tolODE, tolError, maxIter)
            % used to be called finalStateTargeter_StopAtXf

            iter_count = 0;
            exit_flag = 0;

            % - - - - Earth Moon CR3BP Specific Quantities - - - - -
            l_star = 3.8475e5;                 %  Given total distance between primaries         [km]
            GM = 4.0350e5;                     % Earth-Moon barycenter GM   [km3/s2]
            t_star = sqrt(l_star^3 / GM);      % Characteristic time        [s]

            % - - - - Sanity Check of X0_guess Value Provided - - - -
            if ~isvector(X0_start)
                error('X0_guess must be a single vector of 6 or 7 elements, not a matrix.');
            end

            if isrow(X0_start)                                             % If X0_guess is a row vector
                X0_start = X0_start';                                      % Make it a column vector
            end

            if length(X0_start) < 6 || length(X0_start) > 7                % Check for number of elements in X0_guess
                error('X0_guess must contain exactly 6 or 7 elements.');
            end

            X0_start = X0_start(1:6);       % Grab first 6 elements only in case in contains the period

            % - - - Add the dV guess to the start state - - -
            dVx = dV_guess(1);
            dVy = dV_guess(2);
            dVz = dV_guess(3);

            xf_tar = rf_target(1);                          % Extract x component of target position
            yf_tar = rf_target(2);                          % Extract y component of target position
            zf_tar = rf_target(3);                          % Extract z component of target position

            X0_start = X0_start + [0;0;0;dVx;dVy;dVz];      % Add the provided deltaV to the start condition

            % - - - Initialize output vectors - - -
            X0_optimized = X0_start(1:6);            % First 6 elements only of X0 in case it included Tspan as part of state
            Ttraj = 0;                               % Initialize Topt

            % - - - - Set Up Options for ODE - - - -
            options = odeset('Events', @eventStopAtXf, 'RelTol', tolODE, 'AbsTol', tolODE);

            % - - - - Targeter Algorithm Begins - - - -
            while iter_count < maxIter              % Check if number of iterations has been exceeded

                % Propagate the state and STM from the initial
                % conditions guess
                [T, Xaug] = ode89(@(t, Xaug) obj.augmentedDynamicsCR3BP(t, Xaug, mu, n), [0, Tmax], [X0_optimized; reshape(eye(6), 36, 1)], options);

                Xf = Xaug(end, 1:6);                % Extract the final state of propagation, when y = 0
                xf = Xf(1);                         % Get ref trajectory position in x
                yf = Xf(2);                         % Get ref trajectory position in y
                zf = Xf(3);                         % Get ref trajectory position in z
                vxf = Xf(4);                        % Get velocity in x
                vyf = Xf(5);                        % Get velocity in y
                vzf = Xf(6);                        % Get velocity in z

                yf_error = yf_tar - yf;             % Calculate delta in position y
                zf_error = zf_tar - zf;             % Calculate delta in position z

                error = sqrt(yf_error^2 + zf_error^2);          % Position squared vector #CHECK

                Phi_mtx = reshape(Xaug(end, 7:end), 6, 6);      % Extract t0,tf STM matrix, showing xf/x0 sensitivity

                % Check for the error in zf, yf condition (small error
                % in both positions)
                if error < tolError                 % If error is below the requested value
                    exit_flag = 1;                  % Success flag!
                    Ttraj = T(end);                 % Final optimal propagation time
                    deltaVf = - [vxf, vyf, vzf];    % Final delta V value required is negative of final v
                    break;                          % The orbit is periodic and meets the crossing condition
                end

                % Extract necessary elements from the STM for the
                % correction
                phi14 = Phi_mtx(1, 4); phi15 = Phi_mtx(1, 5); phi16 = Phi_mtx(1, 6);
                phi24 = Phi_mtx(2, 4); phi25 = Phi_mtx(2, 5); phi26 = Phi_mtx(2, 6);
                phi34 = Phi_mtx(3, 4); phi35 = Phi_mtx(3, 5); phi36 = Phi_mtx(3, 6);

                yf_over_vxf = vyf / vxf;
                zf_over_vxf = vzf / vxf;

                % Calculate the matrix from the equation (not square)
                K = [phi24 - (phi14 * yf_over_vxf), phi25 - (phi15 * yf_over_vxf), phi26 - (phi16 * yf_over_vxf);
                    phi34 - (phi14 * zf_over_vxf), phi35 - (phi15 * zf_over_vxf), phi36 - (phi16 * zf_over_vxf)];

                % Error vector delta yf and delta zf
                deltaError = [yf_error ; zf_error];

                warning('off', 'MATLAB:rankDeficientMatrix');

                % Solve for deltaY0_dot and deltaZ0 correction =
                % K\deltaError;
                correction = pinv(K) * deltaError;


                deltaVx0 = correction(1);
                deltaVy0 = correction(2);
                deltaVz0 = correction(3);

                % Update initial conditions  and propagation time
                % #CHECK (not updating time for now)

                X0_optimized(4) = X0_optimized(4) + deltaVx0;        % Adjust initial position in Z
                X0_optimized(5) = X0_optimized(5) + deltaVy0;       % Adjust initial speed in Y
                X0_optimized(6) = X0_optimized(6) + deltaVz0;       % Adjust initial speed in Y

                iter_count = iter_count + 1;                        % Increase iteration count
            end

            if iter_count == maxIter
                fprintf('Maximum iterations reached without convergence. Check your initial conditions and guesses.\n');
                % Assign NaN to indicate non-convergence in time
                Ttraj = NaN;
                X0_optimized = NaN(6, 1);
                deltaV0opt = NaN;
                deltaVf = NaN; Ttraj = NaN;
                iter_count = NaN;
                exit_flag = -1;         % Indicate failure to converge
            else

                % Calculate optimized delta V in nondimensional units
                % [vx, vy, vz]
                deltaV0opt = [X0_optimized(4) - X0_start(4), X0_optimized(5) - X0_start(5), X0_optimized(6) - X0_start(6)];

                % fprintf('Convergence achieved after %d
                % iterations.\n', iter_count); fprintf('Optimized
                % Initial Conditions:\n'); fprintf('  dVx0 (nd) =
                % %.4f\n', deltaV0opt(1)); fprintf('  dVy0 (nd) =
                % %.4f\n', deltaV0opt(2)); fprintf('  dVz0 (nd) =
                % %.4f\n', deltaV0opt(3)); fprintf('Transfer Period
                % (Topt): %.4f (nd)\n', T(end));
            end

            % This is the event function required to stop the numerical
            % integration when x = xf_target

            function [value, isterminal, direction] = eventStopAtXf(t, Xaug)
                x = Xaug(1);              % x is the second element of the state vector
                value = x - xf_tar;       % When value is zero, an event is triggered
                isterminal = 1;           % Halt integration when the event is triggered
                direction = 0;            % The zero can be approached from either direction
            end
        end


        % ===============================================================
        %  Final-Velocity targeter – stop at location outside of a SOI
        % ===============================================================
        function [X0_optimized , dV0_opt , dVf , Ttraj , exit_flag , ...
                iter_count   , errHistory , iterLog , trajHistory , X0_hist , Xsoi_hist] = ...
                finalVelocityTargeter_OnSOI( ...
                obj, X0_start , dV_guess , v_target , mu , n , Rsoi_nd , ...
                Tmax , tolODE , tolErr , maxIter)
            % ===============================================================
            %  Velocity-only targeter at Earth SOI (direction + magnitude)
            %  - Propagate in EM-CR3BP until crossing the Earth SOI sphere.
            %  - Enforce hemisphere gate aligned with v_target direction.
            %  - Correct initial velocity using linear map to final velocity.
            %
            %  Inputs
            %    obj         : object exposing augmentedDynamicsCR3BP / dynamicsCR3BP
            %    X0_start    : initial state [x y z vx vy vz] (nd)
            %    dV_guess    : initial guess on ΔV at t0 (nd, 3x1 or 1x3)
            %    v_target    : desired final velocity at SOI (nd, 3x1 or 1x3)
            %    mu, n       : CR3BP parameters (mass ratio, mean motion)
            %    Rsoi_nd     : Earth SOI radius in normalized units
            %    Tmax        : maximum allowed propagation time
            %    tolODE      : ODE tolerances (scalar used for RelTol=AbsTol)
            %    tolErr      : convergence tolerance on ‖v_target - v_f‖
            %    maxIter     : maximum correction iterations
            %
            %  Outputs
            %    X0_optimized: optimized initial state [x y z vx vy vz]
            %    dV0_opt     : net ΔV applied at t0 (nd, 1x3)
            %    dVf         : residual ΔV at SOI (v_target - v_f) on exit
            %    Ttraj       : time of flight to SOI
            %    exit_flag   : 1 = converged, -1 = no convergence
            %    iter_count  : number of iterations performed
            %    errHistory  : ‖velocity error‖ per iteration
            %    iterLog     : [iter, tf, rmag_gap, hem_ok, dvx, dvy, dvz, ...
            %                   speed_err, ang_err_deg, errNorm, dVcorrNorm]
            %    trajHistory : cell of [x y z vx vy vz t] (downsampled)
            %    X0_hist     : [x0..vz0 t0] per iteration
            %    Xsoi_hist   : [x y z vx vy vz tf hit hem_ok rmag_gap] per iteration
            %
            %  Notes
            %   - Assumes standard EM-CR3BP synodic frame with Earth at r_E = [-mu,0,0].
            %   - Hemisphere gate: r_rel·v_hat_target ≥ 0.
            %   - Uses STM-based map: Ksoi = Φ_vv - (a_f r_fᵀ / (r_fᵀ v_f)) Φ_rv.
            % ===============================================================

            verbose     = false;
            iter_count  = 0;
            exit_flag   = 0;

            errHistory  = [];
            iterLog     = [];
            trajHistory = {};
            X0_hist     = [];
            Xsoi_hist   = [];

            if isrow(X0_start),  X0_start = X0_start.'; end
            if isrow(dV_guess),  dV_guess = dV_guess.'; end
            if isrow(v_target),  v_target = v_target.'; end

            % Earth location in normalized synodic coords (m1 at [-mu,0,0])
            rE = [-mu; 0; 0];

            % Normalize desired velocity direction for hemisphere gate
            vhat_req = v_target / max(norm(v_target), eps);

            % Initialize the iterate
            X0_optimized      = X0_start(1:6);
            X0_optimized(4:6) = X0_optimized(4:6) + dV_guess(:);

            % ODE options with SOI event and a small MaxStep to avoid root skipping
            options = odeset('Events',@eventStopAtSOI,'RelTol',tolODE,'AbsTol',tolODE, ...
                'MaxStep', Tmax/2000);

            % ===== Iteration loop =================================================
            while iter_count < maxIter

                % Log the starting state for this iteration
                X0_hist(end+1,:) = [X0_optimized(:).'  0];

                % Propagate state + STM (6 + 36)
                [T, Xaug, TE, YE, ~] = ode89(@(t,X) obj.augmentedDynamicsCR3BP(t,X,mu,n), ...
                    [0 Tmax] , [X0_optimized ; reshape(eye(6),36,1)] , ...
                    options);

                % Downsample trajectory for history (≤500 pts)
                if numel(T) > 500
                    idx = round(linspace(1,numel(T),500));
                else
                    idx = 1:numel(T);
                end
                trajHistory{end+1,1} = [ Xaug(idx,1:6) , T(idx) ];

                % Extract final state and STM (prefer event)
                hit = ~isempty(TE);
                if hit
                    tf     = TE(end);
                    Xaug_f = YE(end,:);
                else
                    tf     = T(end);
                    Xaug_f = Xaug(end,:);
                end
                Xf     = Xaug_f(1:6).';
                Phi_tf = reshape(Xaug_f(7:end),6,6);

                rf     = Xf(1:3);       % barycentric position
                vf     = Xf(4:6);
                rf_rel = rf - rE;       % Earth-centered position for SOI event
                af     = obj.dynamicsCR3BP(0,Xf,mu,n);  % returns [ẋ ẏ ż v̇x v̇y v̇z]
                af     = af(4:6);

                % SOI diagnostics
                rmag_gap     = norm(rf_rel) - Rsoi_nd;
                hem_ok       = double(dot(rf_rel, vhat_req) >= 0);

                % --- Build K_soi = Φ_vv - (a_f r_f^T / (r_f^T v_f)) Φ_rv ------
                Phi_rv = Phi_tf(1:3,4:6);
                Phi_vv = Phi_tf(4:6,4:6);
                den    = dot(rf_rel, vf);
                if abs(den) < 1e-10
                    den = sign(den + (den==0))*1e-10;  % regularize near-tangent condition
                end

                Ksoi = Phi_vv - (af * (rf_rel.'))/den * Phi_rv;

                % --- Velocity error (target - actual) --------------------------
                velErr  = (v_target - vf);
                errNorm = norm(velErr);
                errHistory(end+1,1) = errNorm;

                % Log the SOI hit state for this iteration
                Xsoi_hist(end+1,:) = [Xf(:).'  tf  double(hit)  hem_ok  rmag_gap];

                % Extra diagnostics (speed and angle errors)
                spd_err     = norm(vf) - norm(v_target);
                cosang      = dot(vf, v_target) / (max(norm(vf),eps)*max(norm(v_target),eps));
                ang_err_deg = real(acosd(max(-1,min(1,cosang))));

                if verbose
                    fprintf('iter %3d  tf=%.6f  hit=%d  hem=%d  rgap=%.3e  |dV|=%.3e  spd_err=%.3e  ang=%.2f°\n', ...
                        iter_count, tf, hit, hem_ok, rmag_gap, errNorm, spd_err, ang_err_deg);
                end

                % --- Convergence check -----------------------------------------
                if errNorm < tolErr && hit && hem_ok==1
                    exit_flag  = 1;
                    Ttraj      = tf;
                    dVf        = (v_target - vf);                 % residual at SOI (should be ~0)
                    dV0_opt    = (X0_optimized(4:6) - X0_start(4:6)).';  % row 1x3
                    iterLog(end+1,:) = [iter_count, tf, rmag_gap, hem_ok, ...
                        velErr(:).', spd_err, ang_err_deg, errNorm, NaN];
                    return
                end

                % --- Solve least squares for ΔV correction ---------------------
                dVcorr = pinv(Ksoi) * velErr;

                % Apply and log
                X0_optimized(4:6) = X0_optimized(4:6) + dVcorr;
                iterLog(end+1,:)  = [iter_count, tf, rmag_gap, hem_ok, ...
                    velErr(:).', spd_err, ang_err_deg, errNorm, norm(dVcorr)];

                iter_count = iter_count + 1;

                if verbose
                    fprintf('iter %2d  ||err||=%.3e   ||ΔVcorr||=%.3e nd\n', iter_count, errNorm, norm(dVcorr));
                end
            end

            % ===== No convergence ===============================================
            exit_flag    = -1;
            X0_optimized = NaN(6,1);
            dV0_opt      = [NaN NaN NaN];
            dVf          = [NaN NaN NaN].';
            Ttraj        = NaN;

            % ---------- nested event: SOI sphere with hemisphere gate -----------
            function [value, isterminal, direction] = eventStopAtSOI(~,Xaug)
                r  = Xaug(1:3);
                v  = Xaug(4:6);
                rr = r - rE;                        % Earth-centered
                value      = norm(rr) - Rsoi_nd;    % sphere
                % Gate: only terminate if on hemisphere aligned with vhat_req
                isterminal = double( dot(rr, vhat_req) >= 0 );   % 1(stop) or 0(ignore)
                direction  = 0;                     % any crossing
            end
        end


        % ===============================================================
        %  Final-Velocity targeter – stop at location outside of a SOI (Wrapper)
        % ===============================================================
        function shot = runSOIVelocityShot(obj, cfg)
            %RUNSOIVELOCITYSHOT  One-call SOI velocity targeting run.
            % This is just a wrapper function for the targeter.
            %
            % Required cfg fields:
            %   obj.orb              : KeplerianOrbitalMechanicsLibrary instance
            %   cfg.Xpre_nd          : 6x1 pre-burn CR3BP state in synodic (nd)
            %   cfg.theta0           : synodic frame angle at burn epoch (rad)
            %   cfg.V_dep_hci_kms    : 1x3 desired heliocentric S/C velocity at SOI (km/s)
            %   cfg.rE0_km, cfg.vE0_km : Earth HCI state at burn epoch (km, km/s)
            %   cfg.mu, cfg.n        : CR3BP parameters
            %   cfg.Earth_soi_nd     : SOI radius (nd)
            %   cfg.l_star, cfg.v_star : CR3BP scales
            %
            % Optional cfg fields (defaults shown):
            %   cfg.Tmax_nd  (10*86400/(l_star/v_star))
            %   cfg.tolODE   (1e-10)
            %   cfg.tolErr   (1e-10)
            %   cfg.maxIter  (50)
            %   cfg.factorSOI(1.1)

            % ---- required fields check
            req = ["Xpre_nd","theta0","V_dep_hci_kms","rE0_km","vE0_km", ...
                "mu","n","Earth_soi_nd","l_star","v_star"];
            for f = req
                assert(isfield(cfg,f), "runSOIVelocityShot: missing cfg.%s", f);
            end

            % ---- defaults
            if ~isfield(cfg,'Tmax_nd'),   cfg.Tmax_nd   = 10*86400/(cfg.l_star/cfg.v_star); end
            if ~isfield(cfg,'tolODE'),    cfg.tolODE    = 1e-12; end
            if ~isfield(cfg,'tolErr'),    cfg.tolErr    = 1e-12; end
            if ~isfield(cfg,'maxIter'),   cfg.maxIter   = 50;    end
            if ~isfield(cfg,'factorSOI'), cfg.factorSOI = 1.0;   end

            % ---- SOI point aligned with desired HCI velocity (visual + ω×r term)
            vhat_HCI  = cfg.V_dep_hci_kms / norm(cfg.V_dep_hci_kms);

            % We are placing the desired velocity at a point in the SOI along its direction
            rSOI_HCI  = cfg.rE0_km + (cfg.Earth_soi_nd*cfg.l_star) * vhat_HCI;  % [km] # check

            % HCI -> ECI: make a target *state* at SOI (needs position)
            X_dep_ECI = obj.orb.hci2eci_knownEarthICs([rSOI_HCI cfg.V_dep_hci_kms], ...
                cfg.rE0_km, cfg.vE0_km);    % [rSOI v]_ECI

            % ECI -> syn (nd) to get the correct rotating-frame velocity target
            X_tar_syn  = obj.bci_to_syn([X_dep_ECI(1:3)/cfg.l_star, X_dep_ECI(4:6)/cfg.v_star], ...
                cfg.theta0, cfg.mu, 'primary');

            v_target_nd = X_tar_syn(4:6).';   % 3x1

            % Initial ΔV guess from patched conics: V∞ (ECI) -> syn (nd)
            Vinf_eci_kms = cfg.V_dep_hci_kms - cfg.vE0_km;                % km/s
            Xinf_syn     = obj.bci_to_syn([0 0 0 (Vinf_eci_kms/cfg.v_star)], ...
                cfg.theta0, cfg.mu, 'primary'); % r=0 at Earth
            dv_guess_nd  = Xinf_syn(4:6).';

            % ---- Call SOI velocity targeter
            [X0_opt, dV0_opt_nd, dVf, Ttraj_SOI, flag, iters, errHist, ...
                iterLog, trajHist, X0_hist, Xsoi_hist] = ...
                obj.finalVelocityTargeter_OnSOI( ...
                cfg.Xpre_nd, dv_guess_nd, v_target_nd, ...
                cfg.mu, cfg.n, cfg.factorSOI*cfg.Earth_soi_nd, ...
                cfg.Tmax_nd, cfg.tolODE, cfg.tolErr, cfg.maxIter );

            % ---- Pack scalar struct (force row vectors for convenience)
            shot = struct();
            shot.X0_opt       = X0_opt;
            shot.dV0_opt_nd   = dV0_opt_nd(:).';
            shot.dVf          = dVf(:).';
            shot.Ttraj_SOI    = Ttraj_SOI;
            shot.flag         = flag;
            shot.iters        = iters;
            shot.iterLog      = iterLog;
            shot.trajHist     = trajHist;
            shot.X0_hist      = X0_hist;
            shot.Xsoi_hist    = Xsoi_hist;
            shot.v_target_nd  = v_target_nd(:).';
            shot.dv_guess_nd  = dv_guess_nd(:).';
            shot.errHist      = errHist(:);
        end




        %% ===============================================================
        %% 3.5) Targeters to Find Single Orbits

        % ======================================================================
        % Perpendicular-crossing corrector (vary vy0 only → enforce vx_f ≈ 0)
        % ======================================================================

        %% 3.5.1) Planar Perpendicular - Lyapunov
        function [Xcorr_nd, T_half_nd, T_full_nd, iters] = ...
                 perpTargeterVyOnly(obj, X0_nd, mu, n, vstar_km_s, tspan, tol_vx_kms, max_iter)
            % PERPTARGETERVYONLY
            % Purpose: adjust initial vy0 so the first y=0 crossing is perpendicular,
            % i.e., vx_f ≈ 0 at the crossing (planar Lyapunov seed generator).
            % Inputs:
            %   X0_nd(6×1)  : initial guess [x y z vx vy vz] (nd)
            %   mu, n       : CR3BP params (nd)
            %   vstar_km_s  : characteristic velocity (km/s) for reporting tolerance
            %   tspan       : [t0 tfmax] integration window (nd) for initial
            %   tol_vx_kms  : stop when |vx_f|*v* < tol_vx_kms (km/s)
            %   max_iter    : max Newton iterations
            % Outputs:
            %   Xcorr_nd    : corrected IC (only vy0 changed)
            %   T_half_nd   : time to the first y=0 crossing (nd)
            %   T_full_nd   : full period estimate (=2*T_half_nd) (nd)
            %   iters       : iterations used

            opts_evt = odeset('RelTol',1e-13,'AbsTol',1e-13,'Events',@(t,X) eventYeq0(t,X));


            Phi0     = eye(6);

            vx0 = X0_nd(4);
            vy0 = X0_nd(5);

            tf = NaN;  % guard for non-convergence path

            for k = 1:max_iter
                X0k   = X0_nd;  
                X0k(4) = vx0; 
                X0k(5) = vy0;
                Xaug0 = [X0k; Phi0(:)];

                [~, Yaug, te, Ye] = ode89(@(t,X) obj.augmentedDynamicsCR3BP(t,X,mu,n), tspan, Xaug0, opts_evt);
                tf = te(end);
                Xf = Ye(end,1:6).';             % state at y=0 (x axis crossing)
                vx_f = Xf(4);                   % nd
                vy_f = Xf(5);                   % nd

                % STM and sensitivity s = d(vx_f)/d(vy0) (event-time corrected)
                Phi6 = reshape(Yaug(end,7:end), 6, 6);

                % planar block rows/cols [x y vx vy] = [1 2 4 5]
                Phi4 = Phi6([1 2 4 5],[1 2 4 5]);
                ax_f = obj.compute_ax_nd(Xf, mu, n); % nd
                s    = Phi4(3,4) - (ax_f / vy_f) * Phi4(2,4);
                               
                % Newton update on vy0, correction
                dvy0 = - vx_f / s;
                vy0  = vy0 + dvy0;              

                % Convergence in km/s
                if abs(vx_f * vstar_km_s) < tol_vx_kms
                    Xcorr_nd   = X0_nd; Xcorr_nd(5) = vy0;
                    T_half_nd  = tf;
                    T_full_nd  = 2*tf;
                    iters      = k;
                    return;
                end
            end

            % Not converged: return best found
            Xcorr_nd   = X0_nd; Xcorr_nd(5) = vy0;
                        
            T_half_nd  = tf;
            T_full_nd  = 2*tf;
            iters      = max_iter;

            function [value,isterminal,direction] = eventYeq0(t,X)
                value = X(2); 
                isterminal = 1; 
                direction = 0;
                % if t <= 1e-10, value = 1; end   % ignore the initial y=0
            end

        end


        %% 3.5.2) Non-Planar Perpendicular - Halo like

        % Perpendicular - Non Planar - XZ Plane Targeter Function - Fixed X0
        function [X0_optimized, Topt, exit_flag, iter_count] = targeterPerpendicularXZ_FixedX(obj,X0_guess, mu, n, Tmax, tolODE, tolError, maxIter)
            % This function finds a periodic orbit within the Circular
            % Restricted Three-Body Problem (CR3BP) that intersects the
            % xz-plane perpendicularly, with a fixed x-coordinate. It
            % iteratively adjusts the initial state vector to satisfy the
            % perpendicular crossing condition, within a specified tolerance
            % for error.
            %
            % Inputs:
            %   X0_guess: Initial guess for the state vector [x, 0, z, 0, vy,
            %   vz] as either a row or column vector. mu: Gravitational
            %   parameter of the CR3BP system, representing the mass ratio of
            %   the two primary bodies. n: Mean motion (average angular
            %   velocity) of the system. Usually set to 1 in normalized
            %   units. Tmax: Maximum simulation time for orbit propagation.
            %   tolODE: Tolerance for the ODE solver. tolError: Tolerance for
            %   the error in meeting the perpendicular crossing condition.
            %   maxIter: Maximum number of iterations to attempt for
            %   convergence.
            %
            % Outputs:
            %   X0_optimized: Optimized initial conditions for achieving a
            %   periodic orbit. Topt: Optimal time for one complete orbit,
            %   indicating the period. exit_flag: Indicator of success (1) or
            %   failure (0) in finding a periodic orbit. iter_count: The
            %   number of iterations performed before termination.


            iter_count = 0;
            exit_flag = 0;

            % - - - - Sanity Check of X0_guess Value Provided - - - -
            if ~isvector(X0_guess)
                error('X0_guess must be a single vector of 6 or 7 elements, not a matrix.');
            end

            if isrow(X0_guess)                                             % If X0_guess is a row vector
                X0_guess = X0_guess';                                      % Make it a column vector
            end

            if length(X0_guess) < 6 || length(X0_guess) > 7                % Check for number of elements in X0_guess
                error('X0_guess must contain exactly 6 or 7 elements.');
            end

            X0_optimized = X0_guess(1:6);           % First 6 elements only of X0 in case it includes Tspan
            Topt = 0;                               % Initialize Topt

            % - - - - Set Up Options for ODE - - - -
            options = odeset('Events', @eventStopZeroY, 'RelTol', tolODE, 'AbsTol', tolODE);

            % - - - - Targeter Algorithm Begins - - - -
            while iter_count < maxIter              % Check if number of iterations has been exceeded
                % Propagate the state and STM from the initial conditions
                % guess (ref trajectory) STM initialized as an identity
                % matrix 6x6
                [T, Xaug] = ode89(@(t, Xaug) obj.augmentedDynamicsCR3BP(t, Xaug, mu), [0, Tmax], [X0_optimized; reshape(eye(6), 36, 1)], options);

                Xf = Xaug(end, 1:6);                % Extract the final state of propagation, when y = 0
                vxf = Xf(4);                        % Get velocity in x
                vyf = Xf(5);                        % Get velocity in y
                vzf = Xf(6);                        % Get velocity in z
                error = sqrt(vxf^2+vzf^2);          % Velocity squared vector

                Phi_mtx = reshape(Xaug(end, 7:end), 6, 6); % Extract t0,tf STM matrix, showing xf/x0 sensitivity

                % Check for the perpendicular crossing condition (yf = 0,
                % vxf = 0, vzf = 0)
                if error < tolError                 % If error is below the requested value
                    exit_flag = 1;                  % Success flag!
                    Topt = 2 * T(end);              % Final optimal time is twice the one found
                    break;                          % The orbit is periodic and meets the crossing condition
                end

                dXfdt = obj.dynamicsCR3BP(obj, Xf, mu, n);   % Calculate dynamics at tf [vx, vy, vz, ax, ay, az]
                axf = dXfdt(4);                     % Get acceleration in x
                azf = dXfdt(6);                     % Get acceleration in z

                % Extract necessary elements from the STM for the
                % correction
                phi23 = Phi_mtx(2, 3); phi25 = Phi_mtx(2, 5); phi43 = Phi_mtx(4, 3);
                phi45 = Phi_mtx(4, 5); phi63 = Phi_mtx(6, 3); phi65 = Phi_mtx(6, 5);

                % Calculate the matrix from the equation (not square)
                K = [phi43 phi45; phi63 phi65]- 1/vyf * [axf;azf] * [phi23 phi25];

                % Calculate errors deltaVx and deltaVz at the crossing
                deltaVx = -vxf;                      % Opposite sign so it cancels out
                deltaVz = -vzf;                      % Opposite sign so it cancels out
                deltaError = [deltaVx ; deltaVz];

                % Solve for deltaY0_dot and deltaZ0
                correction =  K\deltaError;
                deltaZ0 = correction(1);
                deltaVy0 = correction(2);

                % Update initial conditions and propagation time
                X0_optimized(3) = X0_optimized(3) + deltaZ0;        % Adjust initial position in Z
                X0_optimized(5) = X0_optimized(5) + deltaVy0;       % Adjust initial speed in Y

                iter_count = iter_count + 1;                        % Increase iteration count
            end

            if iter_count == maxIter
                disp('Maximum iterations reached without convergence.');
            else
                fprintf('Convergence achieved after %d iterations.\n', iter_count);
                fprintf('Optimized Initial Conditions:\n');
                fprintf('  x0 = %.4f\n', X0_optimized(1));
                fprintf('  z0 = %.4f\n', X0_optimized(3));
                fprintf('  vy0 = %.4f\n', X0_optimized(5));
                fprintf('Period (Topt): %.4f (nd)\n', 2 * T(end));
            end

            % This is the event function required to stop the numerical
            % integration at y=0
            function [value, isterminal, direction] = eventStopZeroY(t, Xaug)
                y = Xaug(2);     % y is the second element of the state vector
                value = y;       % When value is zero, an event is triggered
                isterminal = 1;  % Halt integration when the event is triggered
                direction = 0;   % The zero can be approached from either direction
            end
        end

        % Perpendicular - XZ Plane Targeter Function - Fixed Z0
        function [X0_optimized, Topt, exit_flag, iter_count] = targeterPerpendicularXZ_FixedZ(obj,X0_guess, mu, n, Tmax, tolODE, tolError, maxIter)
            % This function finds a periodic orbit within the Circular
            % Restricted Three-Body Problem (CR3BP) that intersects the
            % xz-plane perpendicularly, with a fixed z-coordinate. It
            % iteratively adjusts the initial state vector to satisfy the
            % perpendicular crossing condition, within a specified tolerance
            % for error.
            %
            % Inputs:
            %   X0_guess: Initial guess for the state vector [x, 0, z, 0, vy,
            %   vz] as either a row or column vector. mu: Gravitational
            %   parameter of the CR3BP system, representing the mass ratio of
            %   the two primary bodies. n: Mean motion (average angular
            %   velocity) of the system. Usually set to 1 in normalized
            %   units. Tmax: Maximum simulation time for orbit propagation.
            %   tolODE: Tolerance for the ODE solver. tolError: Tolerance for
            %   the error in meeting the perpendicular crossing condition.
            %   maxIter: Maximum number of iterations to attempt for
            %   convergence.
            %
            % Outputs:
            %   X0_optimized: Optimized initial conditions for achieving a
            %   periodic orbit. Topt: Optimal time for one complete orbit,
            %   indicating the period. exit_flag: Indicator of success (1) or
            %   failure (0) in finding a periodic orbit. iter_count: The
            %   number of iterations performed before termination.

            iter_count = 0;
            exit_flag = 0;

            % - - - - Sanity Check of X0_guess Value Provided - - - -
            if ~isvector(X0_guess)
                error('X0_guess must be a single vector of 6 or 7 elements, not a matrix.');
            end

            if isrow(X0_guess)                                             % If X0_guess is a row vector
                X0_guess = X0_guess';                                      % Make it a column vector
            end

            if length(X0_guess) < 6 || length(X0_guess) > 7                % Check for number of elements in X0_guess
                error('X0_guess must contain exactly 6 or 7 elements.');
            end

            X0_optimized = X0_guess(1:6);           % First 6 elements only of X0 in case it includes Tspan
            Topt = 0;                               % Initialize Topt

            % - - - - Set Up Options for ODE - - - -
            options = odeset('Events', @eventStopZeroY, 'RelTol', tolODE, 'AbsTol', tolODE);

            % - - - - Targeter Algorithm Begins - - - -
            while iter_count < maxIter              % Check if number of iterations has been exceeded
                % Propagate the state and STM from the initial conditions
                % guess
                [T, Xaug] = ode89(@(t, Xaug) obj.augmentedDynamicsCR3BP(t, Xaug, mu), [0, Tmax], [X0_optimized; reshape(eye(6), 36, 1)], options);

                Xf = Xaug(end, 1:6);                % Extract the final state of propagation, when y = 0
                vxf = Xf(4);                        % Get velocity in x
                vyf = Xf(5);                        % Get velocity in y
                vzf = Xf(6);                        % Get velocity in z
                error = sqrt(vxf^2+vzf^2);          % Velocity squared vector

                Phi_mtx = reshape(Xaug(end, 7:end), 6, 6); % Extract t0,tf STM matrix, showing xf/x0 sensitivity

                % Check for the perpendicular crossing condition (yf = 0,
                % vxf = 0, vzf = 0)
                if error < tolError                 % If error is below the requested value
                    exit_flag = 1;                  % Success flag!
                    Topt = 2 * T(end);              % Final optimal time is twice the one found
                    break;                          % The orbit is periodic and meets the crossing condition
                end

                dXfdt = obj.dynamicsCR3BP(obj, Xf, mu, n);   % Calculate dynamics at tf [vx, vy, vz, ax, ay, az]
                axf = dXfdt(4);
                azf = dXfdt(6);

                % Extract necessary elements from the STM for the
                % correction
                phi41 = Phi_mtx(4, 1); phi45 = Phi_mtx(4, 5); phi61 = Phi_mtx(6, 1);
                phi65 = Phi_mtx(6, 5); phi21 = Phi_mtx(2, 1); phi25 = Phi_mtx(2, 5);

                K = [phi41 phi45; phi61 phi65]- 1/vyf * [axf;azf] * [phi21 phi25];

                % Calculate errors deltaVx and deltaVz at the crossing
                deltaVx = -vxf;                      % Opposite sign so it cancels out
                deltaVz = -vzf;                      % Opposite sign so it cancels out
                deltaError = [deltaVx ; deltaVz];

                % Solve for deltaY0_dot and deltaX0
                correction =  K\deltaError;
                deltaX0 = correction(1);
                deltaVy0 = correction(2);

                % Update initial conditions and propagation time
                X0_optimized(1) = X0_optimized(1) + deltaX0;        % Adjust initial position in X
                X0_optimized(5) = X0_optimized(5) + deltaVy0;       % Adjust initial speed in Y

                iter_count = iter_count + 1;                        % Increase iteration count
            end

            if iter_count == maxIter
                fprintf('Maximum iterations reached without convergence. Could not find any periodic orbits given those initial conditions guess');
            else
                fprintf('Convergence achieved after %d iterations.\n', iter_count);
                fprintf('Optimized Initial Conditions:\n');
                fprintf('  x0 = %.4f\n', X0_optimized(1));
                fprintf('  z0 = %.4f\n', X0_optimized(3));
                fprintf('  vy0 = %.4f\n', X0_optimized(5));
                fprintf('Period (Topt): %.4f (nd)\n', 2 * T(end));
            end

            % This is the event function required to stop the numerical
            % integration at y=0
            function [value, isterminal, direction] = eventStopZeroY(t, Xaug)
                y = Xaug(2);     % y is the second element of the state vector
                value = y;       % When value is zero, an event is triggered
                isterminal = 1;  % Halt integration when the event is triggered
                direction = 0;   % The zero can be approached from either direction
            end
        end

        % Perpendicular Targeter Function - Fixed Vy0
        function [X0_optimized, Topt, exit_flag, iter_count] = targeterPerpendicularXZ_FixedVy(obj,X0_guess, mu, n, Tmax, tolODE, tolError, maxIter)
            % This function finds a periodic orbit within the Circular
            % Restricted Three-Body Problem (CR3BP) that intersects the
            % xz-plane perpendicularly, with a fixed vy value It
            % iteratively adjusts the initial state vector to satisfy the
            % perpendicular crossing condition, within a specified
            % tolerance for error.
            %
            % Inputs:
            %   X0_guess: Initial guess for the state vector [x, 0, z, 0,
            %   vy, vz] as either a row or column vector. mu:
            %   Gravitational parameter of the CR3BP system, representing
            %   the mass ratio of the two primary bodies. n: Mean motion
            %   (average angular velocity) of the system. Usually set to
            %   1 in normalized units. Tmax: Maximum simulation time for
            %   orbit propagation. tolODE: Tolerance for the ODE solver.
            %   tolError: Tolerance for the error in meeting the
            %   perpendicular crossing condition. maxIter: Maximum number
            %   of iterations to attempt for convergence.
            %
            % Outputs:
            %   X0_optimized: Optimized initial conditions for achieving
            %   a periodic orbit. Topt: Optimal time for one complete
            %   orbit, indicating the period. exit_flag: Indicator of
            %   success (1) or failure (0) in finding a periodic orbit.
            %   iter_count: The number of iterations performed before
            %   termination.


            iter_count = 0;
            exit_flag = 0;

            % - - - - Sanity Check of X0_guess Value Provided - - - -
            if ~isvector(X0_guess)
                error('X0_guess must be a single vector of 6 or 7 elements, not a matrix.');
            end

            if isrow(X0_guess)                                             % If X0_guess is a row vector
                X0_guess = X0_guess';                                      % Make it a column vector
            end

            if length(X0_guess) < 6 || length(X0_guess) > 7                % Check for number of elements in X0_guess
                error('X0_guess must contain exactly 6 or 7 elements.');
            end

            X0_optimized = X0_guess(1:6);           % First 6 elements only of X0 in case it includes Tspan
            Topt = 0;                               % Initialize Topt

            % - - - - Set Up Options for ODE - - - -
            options = odeset('Events', @eventStopZeroY, 'RelTol', tolODE, 'AbsTol', tolODE);

            % - - - - Targeter Algorithm Begins - - - -
            while iter_count < maxIter              % Check if number of iterations has been exceeded
                % Propagate the state and STM from the initial
                % conditions guess
                [T, Xaug] = ode89(@(t, Xaug) obj.augmentedDynamicsCR3BP(t, Xaug, mu), [0, Tmax], [X0_optimized; reshape(eye(6), 36, 1)], options);

                Xf = Xaug(end, 1:6);                % Extract the final state of propagation, when y = 0
                vxf = Xf(4);                        % Get velocity in x
                vyf = Xf(5);                        % Get velocity in y
                vzf = Xf(6);                        % Get velocity in z
                error = sqrt(vxf^2+vzf^2);          % Velocity squared vector

                Phi_mtx = reshape(Xaug(end, 7:end), 6, 6); % Extract t0,tf STM matrix, showing xf/x0 sensitivity

                % Check for the perpendicular crossing condition (yf =
                % 0, vxf = 0, vzf = 0)
                if error < tolError                 % If error is below the requested value
                    exit_flag = 1;                  % Success flag!
                    Topt = 2 * T(end);              % Final optimal time is twice the one found
                    break;                          % The orbit is periodic and meets the crossing condition
                end

                dXfdt = obj.dynamicsCR3BP(obj, Xf, mu, n);   % Calculate dynamics at tf [vx, vy, vz, ax, ay, az]
                axf = dXfdt(4);                     % Get acceleration in x
                azf = dXfdt(6);                     % Get acceleration in z

                % Extract necessary elements from the STM for the
                % correction
                phi21 = Phi_mtx(2, 1); phi23 = Phi_mtx(2, 3);
                phi41 = Phi_mtx(4, 1); phi43 = Phi_mtx(4, 3);
                phi61 = Phi_mtx(6, 1); phi63 = Phi_mtx(6, 3);

                % Calculate the matrix from the equation (not square)
                K = [phi41 phi43; phi61 phi63]- 1/vyf * [axf;azf] * [phi21 phi23];

                % Calculate errors deltaVx and deltaVz at the crossing
                deltaVx = -vxf;                      % Opposite sign so it cancels out
                deltaVz = -vzf;                      % Opposite sign so it cancels out
                deltaError = [deltaVx ; deltaVz];

                % Solve for deltaY0_dot and deltaZ0
                correction =  K\deltaError;
                deltaX0 = correction(1);
                deltaZ0 = correction(2);

                % Update initial conditions and propagation time
                X0_optimized(1) = X0_optimized(1) + deltaX0;        % Adjust initial position in X
                X0_optimized(3) = X0_optimized(3) + deltaZ0;        % Adjust initial position in Z

                iter_count = iter_count + 1;                        % Increase iteration count
            end

            if iter_count == maxIter
                disp('Maximum iterations reached without convergence.');
            else
                fprintf('Convergence achieved after %d iterations.\n', iter_count);
                fprintf('Optimized Initial Conditions:\n');
                fprintf('  x0 = %.4f\n', X0_optimized(1));
                fprintf('  z0 = %.4f\n', X0_optimized(3));
                fprintf('  vy0 = %.4f\n', X0_optimized(5));
                fprintf('Period (Topt): %.4f (nd)\n', 2 * T(end));
            end

            % This is the event function required to stop the numerical
            % integration at y=0
            function [value, isterminal, direction] = eventStopZeroY(t, Xaug)
                y = Xaug(2);     % y is the second element of the state vector
                value = y;       % When value is zero, an event is triggered
                isterminal = 1;  % Halt integration when the event is triggered
                direction = 0;   % The zero can be approached from either direction
            end
        end


        %% 3.5.2 - Auxiliary functions

        % ======================================================================
        % Utility: ax-component of acceleration at state (nd units)
        % ======================================================================

        function ax = compute_ax_nd(obj, X_nd, mu, n)
            % COMPUTE_AX_ND  Return dvx/dt at X_nd in rotating frame (nd).
            dxdt = obj.dynamicsCR3BP(0, X_nd(:), mu, n); % [vx vy vz dvx dvy dvz]
            ax   = dxdt(4);
        end




        %% ===============================================================
        %% 3.6) Continuation Method to Find Families

        %% 3.6.1) Natural Parameter Continuation

        % ======================================================================
        % Natural-parameter continuation in x0 with Polynomial fitting
        % ======================================================================
        function [ICs_family, iters_vec] = lyapunovOrbits_NP_Polyfit(obj, IC_seed, t2, N, beta, eps_nd, mu, n, vstar_km_s, deg)
            % NP_CONTINUATION_POLYFIT_CTBP
            % Build a Lyapunov family by stepping x0 and predicting vy0 via polyfit.
            % Inputs:
            %   IC_seed : 1x6 or 6x1 seed [x0 y0 z0 vx0 vy0 vz0] (nd)
            %   t2      : max time to first y=0 crossing (nd)
            %   N       : number of family members
            %   beta    : step in x0 per member (nd)
            %   eps_nd  : |vx_f| tolerance (nd) used to form km/s tol
            %   mu,n    : CR3BP params (nd)
            %   vstar_km_s : characteristic velocity (km/s)
            %   deg     : polyfit degree (default 2)
            % Outputs:
            %   ICs_family : [N x 7] rows = [x y z vx vy vz T]
            %   iters_vec  : [N x 1] Newton iterations per orbit
            if nargin < 11 || isempty(deg), deg = 2; end
            IC_seed = IC_seed(:);
            ICs_family = zeros(N,7);
            iters_vec  = zeros(N,1);

            tol_vx_kms = abs(eps_nd)*vstar_km_s;
            max_iter   = 50;
            tspan_evt  = [0, t2];

            x_hist  = [];
            vy_hist = [];

            for k = 1:N
                x0_try = IC_seed(1) + (k-1)*beta;

                if k == 1
                    vy0_guess = IC_seed(5);
                elseif numel(x_hist) >= deg+1
                    vy0_guess = obj.vy0PredictPolyfit(x_hist(end-deg:end), vy_hist(end-deg:end), x0_try, deg);
                else
                    vy0_guess = vy_hist(end);
                end

                X0_guess = [x0_try; 0; 0; 0; vy0_guess; 0];
                [Xcorr, T_half_nd, T_full_nd, iters] = obj.perpTargeterVyOnly( ...
                    X0_guess, mu, n, vstar_km_s, tspan_evt, tol_vx_kms, max_iter);

                ICs_family(k,:) = [Xcorr(:).', T_full_nd];
                iters_vec(k)    = iters;

                x_hist(end+1,1)  = Xcorr(1);
                vy_hist(end+1,1) = Xcorr(5);
            end
        end


        function [ICs_family, iters_vec] = lyapunovOrbits_NP_Continuation(obj, IC, t_2, N, beta, eps_nd, mu, n, vstar_km_s)
            % NATURAL_PARAMETER_CONTINUATION
            % Purpose: march along a periodic family by stepping x0 and correcting vy0
            % so that the trajectory crosses y=0 perpendicularly (vx_f≈0).
            % Inputs:
            %   IC(1×6)     : seed state [x y z vx vy vz] (nd). Typically planar: [x 0 0 0 vy 0].
            %   t_2         : max integration time to reach the first y=0 crossing (nd)
            %   N           : number of family members to generate
            %   beta        : step size in x0 between successive family members (nd)
            %   eps_nd      : desired |vx_f| tolerance in nd units (converted to km/s with v*)
            %   mu, n       : CR3BP params (nd)
            %   vstar_km_s  : characteristic velocity to scale tolerance (km/s)
            % Outputs:
            %   ICs_family  : N×7 [x y z vx vy vz  T] corrected ICs and full period (nd)
            %   iters_vec   : N×1 Newton iterations used by the corrector

            IC         = IC(:).';
            tol_vx_kms = abs(eps_nd) * vstar_km_s;
            max_iter   = 20;
            tspan_evt  = [0, t_2];

            vy_guess   = IC(5);
            ICs_family = zeros(N, 7);
            iters_vec  = zeros(N, 1);

            for i = 1:N
                x0_try   = IC(1) + (i-1)*beta;
                X0_guess = [x0_try, 0, 0, 0, vy_guess, 0].';

                [Xcorr, T_half_nd, T_full_nd, iters] = ...
                    obj.perpTargeterVyOnly(X0_guess, mu, n, vstar_km_s, tspan_evt, tol_vx_kms, max_iter);

                ICs_family(i,:) = [Xcorr(:).', T_full_nd];
                iters_vec(i)    = iters;
                vy_guess        = Xcorr(5);   % warm-start next step
            end
        end



        % Continuation Method - XZ Perpendicular Orbits - Fixed X0
        function [ICs_Optimized, TestResults] = findPerpendicularXZPeriodicOrbits_FixedX (obj,startX0, dx, N, dir, mu, n, Tmax, tolODE, tolError, maxIter)
            % findPerpendicularXZPeriodicOrbits_FixedX Continuation method to
            % find a series of periodic orbits in the Circular Restricted
            % Three-Body Problem (CR3BP) by varying initial conditions.
            %
            % Usage:
            %   [ICs_Optimized, TestResults] =
            %   obj.findPerpendicularXZPeriodicOrbits_FixedX(knownX0, dx,
            %   N, dir, mu, n, Tmax, tolODE, tolError, maxIter)
            %
            % Inputs:
            %   startX0 - Known initial condition vector [xval, 0, zval, 0,
            %   vyval, 0] from which to start the continuation. dx -
            %   Increment to be applied to the x component of the initial
            %   conditions for each continuation step. N - Number of new
            %   orbits to find. dir - Direction to apply the continuation:
            %   'pos' for positive, 'neg' for negative, 'both' for both
            %   directions. mu - Gravitational parameter for the CR3BP. n -
            %   Mean motion (average angular velocity), typically 1 in
            %   normalized units. Tmax - Maximum time for orbit
            %   propagation. tolODE - Tolerance for the ODE solver.
            %   tolError - Tolerance for the error in the periodic orbit
            %   calculation. maxIter - Maximum number of iterations for the
            %   targeter.
            %
            % Outputs:
            %   ICs_Optimized - A matrix of optimized initial conditions
            %   for the found orbits. Each row represents an orbit,
            %   containing the state vector [x, y, z, vx, vy, vz] and the
            %   period of the orbit as the last column. TestResults - A
            %   table containing the test results for each orbit found,
            %   including the initial guess, optimized initial conditions,
            %   period, exit flag, and iteration count.
            %
            % The function employs a continuation method by perturbing the
            % x component of the initial conditions and attempting to find
            % new periodic orbits using a targeter. The direction and
            % magnitude of the perturbation can be specified. The
            % continuation process is repeated N times or until the
            % specified number of orbits is found. The function returns the
            % optimized initial conditions for each successful orbit, along
            % with a detailed report of the results in a table format.
            %
            % Example:
            %   mu = 0.01215058162343; % Earth-Moon CR3BP knownX0 = [0.8,
            %   0, 0, 0, 0.1, 0]; dx = 0.0001; N = 10; dir = 'both';
            %   [ICs_Optimized, TestResults] =
            %   obj.findPerpendicularXZPeriodicOrbits_FixedX(knownX0, dx,
            %   N, dir, mu, 1, 3, 1e-12, 1e-10, 100); This will attempt to
            %   find 10 new orbits in both positive and negative directions
            %   from the known initial condition, varying the x component
            %   by 0.0001 each step.


            % - - - - Sanity Checks on startX0 Provided - - - -
            % Ensure startX0 is a vector and has the correct number of
            % elements
            if ~isvector(startX0) || ~(length(startX0) == 6 || length(startX0) == 7)
                error('Initial guess must be a row or column vector with 6 or 7 elements. [x; y; z; vx; vy; vz; (optional)T]');
            end

            if isrow(startX0)           % If startX0 is a row vector
                startX0 = startX0';     % Convert it to a column vector
            end

            startX0 = startX0(1:6);      % Grab only 1st 6 elements we need

            % - - - - Determine Direction Multipliers - - - -
            % Direction multipliers to find new orbits, positive,
            % negative or both
            dirMultipliers = [];
            if strcmp(dir, 'pos')
                dirMultipliers = 1;
            elseif strcmp(dir, 'neg')
                dirMultipliers = -1;
            elseif strcmp(dir, 'both')
                dirMultipliers = [-1, 1];
            end

            % Check if number of directions being requested
            iterMultiplier = length(dirMultipliers);

            % - - - - Initialize Output Arrays - - - -
            % Initialize the arrays that store the results
            ICs_Optimized = [];
            TestResultsArray = [];

            % - - - - Run the Continuation Algorithm - - - -
            for j = 1:iterMultiplier                                    % For each direction
                knownX0 = startX0;                                      % We start at the same user-provided IC
                for i = 1:N                                             % For the number of orbits requested
                    guessX0 = knownX0;                                  % Known initial conditions becomes next guess
                    guessX0(1) = knownX0(1) + dirMultipliers(j) * dx;   % Add perturbation dx to x0
                    guessX0(5) = knownX0(5);                            % Keep vy0 the same as the knownX0

                    % Run the XZ perpendicular targeter for the current
                    % guess
                    [X0_optimized, Topt, exit_flag, iter_count] = obj.targeterPerpendicularXZ_FixedX(guessX0, mu, n, Tmax, tolODE, tolError, maxIter);

                    % Store the optimized initial conditions and period
                    ICs_Optimized(end+1,:) = [X0_optimized', Topt];         % Append to the matrix

                    % Append test results
                    TestResultsArray(end+1,:) = [guessX0(1), guessX0(5), X0_optimized', Topt, exit_flag, iter_count];

                    % Update knownX0 if successful for the next iteration
                    if exit_flag == 1
                        knownX0 = X0_optimized';
                    end
                end
            end

            % Prepare output test results table
            TestResults = array2table(TestResultsArray, ...
                'VariableNames', {'x0', 'vy0_guess', 'x_opt', 'y_opt', 'z_opt', 'vx_opt', 'vy_opt', 'vz_opt', 'Topt', 'exitFlag', 'iterCount'});

            % Display message regarding the number of orbits found
            numOrbitsFound = sum(TestResults.exitFlag == 1);
            fprintf('%d orbits were found with exit flag 1.\n', numOrbitsFound);
        end


        % Continuation Method - XZ Perpendicular Orbits - Fixed Z0
        function [ICs_Optimized, TestResults] = findPerpendicularXZPeriodicOrbits_FixedZ (obj,startX0, dz, N, dir, mu, n, Tmax, tolODE, tolError, maxIter)
            % findPerpendicularXZPeriodicOrbits_FixedZ Continuation method to
            % find a series of periodic orbits in the Circular Restricted
            % Three-Body Problem (CR3BP) by varying initial conditions.
            %
            % Usage:
            %   [ICs_Optimized, TestResults] =
            %   obj.findPerpendicularXZPeriodicOrbits_FixedZ(knownX0, dx,
            %   N, dir, mu, n, Tmax, tolODE, tolError, maxIter)
            %
            % Inputs:
            %   startX0 - Known initial condition vector [xval, 0, zval, 0,
            %   vyval, 0] from which to start the continuation. dz -
            %   Increment to be applied to the Z component of the initial
            %   conditions for each continuation step. N - Number of new
            %   orbits to find. dir - Direction to apply the continuation:
            %   'pos' for positive, 'neg' for negative, 'both' for both
            %   directions. mu - Gravitational parameter for the CR3BP. n -
            %   Mean motion (average angular velocity), typically 1 in
            %   normalized units. Tmax - Maximum time for orbit
            %   propagation. tolODE - Tolerance for the ODE solver.
            %   tolError - Tolerance for the error in the periodic orbit
            %   calculation. maxIter - Maximum number of iterations for the
            %   targeter.
            %
            % Outputs:
            %   ICs_Optimized - A matrix of optimized initial conditions
            %   for the found orbits. Each row represents an orbit,
            %   containing the state vector [x, y, z, vx, vy, vz] and the
            %   period of the orbit as the last column. TestResults - A
            %   table containing the test results for each orbit found,
            %   including the initial guess, optimized initial conditions,
            %   period, exit flag, and iteration count.
            %
            % The function employs a continuation method by perturbing the
            % x component of the initial conditions and attempting to find
            % new periodic orbits using a targeter. The direction and
            % magnitude of the perturbation can be specified. The
            % continuation process is repeated N times or until the
            % specified number of orbits is found. The function returns the
            % optimized initial conditions for each successful orbit, along
            % with a detailed report of the results in a table format.
            %
            % Example:
            %   mu = 0.01215058162343; % Earth-Moon CR3BP knownX0 = [0.8,
            %   0, 0, 0, 0.1, 0]; dx = 0.0001; N = 10; dir = 'both';
            %   [ICs_Optimized, TestResults] =
            %   obj.findPerpendicularXZPeriodicOrbits_FixedZ(knownX0, dx,
            %   N, dir, mu, 1, 3, 1e-12, 1e-10, 100); This will attempt to
            %   find 10 new orbits in both positive and negative directions
            %   from the known initial condition, varying the x component
            %   by 0.0001 each step.


            % - - - - Sanity Checks on startX0 Provided - - - -
            % Ensure startX0 is a vector and has the correct number of
            % elements
            if ~isvector(startX0) || ~(length(startX0) == 6 || length(startX0) == 7)
                error('Initial guess must be a row or column vector with 6 or 7 elements. [x; y; z; vx; vy; vz; (optional)T]');
            end

            if isrow(startX0)           % If startX0 is a row vector
                startX0 = startX0';     % Convert it to a column vector
            end

            startX0 = startX0(1:6);      % Grab only 1st 6 elements we need

            % - - - - Determine Direction Multipliers - - - -
            % Direction multipliers to find new orbits, positive,
            % negative or both
            dirMultipliers = [];
            if strcmp(dir, 'pos')
                dirMultipliers = 1;
            elseif strcmp(dir, 'neg')
                dirMultipliers = -1;
            elseif strcmp(dir, 'both')
                dirMultipliers = [-1, 1];
            end

            % Check if number of directions being requested
            iterMultiplier = length(dirMultipliers);

            % - - - - Initialize Output Arrays - - - -
            % Initialize the arrays that store the results
            ICs_Optimized = [];
            TestResultsArray = [];

            % - - - - Run the Continuation Algorithm - - - -
            for j = 1:iterMultiplier                                    % For each direction
                knownX0 = startX0;                                      % We start at the same user-provided IC
                for i = 1:N                                             % For the number of orbits requested
                    guessX0 = knownX0;                                  % Known initial conditions becomes next guess
                    guessX0(3) = knownX0(3) + dirMultipliers(j) * dz;   % Add perturbation dz to z0
                    guessX0(5) = knownX0(5);                            % Keep vy0 the same as the knownX0

                    % Run the XZ perpendicular targeter for the current
                    % guess
                    [X0_optimized, Topt, exit_flag, iter_count] = obj.targeterPerpendicularXZ_FixedZ(guessX0, mu, n, Tmax, tolODE, tolError, maxIter);

                    % Store the optimized initial conditions and period
                    ICs_Optimized(end+1,:) = [X0_optimized', Topt];     % Append to the matrix of optimized initial conditions

                    % Append test results
                    TestResultsArray(end+1,:) = [guessX0(3), guessX0(5), X0_optimized', Topt, exit_flag, iter_count];

                    % Update knownX0 if successful for the next iteration
                    if exit_flag == 1
                        knownX0 = X0_optimized';
                    end
                end
            end

            % Prepare output test results table
            TestResults = array2table(TestResultsArray, ...
                'VariableNames', {'z0', 'vy0_guess', 'x_opt', 'y_opt', 'z_opt', 'vx_opt', 'vy_opt', 'vz_opt', 'Topt', 'exitFlag', 'iterCount'});

            % Display message regarding the number of orbits found
            numOrbitsFound = sum(TestResults.exitFlag == 1);
            fprintf('%d orbits were found with exit flag 1.\n', numOrbitsFound);
        end


        

        % Continuation Method - XZ Perpendicular Orbits - Fixed VY0
        function [ICs_Optimized, TestResults] = findPerpendicularXZPeriodicOrbits_FixedVy (obj,startX0, dvy, N, dir, mu, n, Tmax, tolODE, tolError, maxIter)
            % findPerpendicularXZPeriodicOrbits_FixedVy Continuation method
            % to find a series of periodic orbits in the Circular Restricted
            % Three-Body Problem (CR3BP) by varying initial conditions.
            %
            % Usage:
            %   [ICs_Optimized, TestResults] =
            %   obj.findPerpendicularXZPeriodicOrbits_FixedVy(knownX0, dvy,
            %   N, dir, mu, n, Tmax, tolODE, tolError, maxIter)
            %
            % Inputs:
            %   startX0 - Known initial condition vector [xval, 0, zval, 0,
            %   vyval, 0] from which to start the continuation. dvy -
            %   Increment to be applied to the VY component of the initial
            %   conditions for each continuation step. N - Number of new
            %   orbits to find. dir - Direction to apply the continuation:
            %   'pos' for positive, 'neg' for negative, 'both' for both
            %   directions. mu - Gravitational parameter for the CR3BP. n -
            %   Mean motion (average angular velocity), typically 1 in
            %   normalized units. Tmax - Maximum time for orbit
            %   propagation. tolODE - Tolerance for the ODE solver.
            %   tolError - Tolerance for the error in the periodic orbit
            %   calculation. maxIter - Maximum number of iterations for the
            %   targeter.
            %
            % Outputs:
            %   ICs_Optimized - A matrix of optimized initial conditions
            %   for the found orbits. Each row represents an orbit,
            %   containing the state vector [x, y, z, vx, vy, vz] and the
            %   period of the orbit as the last column. TestResults - A
            %   table containing the test results for each orbit found,
            %   including the initial guess, optimized initial conditions,
            %   period, exit flag, and iteration count.
            %
            % The function employs a continuation method by perturbing the
            % x component of the initial conditions and attempting to find
            % new periodic orbits using a targeter. The direction and
            % magnitude of the perturbation can be specified. The
            % continuation process is repeated N times or until the
            % specified number of orbits is found. The function returns the
            % optimized initial conditions for each successful orbit, along
            % with a detailed report of the results in a table format.
            %
            % Example:
            %   mu = 0.01215058162343; % Earth-Moon CR3BP knownX0 = [0.8,
            %   0, 0, 0, 0.1, 0]; dx = 0.0001; N = 10; dir = 'both';
            %   [ICs_Optimized, TestResults] =
            %   obj.findPerpendicularXZPeriodicOrbits_FixedVy(knownX0, dx,
            %   N, dir, mu, 1, 3, 1e-12, 1e-10, 100); This will attempt to
            %   find 10 new orbits in both positive and negative directions
            %   from the known initial condition, varying the x component
            %   by 0.0001 each step.


            % - - - - Sanity Checks on startX0 Provided - - - -
            % Ensure startX0 is a vector and has the correct number of
            % elements
            if ~isvector(startX0) || ~(length(startX0) == 6 || length(startX0) == 7)
                error('Initial guess must be a row or column vector with 6 or 7 elements. [x; y; z; vx; vy; vz; (optional)T]');
            end

            if isrow(startX0)           % If startX0 is a row vector
                startX0 = startX0';     % Convert it to a column vector
            end

            startX0 = startX0(1:6);      % Grab only 1st 6 elements we need

            % - - - - Determine Direction Multipliers - - - -
            % Direction multipliers to find new orbits, positive,
            % negative or both
            dirMultipliers = [];
            if strcmp(dir, 'pos')
                dirMultipliers = 1;
            elseif strcmp(dir, 'neg')
                dirMultipliers = -1;
            elseif strcmp(dir, 'both')
                dirMultipliers = [-1, 1];
            end

            % Check if number of directions being requested
            iterMultiplier = length(dirMultipliers);

            % - - - - Initialize Output Arrays - - - -
            % Initialize the arrays that store the results
            ICs_Optimized = [];
            TestResultsArray = [];

            % - - - - Run the Continuation Algorithm - - - -
            for j = 1:iterMultiplier                                    % For each direction
                knownX0 = startX0;                                      % We start at the same user-provided IC
                for i = 1:N                                             % For the number of orbits requested
                    guessX0 = knownX0;                                  % Known initial conditions becomes next guess
                    guessX0(5) = knownX0(5) + dirMultipliers(j) * dvy;   % Add perturbation dvy to vy0

                    % Run the XZ perpendicular targeter for the current
                    % guess
                    [X0_optimized, Topt, exit_flag, iter_count] = obj.targeterPerpendicularXZ_FixedVy(guessX0, mu, n, Tmax, tolODE, tolError, maxIter);

                    % Store the optimized initial conditions and period
                    ICs_Optimized(end+1,:) = [X0_optimized', Topt];     % Append to the matrix of optimized initial conditions

                    % Append test results
                    TestResultsArray(end+1,:) = [guessX0(1), guessX0(3), guessX0(5), X0_optimized', Topt, exit_flag, iter_count];

                    % Update knownX0 if successful for the next iteration
                    if exit_flag == 1
                        knownX0 = X0_optimized';
                    end
                end
            end

            % Prepare output test results table
            TestResults = array2table(TestResultsArray, ...
                'VariableNames', {'x0', 'z0', 'vy0_guess', 'x_opt', 'y_opt', 'z_opt', 'vx_opt', 'vy_opt', 'vz_opt', 'Topt', 'exitFlag', 'iterCount'});

            % Display message regarding the number of orbits found
            numOrbitsFound = sum(TestResults.exitFlag == 1);
            fprintf('%d orbits were found with exit flag 1.\n', numOrbitsFound);
        end


        %% 3.6.1) Natural Parameter Continuation








        %% ===============================================================
        %% 3.7) State Extraction

        % State Finder At Tau
        function [states_at_taus, times_at_taus] = getStateAtTau(obj, T, X, taus, scale)
            % getStateAtTau Interpolates the states of a spacecraft at
            % given taus and returns the corresponding times.
            %
            % Usage:
            %   [states_at_taus, times_at_taus] = obj.getStateAtTau(T, X,
            %   taus, scale)
            %
            % Inputs:
            %   T      - Time vector from the ODE solver. X      - State
            %   vectors from the ODE solver. taus   - A single value or an
            %   array of desired fractional periods (can be greater than 1)
            %            representing positions along the orbit.
            %   scale  - Specifies whether taus is 'deg' for degrees or
            %   'per' for a percentage.
            %
            % Outputs:
            %   states_at_taus - An NX6 matrix of interpolated state
            %   vectors at times corresponding to taus. times_at_taus  - A
            %   vector of corresponding times at which the states_at_taus
            %   occur.

            if ~ismember(scale, {'deg', 'per'})
                error('Scale must be either "deg" for degrees or "per" for percentage.');
            end

            % Ensure taus is a column vector
            taus = taus(:);

            if strcmp(scale, 'deg')
                taus = taus / 360;  % Convert degrees to fraction of period
            end

            % Subtract integer number of revolutions for taus greater than
            % or equal to 1
            taus = mod(taus, 1.0);

            % Convert taus to corresponding times in T
            times_at_taus = taus * max(T); % Max T is the orbit period

            % Interpolate to find the states at the times corresponding to
            % taus
            states_at_taus = interp1(T, X, times_at_taus, 'spline');

            % Ensure output is column vector if a single tau was requested
            if isscalar(taus)
                states_at_taus = states_at_taus(:);
                times_at_taus = times_at_taus(:);
            end
        end


        %






        %% ===============================================================
        %% 3.8) Initial Conditions Guessers
        % Initial DeltaV Guess Based on Pseudopotential and JC
        function [dV_mag, dV_dir, dV_vec] = estimateDeltaV0_LPOtoMoon(obj,X_start, X_end, factor, mu, n)
            % This function estimates the deltaV required between points in
            % the CR3BP, including magnitude, direction, and the vector
            % itself.

            % #VALIDATE

            % - - - Sanity Checks for Inputs - - -
            if iscolumn(X_start)                                % If it's a column vector, turn it into a row vector
                X_start = X_start';
            end
            if iscolumn(X_end)                                  % If it's a column vector, turn it into a row vector
                X_end = X_end';
            end

            % If it's a matrix check for correct size
            if ~isvector(X_start) && (size(X_start, 2) < 6 || size(X_start, 2) > 7)
                error('Matrix X1 must have 6 or 7 columns.');
            end
            if (size(X_end, 2) < 6 || size(X_end, 2) > 7)
                error('Matrix X2 must have 6 or 7 columns.');
            end

            X_start = X_start(:, 1:6);      % Grab 1st 3 elements just in case it included propagation time or more things
            X_end = X_end(:, 1:6);

            % - - - Prepare X_end fix for size based on X_start - - -
            if isvector(X_end)
                X_end = repmat(X_end, size(X_start, 1), 1); % Repeat to match the number of rows in X1
            end

            % - - - Actual Calculation - - -
            dr_vec = X_end(:,1:3) - X_start(:,1:3); % Adjusted to use X2_repeated

            % Calculate direction and magnitude
            numPoints = size(X_start, 1);
            dV_dir = zeros(numPoints, 3);
            dV_mag = zeros(numPoints, 1);
            dV_vec = zeros(numPoints, 3);

            for i = 1:numPoints
                dV_dir(i,:) = dr_vec(i,:) / norm(dr_vec(i,:));   % Normalize each vector to get direction guess
                U1 = obj.pseudoPotentialCR3BP(X_start(i,:), mu, n);  % Pseudopotential at starting 1
                U2 = obj.pseudoPotentialCR3BP(X_end(i,:), mu, n);
                JC1 = obj.jacobiConstantCR3BP(X_start(i,:), mu, n);
                JC2 = obj.jacobiConstantCR3BP(X_end(i,:), mu, n);
                dV_mag(i) = factor * abs(sqrt(2*U2 - JC2) - sqrt(2*U2 - JC1));  % #CHECK
            end

            % Compute dV_vec as dV_dir scaled by dV_mag
            for i = 1:numPoints
                dV_vec(i,:) = dV_dir(i,:) * dV_mag(i);
            end

        end



        %% ===============================================================
        %% 3.9) CONVERTERS

        %% 3.9.1) Reference Frame Conversions

        % Sun-Centered Inertial Moon Vector to Synodic
        function [r_sm_B, r_sb_S, r_bm_S, r_sm_S] = sunToMoonSynodic(obj, t)
            % sunToMoonSynodic Calculates Sun-to-Moon position vectors in
            % both synodic and inertial (Sun-centered) frames of CR3BP.
            %
            % This function operates in non-dimensional units typical of
            % the CR3BP.
            %
            % Syntax:
            %   [r_sm_B, r_sb_S, r_bm_S] = sunToMoonSynodicWithInertial(t,
            %   r_sb, r_bm, omega_bary, omega_moon)
            %
            % Inputs:
            %   t           - Vector of time points (1 x N) in
            %   non-dimensional time units.
            %
            % Outputs:
            %   r_sm_B      - 3 x N matrix of position vectors in synodic
            %   frame.
            %                 Each column represents [X; Y; Z] at
            %                 corresponding time t.
            %   r_sb_S      - 3 x N matrix of Sun to barycenter vectors in
            %   inertial frame.
            %                 Each column represents [X; Y; Z] at
            %                 corresponding time t.
            %   r_bm_S      - 3 x N matrix of barycenter to Moon vectors in
            %   inertial frame.
            %                 Each column represents [X; Y; Z] at
            %                 corresponding time t.

            % Earth - Moon System Parameters in CR3BP
            mu = 0.012150585609624;
            l_star = 3.8475e5;

            r_se = 149e6 / l_star;      % [nd] Distance from Sun to Earth
            r_sb = r_se + mu;           % [nd] Distance from Sun to Earth-Moon Barycenter
            r_bm = 1 - mu;              % [nd] Distance from barycenter to the Moon
            omega_bary = 1.99e-7;       % [rad/s] Angular velocity of Earth around the Sun same as Barycenter's omega
            omega_moon = 2.6623e-6;             % [rad/s] Angular velocity of Moon around the Barycenter

            % Ensure t is a row vector
            t = t(:)';

            % Compute Angles
            theta = omega_bary .* t;      % Angle of barycenter around Sun
            alpha = omega_moon .* t;      % Angle of Moon around barycenter
            psi = theta + alpha;           % Total rotation angle

            % Position Vectors in Inertial (Sun-Centered) Frame Sun to
            % Barycenter
            r_sb_S = [r_sb .* cos(theta);
                r_sb .* sin(theta);
                zeros(1, length(t))];      % Z-component is zero

            % Barycenter to Moon
            r_bm_S = [r_bm .* cos(psi);
                r_bm .* sin(psi);
                zeros(1, length(t))];      % Z-component is zero

            % Sun to Moon in Inertial Frame
            r_sm_S = r_sb_S + r_bm_S;            % 3 x N

            % Define Rotation Matrices for Synodic Frame Initialize
            % rotation matrices (3 x 3 x N)
            N = length(t);
            R = zeros(3,3,N);

            % Compute cosine and sine of psi
            cos_psi = cos(psi);
            sin_psi = sin(psi);

            % Populate rotation matrices
            for i = 1:N
                R(:,:,i) = [cos_psi(i),  sin_psi(i), 0;
                    -sin_psi(i), cos_psi(i), 0;
                    0,           0,     1];
            end

            % Apply Rotation to Transform to Synodic Frame Initialize
            % synodic frame position matrix
            r_sm_B = zeros(3, N);

            % Apply rotation for each time step
            for i = 1:N
                r_sm_B(:,i) = R(:,:,i) * r_sm_S(:,i);
            end

        end

        % - - -  OLD VERSION - - - COMMENTED OUT DUE TO MISSING VALIDATION.
        % % BCI to Synodic
        % function states_syn_nd = bci_to_syn(obj, states_mtx_eci, theta, mu, celestial_body)
        %     % bci_to_syn - Transforms Cartesian state vectors from the Body Centered Frames frames to the synodic frame
        %     % based on the specified primary celestial body (Earth or Moon).
        %     %
        %     % Handles both position only and position + velocity inputs.
        %     % Can handle both dimensional or adimensional units as long as
        %     % assumptions are consistent throughout usage.
        %     %
        %     % Inputs:
        %     %   states_mtx_eci - Matrix of state vectors in the ECI frame [Nx3] or [Nx6] or [Nx7]
        %     %   theta - Array of rotation angles in radians corresponding to each state [Nx1]
        %     %   d1_or_mu - Adimensional distance from the barycenter to primary 1 or the mass ratio
        %     %   celestial_body - 'primary' for Earth or 'secondary' for Moon
        %     %
        %     % Outputs:
        %     %   states_syn - Matrix of state vectors in the synodic frame [Nx3] or [Nx6] or [Nx7]
        %
        %     % Initialize transformation vector for primary or secondary
        %     if strcmp(celestial_body, 'primary')
        %         trans_vec = [mu; 0; 0]; % Earth
        %
        %     elseif strcmp(celestial_body, 'secondary')
        %         trans_vec = - [1 - mu; 0; 0]; % Moon
        %
        %     else
        %         error('Unsupported celestial body specified. Use ''primary'' or ''secondary''.');
        %     end
        %
        %
        %     % Number of states and number of columns in the input matrix
        %     [num_states, num_cols] = size(states_mtx_eci);
        %
        %     % Check if the velocity is included
        %     has_velocity = num_cols >= 6;
        %
        %     % Check for the presence of a period as the 7th element
        %     has_period = num_cols == 7;
        %
        %     % Preallocate the output matrix
        %     states_syn_nd = zeros(num_states, num_cols);
        %
        %     % Loop over each state vector to transform
        %     for i = 1:num_states
        %
        %         % extract position and velocity vectors from state vector
        %         pos_bci_nd = states_mtx_eci(i, 1:3)';
        %         vel_bci_nd = states_mtx_eci(i, 4:6)';
        %
        %         % Define the DCM from BCI to Synodic, rotation around Z
        %         DCM = obj.adc.dcmFromSingleEulerAngle(3, theta(i), 'row');
        %
        %         % Define the inverse DCM for the current theta
        %         Rinv = [cos(theta(i)), sin(theta(i)), 0;
        %             -sin(theta(i)), cos(theta(i)), 0;
        %             0, 0, 1];
        %
        %         % % Only define the derivative of the inverse DCM matrix if velocity is present
        %         if has_velocity
        %             Rinv_dot = [-sin(theta(i)), cos(theta(i)), 0;
        %                 -cos(theta(i)), -sin(theta(i)), 0;
        %                 0, 0, 0];
        %         end
        %
        %         % Translate the BCI position vector back to the synodic origin
        %         pos_BCI_translated = pos_bci_nd - trans_vec;
        %
        %         % Apply the inverse DCM to transform from BCI to synodic frame
        %         pos_syn_nd = Rinv * pos_BCI_translated;
        %         states_syn_nd(i, 1:3) = pos_syn_nd';
        %
        %         % Store the transformed synodic position vector
        %         % pos_syn_nd = pos_BCI_nd' * DCM;
        %         % states_syn_nd(i, 1:3) = pos_syn_nd;
        %
        %
        %         % If velocity is provided, transform it % check this
        %         if has_velocity
        %             % % Apply inverse transport theorem to get the velocity in the synodic frame
        %             vel_syn_nd = Rinv * vel_bci_nd + Rinv_dot * pos_BCI_translated;
        %
        %             n = 1;
        %             % vel_syn_nd = vel_BCI_nd + [n * states_mtx_eci(i, 2); -n * (1 - mu + states_mtx_eci(i, 1)) ; 0];
        %
        %             % vel_syn_nd_bciFrame = vel_bci_nd' - cross([0 0 n], pos_bci_nd);
        %             % vel_syn_nd = vel_syn_nd_bciFrame * DCM;
        %             % vel_syn_nd = vel_syn_nd_bciFrame;
        %
        %
        %             states_syn_nd(i, 4:6) = vel_syn_nd;         % Store transformed velocity
        %         end
        %
        %
        %         % If a period is present, copy it directly without modification
        %         if has_period
        %             states_syn_nd(i, 7) = states_mtx_eci(i, 7);
        %         end
        %     end
        % end


        % Body Centered Inertial to Synodic
        function states_syn_nd = bci_to_syn(obj, states_mtx_eci, theta, mu, celestial_body)
            % bci_to_syn - Transforms BCI (body-centered-inertial) states into the synodic frame.
            %   states_mtx_eci : [N×3] or [N×6] or [N×7] array of BCI states (pos,v[,period]).
            %   theta          : [N×1] array of rotation angles (rad).
            %   mu             : mass‐ratio parameter (μ or 1−μ) already nondimensional.
            %   celestial_body : 'primary' (Earth) or 'secondary' (Moon).
            %
            % Output:
            %   states_syn_nd  : [N×3] or [N×6] or [N×7] array of synodic states in the same
            %                    format as the input (i.e. position only or position+velocity
            %                    or with extra 7th column preserved).

            % Frames definition
            % B: barycenter inertial frame
            % S: synodic rotating frame
            % E: body inertial frame
            % C: spacecraft (not used as a frame)

            %—Determine the "u" (distance of BCI origin from barycenter)—
            switch lower(celestial_body)
                case 'primary'
                    u = mu;
                case 'secondary'
                    % the BCI origin is at the Moon; its nondim distance from barycenter is (1−μ),
                    % but we want to subtract (−(1−μ) cosθ, −(1−μ) sinθ, 0) below.
                    u = 1 - mu;
                otherwise
                    error("bci_to_syn: unknown 'celestial_body'. Use 'primary' or 'secondary'.")
            end

            [num_states, num_cols] = size(states_mtx_eci);
            has_velocity = (num_cols >= 6);
            has_period   = (num_cols == 7);

            % Preallocate output
            states_syn_nd = zeros(num_states, num_cols);

            for i = 1:num_states

                %--- 1) Extract BCI position (x,y,z) and velocity (vx,vy,vz) ---
                % Position vector from origin of BCI to spacecraft
                pos_ec_bci = states_mtx_eci(i,1:3).';   % [x; y; z]

                if has_velocity
                    vel_bci = states_mtx_eci(i,4:6).';  % [vx; vy; vz]
                else
                    vel_bci = [0;0;0];
                end

                theta_i = theta(i);  % current rotation angle

                %--- 2) Vector from BCI origin to synodic/barycentric frame origin in BCI coordinates ---
                if strcmpi(celestial_body, 'primary')
                    r_eb_bci = [u * cos(theta_i);  u * sin(theta_i );  0];
                else
                    r_eb_bci = [ -u*cos(theta_i );  -u*sin(theta_i );  0]; % #check
                end

                %--- 3) Compute vector from barycenter to spacecraft r_{c/b} ---
                r_bc_bci = pos_ec_bci - r_eb_bci;   % 3×1

                %--- 4) Compute the DCM from synodic‐frame to barycentric/synodic inertial ---
                R = [ cos(theta_i ),  -sin(theta_i ), 0;
                    sin(theta_i ),  cos(theta_i ), 0;
                    0,       0,   1 ];

                % Compute spacecraft position wrt barycenter in synodic coordinates
                pos_bc_syn = r_bc_bci' * R ;
                states_syn_nd(i, 1:3) = pos_bc_syn;

                %--- 5) If velocity is provided, apply transport theorem:
                %      v_c/B  in bci→syn velocity is
                %        [ ẋ + n ( y − u sinθ );
                %          ẏ − n ( x − u cosθ );
                %          ż ]
                if has_velocity
                    x_dot_bci = vel_bci(1);
                    y_dot_bci = vel_bci(2);
                    z_dot_bci = vel_bci(3);

                    % n = 1 in normalized CR3BP
                    n = 1;

                    % note r_bc_bci = [ x−u cosθ;  y−u sinθ;  z ]
                    x_bc_rel = r_bc_bci(1);   % = x − u cosθ
                    y_bc_rel = r_bc_bci(2);   % = y − u sinθ

                    % note vel wrt barycenter depends on where we place the
                    % velocity vector
                    v_BC_bci = [ x_dot_bci +  n * y_bc_rel;
                        y_dot_bci -  n * x_bc_rel;
                        z_dot_bci ];

                    % now rotate that relative‐velocity vector into the synodic frame:
                    vel_syn_nd = v_BC_bci' * R;
                    states_syn_nd(i, 4:6) = vel_syn_nd;
                end

                %--- 6) If there is a 7th column (period), just copy it over unmodified ---
                if has_period
                    states_syn_nd(i,7) = states_mtx_eci(i,7);
                end
            end
        end


        % Synodic → Body-Centered Inertial
        function states_bci_nd = syn_to_bci(obj, states_syn_nd, theta, mu, celestial_body)
            % syn_to_bci - Transforms synodic-frame states back into the
            %               Body-Centered Inertial (BCI) frame.
            %
            %   states_syn_nd  : [N×3] or [N×6] or [N×7] array of synodic states
            %                    (position[,velocity][,period]).
            %   theta          : [N×1] vector of rotation angles (rad).
            %   mu             : nondimensional mass-ratio parameter.
            %   celestial_body : 'primary'  (Earth-centered BCI) or
            %                    'secondary' (Moon-centered BCI).
            %
            % Output:
            %   states_bci_nd  : [N×3] or [N×6] or [N×7] array of BCI states in
            %                    the same format as the input.

            %—determine u = distance of BCI origin from barycenter—%
            switch lower(celestial_body)
                case 'primary'
                    u = mu;
                case 'secondary'
                    u = 1 - mu;
                otherwise
                    error("syn_to_bci: unknown 'celestial_body'. Use 'primary' or 'secondary'.")
            end

            [num_states, num_cols] = size(states_syn_nd);
            has_velocity = (num_cols >= 6);
            has_period   = (num_cols == 7);

            %—preallocate output—%
            states_bci_nd = zeros(num_states, num_cols);

            % mean motion in nondimensional CR3BP
            n = 1;

            for i = 1:num_states
                % current angle
                theta_i = theta(i);

                % DCM from synodic → BCI (row-vector use later)
                R   = [ cos(theta_i), -sin(theta_i), 0;
                    sin(theta_i),  cos(theta_i), 0;
                    0,        0,    1 ];

                Rt  = R.';  % BCI → synodic for row-vectors

                %—1) position: r_c/B in BCI coordinates—%
                pos_syn = states_syn_nd(i,1:3);       % [x_s, y_s, z_s]
                r_bc_bci = pos_syn * Rt;              % row → [x−u cosθ, y−u sinθ, z]

                %—2) add back the BCI-origin offset r_E/B—%
                if strcmpi(celestial_body,'primary')
                    r_eb_bci = [ u*cos(theta_i),  u*sin(theta_i), 0 ];
                else
                    r_eb_bci = [-u*cos(theta_i), -u*sin(theta_i), 0 ];
                end
                states_bci_nd(i,1:3) = r_bc_bci + r_eb_bci;

                %—3) velocity: invert transport theorem if present—%
                if has_velocity
                    vel_syn = states_syn_nd(i,4:6);    % [vx_s, vy_s, vz_s]
                    v_rel_bci = vel_syn * Rt;          % [ẋ + n y_rel, ẏ − n x_rel, ż] pre-corr

                    x_rel = r_bc_bci(1);
                    y_rel = r_bc_bci(2);

                    v_bci_row = [ v_rel_bci(1) - n*y_rel, ...
                        v_rel_bci(2) + n*x_rel, ...
                        v_rel_bci(3) ];
                    states_bci_nd(i,4:6) = v_bci_row;
                end

                %—4) copy period if exists—%
                if has_period
                    states_bci_nd(i,7) = states_syn_nd(i,7);
                end
            end
        end

        % % Synodic to ECI
        % function states_bci = syn_to_bci(obj,states_mtx_syn, theta, d1_or_mu)
        %   % synodicToECI - Transforms Cartesian state vectors from the
        %   % synodic frame to the ECI frame
        %     % > > It can handle both dimensional or adimensional units
        %     % Assumes:
        %     %           *** States are in cartesian coordinates
        %     %           *** Units are adimensional
        %     % Inputs:
        %     %   states_mtx_syn - Matrix of state vectors in the synodic
        %     %   frame [Nx3], [Nx6], or [Nx7] theta - The rotation angle in
        %     %   radians d1_or_mu - Adimensional distance from primary 1 to
        %     %   barycenter or the mass ratio (normalized distance unit)
        %     %
        %     % Outputs:
        %     %   states_eci - Matrix of state vectors in the ECI frame
        %     %   [Nx3], [Nx6], or [Nx7]
        %
        %     % Number of states and number of columns in the input matrix
        %     [num_states, num_cols] = size(states_mtx_syn);
        %
        %     % Check if the velocity and time are included
        %     has_velocity = num_cols >= 6;
        %     has_time = num_cols == 7;
        %
        %     % Preallocate the output matrix
        %     states_bci = zeros(num_states, num_cols);
        %
        %     % Define the DCM matrix based on the provided angle theta
        %     R = [cos(theta), -sin(theta), 0;
        %          sin(theta), cos(theta), 0;
        %          0, 0, 1];
        %
        %     % Only define the derivative of the DCM matrix if velocity is
        %     % present
        %     if has_velocity
        %         R_dot = [-sin(theta), -cos(theta), 0;
        %                   cos(theta), -sin(theta), 0;
        %                   0, 0, 0];
        %     end
        %
        %     % Transform each state vector
        %     for i = 1:num_states
        %         % Translate the synodic frame origin in the x direction by
        %         % d1_or_mu
        %         pos_syn = states_mtx_syn(i, 1:3)'; % Extract the position vector
        %         pos_syn(1) = pos_syn(1) + d1_or_mu; % Translate the x component
        %
        %         % Position vector transformation from translated synodic to
        %         % ECI frame
        %         pos_eci = R * pos_syn;
        %         states_bci(i, 1:3) = pos_eci'; % Store transformed position
        %
        %         % If velocity is provided, transform it
        %         if has_velocity
        %             vel_syn = states_mtx_syn(i, 4:6)'; % Extract the velocity vector
        %             vel_eci = R * vel_syn + R_dot * pos_syn; % Apply transport theorem
        %             states_bci(i, 4:6) = vel_eci'; % Store transformed velocity
        %         end
        %
        %         % If time span is included, copy it to the output
        %         if has_time
        %             states_bci(i, 7) = states_mtx_syn(i, 7);
        %         end
        %     end
        % end



        % Selenographic to Synodic Cartesian
        function r_synodic = selenographicToBarycentricCartesian(obj,lat, long, mu)
            % selenographicToBarycentricCartesian converts selenographic
            % coordinates
            % (latitude and longitude on the Moon's surface) to
            % Cartesian coordinates in the barycentric reference frame
            % for the Earth-Moon system.
            %
            % Inputs: - phi: Latitude in degrees. - theta: Longitude in
            % degrees. - u: Mass parameter (mass of the secondary
            % divided by the total mass).
            %
            % Output: - r_barycentric: 3-element vector of Cartesian
            % coordinates [x, y, z]
            %   in kilometers, in the barycentric reference frame.

            % Mean radius of the Moon in nondimensional
            RM = 1737.1/384750;

            % Convert angles from degrees to radians
            phi_rad = deg2rad(lat);
            theta_rad = deg2rad(long);

            % Spherical to Cartesian conversion for a point on the
            % Moon's surface
            x_moon = RM * cos(phi_rad) * cos(theta_rad);
            y_moon = RM * cos(phi_rad) * sin(theta_rad);
            z_moon = RM * sin(phi_rad);

            % Shift the x-coordinate by (1 - u) to convert to the
            % barycentric frame
            x_barycentric = x_moon + (1 - mu);
            y_barycentric = y_moon;
            z_barycentric = z_moon;

            % Combine into a vector
            r_synodic = [x_barycentric; y_barycentric; z_barycentric];
        end





        % ================== coe to cartesian (body centered inertial) =========================
        function X_cart = coe_to_cart(obj, X_coe, mu)
            %COE_TO_CARTESIAN Convert Keplerian elements to Cartesian state vectors.
            %
            %   X_cart = coe_to_cartesian(obj, X_coe)
            %   X_cart = coe_to_cartesian(obj, X_coe, mu)
            %
            % INPUTS
            %   X_coe : 1×6 or N×6 matrix of classical orbital elements:
            %           [a, e, i, RAAN, omega, M]
            %             • a     — semi-major axis (distance units)
            %             • e     — eccentricity (unitless)
            %             • i     — inclination (rad)
            %             • RAAN  — right ascension of ascending node (rad)
            %             • omega — argument of periapsis (rad)
            %             • M     — mean anomaly (rad)
            %
            %   mu    : (optional) gravitational parameter of the central body
            %           in the same distance^3/time^2 units as a.
            %           If omitted, the method will attempt to use obj.mu or obj.muAU.
            %
            % OUTPUT
            %   X_cart : N×6 matrix of Cartesian states
            %            [x, y, z, vx, vy, vz]
            %            in the same distance and time units as a and mu.
            %
            %   • If you pass a single 1×6 row in, you get a single 1×6 row out.
            %   • All angles must be in radians before calling this function.

            % —— Handle optional mu argument ——
            if nargin < 3 || isempty(mu)
                error('coe_to_cartesian:mu','Please provide mu');
            end

            % —— Ensure row form for single-vector input ——
            if isvector(X_coe)
                X_coe = reshape(X_coe, 1, []);
            end

            [nRows, nCols] = size(X_coe);
            assert(nCols == 6, 'Input must be 1×6 or N×6.');

            % —— Preallocate output: N rows, 6 columns ——
            X_cart = zeros(nRows, 6);  % columns: [x y z vx vy vz]

            % —— Loop over each set of elements ——
            for k = 1:nRows
                % Unpack the keplerian elements
                a     = X_coe(k,1);    % semi-major axis
                e     = X_coe(k,2);    % eccentricity
                i   = X_coe(k,3);      % inclination
                RAAN  = X_coe(k,4);    % RAAN
                omega = X_coe(k,5);    % argument of periapsis
                M     = X_coe(k,6);    % mean anomaly

                % ---- Compute anomalies and radius ----
                E   = obj.orb.E_Me(M, e);        % eccentric anomaly
                nu  = obj.orb.ta_eE(e, E);       % true anomaly
                r   = obj.orb.r_aeta(a, e, nu);  % orbital radius

                % ---- Build DCM from orbital plane to inertial frame ----
                DCM = obj.orb.dcmFromEulerAngleSeq([3 1 3],[RAAN, i, omega + nu],'row');

                % ---- Rotate position into Cartesian coords ----
                r_orbital = [r 0 0];
                r_cart    = r_orbital * DCM;

                % ---- Compute velocity in the orbital plane ----
                v = obj.orb.vel_ura(mu, r, a, 'E');              % velocity magnitude
                fpa = obj.orb.fpa_eta(e,nu);                     % flight path angle at epoch
                v_rot = obj.orb.v_vec_rot_frame_fpav(fpa,v);     % vel vector in RTN frame at epoch


                % ---- Rotate velocity into Cartesian coords ----
                v_cart = v_rot * DCM;

                % ---- Store the full state vector ----
                X_cart(k, :) = [r_cart, v_cart];
            end
        end

        %% 3.9.2) Dimensionality Conversions

        % ===============================================================
        %  State Non Dimensional to Dimensional
        % ===============================================================
        function converted_values = nondim_to_dim(obj, values, l_star, t_star, conversionType)
            % nondim_to_dim - Converts non-dimensional values or vectors to dimensional units
            %
            % Inputs:
            %   values - Non-dimensional value(s) to be converted (can be a single value or a vector)
            %   l_star - Characteristic length (distance between primaries) [km]
            %   t_star - Characteristic time [s]
            %   conversionType - Type of conversion ('pos', 'vel', 'accel', 'time')
            %
            % Output:
            %   converted_values - Values converted to physical units based on conversionType

            % Validate the conversion type
            validTypes = {'pos', 'vel', 'accel', 'time'};
            if ~ismember(conversionType, validTypes)
                error('Invalid conversion type. Must be ''pos'', ''vel'', ''accel'', or ''time''.');
            end

            % Define characteristic velocity and acceleration
            v_star = l_star / t_star;           % Characteristic velocity [km/s]
            a_star = l_star / (t_star^2);       % Characteristic acceleration [km/s^2]

            % Convert the values based on the type
            switch conversionType
                case 'pos'
                    % Position conversion from non-dimensional to km
                    converted_values = values * l_star;
                case 'vel'
                    % Velocity conversion from non-dimensional to km/s
                    converted_values = values * v_star;
                case 'accel'
                    % Acceleration conversion from non-dimensional to km/s^2
                    converted_values = values * a_star;
                case 'time'
                    % Time conversion from non-dimensional to s
                    converted_values = values * t_star;
                otherwise
                    error('Unexpected conversion type.');
            end
        end


        % ===============================================================
        %  State Dimensional to Non Dimensional
        % ===============================================================
        function X_values_nd = dim_to_nondim(obj, X_values, l_star, t_star)
            % dim_to_nondim - Converts dimensional state vectors to adimensional units
            %
            % Inputs:
            %   X_values - Dimensional state vector(s) [x y z vx vy vz] to be converted
            %              Can be a single vector or a matrix of vectors where each row is a state.
            %              If a 7th column (period) is included, it will also be converted.
            %   l_star - Characteristic length (distance between primaries) [km]
            %   t_star - Characteristic time [s]
            %
            % Output:
            %   X_values_nd - State vector(s) converted to adimensional units

            % Ensure input is in row vector form if it's a single vector
            if isvector(X_values) && size(X_values, 1) > 1
                X_values = X_values'; % Transpose to row vector
            end

            % Define characteristic velocity and time
            v_star = l_star / t_star; % Characteristic velocity [km/s]

            % Determine the number of state variables per row
            numVariables = size(X_values, 2);

            % Initialize adimensional values matrix
            X_values_nd = zeros(size(X_values));

            % Convert position and velocity components
            if numVariables >= 6
                X_values_nd(:, 1:3) = X_values(:, 1:3) / l_star;  % Convert positions [x y z]
                X_values_nd(:, 4:6) = X_values(:, 4:6) / v_star;  % Convert velocities [vx vy vz]
                if numVariables == 7
                    X_values_nd(:, 7) = X_values(:, 7) / t_star;  % Convert period if present
                end
            else
                error('Input must have at least 6 columns corresponding to [x y z vx vy vz].');
            end

            % Ensure output maintains the original format (row if single vector)
            if isrow(X_values_nd)
                X_values_nd = X_values_nd';  % Return it as a column vector if the original input was a column vector
            end
        end


        % ===============================================================
        %  State History to Non Dimensional to Dimensional
        % ===============================================================

        function states_mtx_dim = nonDimStatesToDim(obj,states_mtx, l_star, t_star)
            % convertICsToRealUnits - Converts non-dimensional ICs to real
            % units
            %
            % Inputs:
            %   states_eci - N X 7 Array of non-dimensional state values
            %   [x, y, z, vx, vy, vz, Tmax] l_star - Characteristic length
            %   (distance between primaries) [km] t_star - Characteristic
            %   time [s]
            %
            % Outputs:
            %   states_mtx_dim - State matrix converted to physical units
            %   [km, km/s, s]
            %  [x ; y ; z ; vx ; vy ; vz ; t]

            % Define characteristic velocity
            v_star = l_star / t_star;  % Characteristic velocity [m/s]

            % Initialize output array
            states_mtx_dim = zeros(size(states_mtx));

            % Convert each set of states to dimensional
            for i = 1:size(states_mtx, 1)
                % Position conversion from nondimensional to km
                states_mtx_dim(i, 1:3) = states_mtx(i, 1:3) * l_star;  % Convert position to km

                % Velocity conversion from nondimensional to km/s
                states_mtx_dim(i, 4:6) = states_mtx(i, 4:6) * v_star;  % Convert velocity to km/s

                % Time
                states_mtx_dim(i, 7) = states_mtx(i, 7) * t_star;
            end
        end




        %% ===============================================================
        %% 3.10) Essential CR3BP Values


        % Jacobians [ A ] Matrix : Jacobian A = df/dx
        function [A] = dynamicsJacobianA(obj, x, y, z, mu)
            % A_dynamicsJacobian calculates the Jacobian matrix of the CR3BP
            % dynamics with respect to it state vector This function is
            % very similar to the pseudopotential Hessian one, but includes
            % the missing components to form the A matrix
            %
            % Inputs:
            %   x, y, z - Position coordinates in the normalized rotating
            %   frame mu - parameter of the CR3BP system
            %
            % Output:
            %   U_Hessian - 3x3 Hessian matrix of second-order partial
            %   derivatives of
            %               the pseudopotential function U with respect to
            %               x, y, and z

            O3 = zeros(3,3);
            I3 = eye(3,3);
            Omega = [0 2 0;
                -2 0 0;
                0 0 0];

            r1 = sqrt((x + mu)^2 + y^2 + z^2);
            r2 = sqrt((x - 1 + mu)^2 + y^2 + z^2);

            % Second-order partial derivatives of the pseudopotential
            % function
            Uxx = 1 - (1 - mu) / r1^3 - mu / r2^3 + 3 * (1 - mu) * (x + mu)^2 / r1^5 + 3 * mu * (x - 1 + mu)^2 / r2^5;
            Uyy = 1 - (1 - mu) / r1^3 - mu / r2^3 + 3 * (1 - mu) * y^2 / r1^5 + 3 * mu * y^2 / r2^5;
            Uzz =  - (1 - mu) / r1^3 - mu / r2^3 + 3 * (1 - mu) * z^2 / r1^5 + 3 * mu * z^2 / r2^5;
            Uxy = 3 * (1 - mu) * (x + mu) * y / r1^5 + 3 * mu * (x - 1 + mu) * y / r2^5;
            Uxz = 3 * (1 - mu) * (x + mu) * z / r1^5 + 3 * mu * (x - 1 + mu) * z / r2^5;
            Uyz = 3 * (1 - mu) * y * z / r1^5 + 3 * mu * y * z / r2^5;

            % Assemble the Hessian matrix (second-order gradient matrix)
            U_Hessian = [Uxx, Uxy, Uxz;
                Uxy, Uyy, Uyz;
                Uxz, Uyz, Uzz];

            % Assemble the A matrix
            A = [O3 I3;
                U_Hessian Omega];

        end

        % Hessians Pseudopotential Hessian
        function [U_Hessian] = pseudoPotentialHessian(obj,x,y,z,mu)
            % PseudoPotentialHessian calculates the Hessian matrix of the
            % pseudopotential function for the Circular Restricted Three
            % Body Problem (CR3BP)
            %
            % Assumes:
            %           *** No perturbations in the system dynamics
            %           *** Nondimensional units are being used
            % Inputs:
            %   x, y, z - Position coordinates in the normalized rotating
            %   frame mu - parameter of the CR3BP system
            %
            % Output:
            %   U_Hessian - 3x3 Hessian matrix of second-order partial
            %   derivatives of
            %               the pseudopotential function U with respect to
            %               x, y, and z

            % Calculate distances r1 and r2 to the primaries
            r1 = sqrt((x + mu)^2 + y^2 + z^2);                      % Also known as d
            r2 = sqrt((x - 1 + mu)^2 + y^2 + z^2);                  % Also known as r

            % Second-order partial derivatives of the pseudopotential
            % function
            Uxx = 1 - (1 - mu) / r1^3 - mu / r2^3 + 3 * (1 - mu) * (x + mu)^2 / r1^5 + 3 * mu * (x - 1 + mu)^2 / r2^5;
            Uyy = 1 - (1 - mu) / r1^3 - mu / r2^3 + 3 * (1 - mu) * y^2 / r1^5 + 3 * mu * y^2 / r2^5;
            Uzz =  - (1 - mu) / r1^3 - mu / r2^3 + 3 * (1 - mu) * z^2 / r1^5 + 3 * mu * z^2 / r2^5;
            Uxy = 3 * (1 - mu) * (x + mu) * y / r1^5 + 3 * mu * (x - 1 + mu) * y / r2^5;
            Uxz = 3 * (1 - mu) * (x + mu) * z / r1^5 + 3 * mu * (x - 1 + mu) * z / r2^5;
            Uyz = 3 * (1 - mu) * y * z / r1^5 + 3 * mu * y * z / r2^5;

            % Assemble the Hessian matrix (second-order gradient matrix)
            U_Hessian = [Uxx, Uxy, Uxz;
                Uxy, Uyy, Uyz;
                Uxz, Uyz, Uzz];
        end

        %  Pseudopotential Function
        function Ustar = pseudoPotentialCR3BP(obj,X, mu, n)
            % pseudoPotentialCR3BP calculates the pseudopotential in the
            % Circular Restricted
            % Three-Body Problem (CR3BP) for given state vectors. It
            % evaluates the combined gravitational potential of the two
            % primaries and the centrifugal potential experienced by a
            % spacecraft.
            %
            % Inputs:
            %   X  - State vectors as a matrix where each row is [x, y,
            %   z, vx, vy, vz], or a single
            %        state vector for a specific point in time.
            %   mu - Gravitational parameter, ratio of the secondary's
            %   mass to the total system mass. n  - Mean motion of the
            %   system, typically set to 1 in normalized CR3BP units.
            %
            % Outputs:
            %   Ustar - Vector of pseudopotential values at each
            %   provided state.
            %
            % The function ensures X has the correct form and contains
            % at least the position components. It focuses on the first
            % 6 elements of X, processing either a single vector or
            % multiple vectors.


            % - - - - Sanity Check on Vector or Matrix X - - - -
            if isvector(X) && length(X) < 3             % Check if X is a vector with correct dimensions
                error('Incomplete state vector provided. State vector must have at least 3 position elements.');
            end

            if iscolumn(X)                              % If X is column vector
                X = X';                                 % Transpose to row vector
            end

            if ~isvector(X) && size(X, 2) < 3           % If X is a matrix with columns less than 6
                error('Incomplete state matrix provided. State vector must have at least 3 position elements.');
            end

            X = X(:,1:3);                              % We are interested in the 1st 6 cols of X only

            % - - - - Initialize Outputs - - - -
            rows = size(X, 1);           % Number of timesteps in the input matrix
            Ustar = zeros(rows, 1);

            % Compute the pseudopotential for each state vector
            for i = 1:rows
                % Extract the position values x, y, z from the current row
                x = X(i, 1);
                y = X(i, 2);
                z = X(i, 3);

                % Calculate distances to the primary (r1) and secondary
                % (r2) bodies
                r1 = sqrt((x + mu)^2 + y^2 + z^2);              % Distance to the primary body a.k.a as d
                r2 = sqrt((x - 1 + mu)^2 + y^2 + z^2);          % Distance to the secondary body a.k.a as r

                Ustar(i) = (1 - mu) / r1 + mu / r2 + 0.5 * n^2 * (x^2 + y^2);
            end
        end

        % Jacobi Constant, Mean and Standard Deviation
        function [JC, JC_mean, JC_std] = jacobiConstantCR3BP(obj,X, mu, n)
            % jacobiConstantCR3BP Calculates the Jacobi Constant for a set
            % of state vectors in the Circular Restricted Three-Body
            % Problem (CR3BP).
            %
            % Usage:
            %   [JC, JC_mean, JC_std] = obj.jacobiConstantCR3BP(X, mu,
            %   n)
            %
            % Inputs:
            %   X - A row or col vector, or a matrix of state vectors
            %   where each row is a state vector [x, y, z, vx, vy, vz].
            %       X can represent multiple initial conditions from
            %       different orbits or a propagation of a single orbit
            %       over time.
            %   mu - Gravitational parameter of the CR3BP, representing
            %   the mass ratio of the two primary bodies. n - Mean
            %   motion (average angular velocity), typically set to 1
            %   in normalized units.
            %
            % Outputs:
            %   JC - Vector of Jacobi Constant values calculated for
            %   each input state vector. JC_mean - Mean value of the
            %   Jacobi Constant across all input state vectors. JC_std
            %   - Standard deviation of the Jacobi Constant across all
            %   input state vectors.
            %
            % The function computes the Jacobi Constant, a scalar
            % quantity that is conserved along any trajectory in the
            % CR3BP, for each provided state vector. It can be used to
            % analyze the stability and feasibility of orbits by
            % examining how the Jacobi Constant varies over time for a
            % single orbit or across different initial conditions for
            % multiple orbits. The mean and standard deviation of the
            % calculated Jacobi Constants provide insights into the
            % overall dynamics and variability of the system or orbit
            % being studied.
            %
            % Example:
            %   mu = 0.01215058162343; % Earth-Moon CR3BP n = 1; % By
            %   definition in CR3BP normalized units X = [0.8, 0, 0, 0,
            %   0.1, 0; 0.81, 0, 0, 0, 0.105, 0]; % Two different
            %   initial conditions [JC, JC_mean, JC_std] =
            %   obj.jacobiConstantCR3BP(X, mu, n); This calculates the
            %   Jacobi Constant for two different initial conditions
            %   and returns the mean and standard deviation of both
            %   JCs.


            % - - - - Sanity Check on Vector or Matrix X - - - -
            if isvector(X) && length(X) < 6             % Check if X is a vector with correct dimensions
                error('Incomplete state vector provided. Each state vector must have 6 or 7 elements.');
            end

            if iscolumn(X)                              % If X is column vector
                X = X';                                 % Transpose to row vector
            end

            if  ismatrix(X) && (size(X, 2) < 6 || size(X, 2) > 7)            % If X is a matrix with columns less than 6
                error('Incomplete state vectors provided. Each state vector in the matrix must have 6 or 7 elements (cols).');
            end

            X = X(:,1:6);                              % We are interested in the 1st 6 cols of X only

            % - - - -  Initialization of outputs - - - -
            rows = size(X, 1);                          % Number of timesteps in the input matrix
            JC = zeros(rows, 1);

            % - - - - Compute JC for each iteration - - - -
            for i = 1:rows
                % Extract the state values from the current row
                x = X(i, 1);  y = X(i, 2);   z = X(i, 3);
                vx = X(i, 4); vy = X(i, 5); vz = X(i, 6);
                v = norm([vx, vy, vz], 2);               % Magnitude of the velocity

                % Calculate distances to the primary (r1) and secondary
                % (r2) bodies
                r1 = sqrt((x + mu)^2 + y^2 + z^2);      % Distance to the primary body a.k.a as d
                r2 = sqrt((x - 1 + mu)^2 + y^2 + z^2);  % Distance to the secondary body a.k.a as r

                Ustar = (1 - mu) / r1 + mu / r2 + 0.5 * n^2 * (x^2 + y^2);

                JC(i) = 2 * Ustar - v^2;            % Calculate the JC value
            end

            % Calculate mean and standard deviation of the Jacobi constants
            % as ref
            JC_mean = mean(JC);
            JC_std = std(JC);
        end



        function [mu, Mtot, Tstar_s, P_s, Vstar_km_s] = cr3bp_characteristics(obj, m1, m2, a_km)
            % cr3bp_characteristics  Characteristic quantities for a CR3BP system.
            %
            % Units:
            %   - m1, m2 in kilograms [kg]
            %   - a_km  in kilometers [km]
            %
            % Computes:
            %   mu          = m2/(m1+m2)                          (mass ratio, nondimensional)
            %   Mtot        = (m1+m2)                             (total mass, kg)
            %   Tstar_s     = sqrt(a^3/(G*Mtot))                  (characteristic time, s)
            %   P_s         = 2*pi*Tstar_s                        (characteristic period, s)
            %   Vstar_km_s  = sqrt(G*Mtot/a) = a/Tstar_s          (characteristic velocity, km/s)
            %
            % Usage:
            %   [mu, Mtot, Tstar_s, P_s, Vstar_km_s] = cr3bp_characteristics(m1, m2, a_km)
            %
            % Notes:
            %   - All inputs can be scalars or equally sized arrays (element-wise ops).
            %   - Errors if masses are negative, m1+m2==0, or a_km<=0.

            % Gravitational constant in km^3/(kg·s^2)
            G_km = 6.67430e-20;

            % Basic validation
            if any(m1(:) < 0) || any(m2(:) < 0)
                error('cr3bp_characteristics:NegativeMass','m1 and m2 must be nonnegative.');
            end
            if any(a_km(:) <= 0)
                error('cr3bp_characteristics:NonpositiveSeparation','a_km must be > 0.');
            end

            Mtot = m1 + m2;
            if any(Mtot(:) == 0)
                error('cr3bp_characteristics:ZeroTotalMass','m1 + m2 must be > 0.');
            end

            mu         = m2 ./ Mtot;
            Tstar_s    = sqrt( (a_km.^3) ./ (G_km .* Mtot) );
            P_s        = 2*pi*Tstar_s;

            % Characteristic velocity: v* = a / T* = sqrt(G*M / a)
            Vstar_km_s = sqrt( (G_km .* Mtot) ./ a_km );
        end



        %% ===============================================================
        %% 3.11) Lagrange Equilibrium Points L1, L2, L3


        function out = getL1L2L3(obj, name, m1, m2, a_km, tol, maxit, verbose)
            % getL1L2L3  Solve for L1, L2, L3 locations and residual checks (CR3BP).
            % Usage:
            %   out = obj.getL1L2L3(name, m1, m2, a_km)
            %   out = obj.getL1L2L3(name, m1, m2, a_km, tol, maxit, verbose)
            %
            % Inputs
            %   name   : char label for console prints (e.g., 'Earth–Moon')
            %   m1,m2  : primary masses [kg]  (m2 is the smaller body by convention)
            %   a_km   : primary separation a* [km]
            %   tol    : NR step tolerance on gamma (default 1e-13)
            %   maxit  : NR max iterations (default 100)
            %   verbose: true to print a short report (default true)
            %
            % Requires the following methods in the same class:
            %   [gamma, iters] = obj.L1gammaNR(g0, mu, tol, maxit)
            %   [gamma, iters] = obj.L2gammaNR(g0, mu, tol, maxit)
            %   [gamma, iters] = obj.L3gammaNR(g0, mu, tol, maxit)

            if nargin < 7 || isempty(maxit),   maxit   = 100;     end
            if nargin < 6 || isempty(tol),     tol     = 1e-13;   end
            if nargin < 8 || isempty(verbose), verbose = true;    end

            % --- Mass ratio
            mu = m2/(m1+m2);

            % --- NR starting guesses (robust, system-agnostic)
            g2_0 = max(1e-12, (mu/3)^(1/3));   % L2 (to the right of small body)
            g1_0 = min(0.9,    max(1e-8,(mu/3)^(1/3))); % L1 (between primaries)
            g3_0 = 1 + (5/12)*mu;              % L3 (left of large body)

            % --- Solve quintics by Newton–Raphson (using class methods)
            [g1, it1] = obj.L1gammaNR(g1_0, mu, tol, maxit);
            [g2, it2] = obj.L2gammaNR(g2_0, mu, tol, maxit);
            [g3, it3] = obj.L3gammaNR(g3_0, mu, tol, maxit);

            % --- Barycentric x-locations (nondimensional)
            xL1_nd = 1 - mu - g1;
            xL2_nd = 1 - mu + g2;
            xL3_nd = -mu - g3;

            % --- Dimensional x-locations [km]
            xL1_km = xL1_nd * a_km;
            xL2_km = xL2_nd * a_km;
            xL3_km = xL3_nd * a_km;

            % --- Distances from relevant primary [km] and % of a*
            dSmall_to_L1_km = g1 * a_km;   pctL1 = 100*g1;
            dSmall_to_L2_km = g2 * a_km;   pctL2 = 100*g2;
            dLarge_to_L3_km = g3 * a_km;   pctL3 = 100*g3;

            % --- Raw residuals (no scaling)
            r1 = fL1(g1,mu);
            r2 = fL2(g2,mu);
            r3 = fL3(g3,mu);

            % --- Package outputs
            out.name  = string(name);
            out.mu    = mu;

            out.gamma1 = g1;  out.it1 = it1;
            out.gamma2 = g2;  out.it2 = it2;
            out.gamma3 = g3;  out.it3 = it3;

            out.xL1_nd = xL1_nd;  out.xL1_km = xL1_km;
            out.xL2_nd = xL2_nd;  out.xL2_km = xL2_km;
            out.xL3_nd = xL3_nd;  out.xL3_km = xL3_km;

            out.dSmall_to_L1_km = dSmall_to_L1_km;  out.pctL1 = pctL1;
            out.dSmall_to_L2_km = dSmall_to_L2_km;  out.pctL2 = pctL2;
            out.dLarge_to_L3_km = dLarge_to_L3_km;  out.pctL3 = pctL3;

            out.residuals.fL1 = r1;
            out.residuals.fL2 = r2;
            out.residuals.fL3 = r3;

            % --- Optional console report
            if verbose
                fprintf('%s\n', name);
                fprintf('  mu = %.9g\n', mu);

                fprintf('  L1: gamma1=%.12g  x(nd)=%.12g  x(km)=%.6f   it=%d\n', g1, xL1_nd, xL1_km, it1);
                fprintf('      |f_{L1}(gamma1)| = %.3e\n', abs(r1));

                fprintf('  L2: gamma2=%.12g  x(nd)=%.12g  x(km)=%.6f   it=%d\n', g2, xL2_nd, xL2_km, it2);
                fprintf('      |f_{L2}(gamma2)| = %.3e\n', abs(r2));

                fprintf('  L3: gamma3=%.12g  x(nd)=%.12g  x(km)=%.6f   it=%d\n', g3, xL3_nd, xL3_km, it3);
                fprintf('      |f_{L3}(gamma3)| = %.3e\n\n', abs(r3));
            end

            % ===== local helpers: exact quintic polynomials (raw residuals) =====
            function f = fL1(g,mu)
                f = g.^5 + (mu-3).*g.^4 + (3-2*mu).*g.^3 - mu.*g.^2 + 2*mu.*g - mu;
            end
            function f = fL2(g,mu)
                f = g.^5 + (3-mu).*g.^4 + (3-2*mu).*g.^3 - mu.*g.^2 - 2*mu.*g - mu;
            end
            function f = fL3(g,mu)
                f = g.^5 + (mu+2).*g.^4 + (2*mu+1).*g.^3 + (mu-1).*g.^2 + (2*mu-2).*g + (mu-1);
            end

        end



        % ----- Newton–Raphson for L1:  f(g)=0 with derivative f'(g) -----
        % f_L1(g) = g^5 + (mu-3)g^4 + (3-2mu)g^3 - mu g^2 + 2mu g - mu
        % f'_L1(g)= 5g^4 + 4(mu-3)g^3 + 3(3-2mu)g^2 - 2mu g + 2mu
        function [gamma, iters] = L1gammaNR(obj, g0, mu, tol, maxit)
            g = g0; iters = 0;
            for k = 1:maxit
                f  = g^5 + (mu-3)*g^4 + (3-2*mu)*g^3 - mu*g^2 + 2*mu*g - mu;
                df = 5*g^4 + 4*(mu-3)*g^3 + 3*(3-2*mu)*g^2 - 2*mu*g + 2*mu;
                step = f/df;
                gnew = g - step;
                iters = k;
                if abs(gnew - g) < tol, g = gnew; break; end
                g = gnew;
            end
            gamma = g;
        end

        function [gamma, iters] = L2gammaNR(obj, gamma0, mu, tol, maxit)
            % Newton–Raphson for the L2 quintic:
            % f(g) = g^5 + (3-mu)g^4 + (3-2mu)g^3 - mu*g^2 - 2mu*g - mu = 0

            g  = gamma0;
            it = 0;
            while it < maxit
                f  = g^5 + (3-mu)*g^4 + (3-2*mu)*g^3 - mu*g^2 - 2*mu*g - mu;
                df = 5*g^4 + 4*(3-mu)*g^3 + 3*(3-2*mu)*g^2 - 2*mu*g - 2*mu;
                gnew = g - f/df;
                it = it + 1;
                if abs(gnew - g) < tol
                    g = gnew;
                    break
                end
                g = gnew;
            end
            gamma = g;
            iters = it;
        end

        % ----- Newton–Raphson for L3:  f(g)=0 with derivative f'(g) -----
        % f_L3(g) = g^5 + (mu+2)g^4 + (2mu+1)g^3 + (mu-1)g^2 + (2mu-2)g + (mu-1)
        % f'_L3(g)= 5g^4 + 4(mu+2)g^3 + 3(2mu+1)g^2 + 2(mu-1)g + (2mu-2)
        function [gamma, iters] = L3gammaNR(obj, g0, mu, tol, maxit)
            g = g0; iters = 0;
            for k = 1:maxit
                f  = g^5 + (mu+2)*g^4 + (2*mu+1)*g^3 + (mu-1)*g^2 + (2*mu-2)*g + (mu-1);
                df = 5*g^4 + 4*(mu+2)*g^3 + 3*(2*mu+1)*g^2 + 2*(mu-1)*g + (2*mu-2);
                step = f/df;
                gnew = g - step;
                iters = k;
                if abs(gnew - g) < tol, g = gnew; break; end
                g = gnew;
            end
            gamma = g;
        end




        %% ===============================================================
        %% 3.12) Linear Orbits around Lagrange Points

        % ======================================================================
        % In-plane LI initial velocities (suppress unstable modes)
        % ======================================================================
        function [zetaDot0, etaDot0, out] = liInplaneICVelocities(obj, mu, xL, zeta0, eta0)
            % LIINPLANEICVELOCITIES  Compute (ζ̇0, η̇0) given (ζ0, η0) at a collinear point.
            % Inputs:
            %   mu, xL     : mass parameter and x-location of collinear point
            %   zeta0,eta0 : in-plane modal amplitudes
            % Outputs:
            %   zetaDot0, etaDot0 : corresponding initial rates
            %   out (struct)       : s, beta's, Uxx,Uyy for inspection

            H   = obj.pseudoPotentialHessian(xL, 0, 0, mu);
            Uxx = H(1,1); Uyy = H(2,2);

            beta1   = 2 - 0.5*(Uxx + Uyy);
            beta2sq = -Uxx*Uyy;

            % oscillation frequency s (stable planar pair)
            s = sqrt( max(0, beta1 + sqrt(beta1^2 + beta2sq)) );

            % β3 (see standard LI linear theory)
            beta3    = (s^2 + Uxx) / (2*s);

            % required initial velocities
            zetaDot0 = (eta0 * s) / beta3;
            etaDot0  = -beta3 * zeta0 * s;

            if nargout > 2
                out = struct('s',s,'beta1',beta1,'beta2',sqrt(beta2sq), ...
                    'beta3',beta3,'Uxx',Uxx,'Uyy',Uyy);
            end
        end








        %% ===============================================================
        %% 3.13) Stability


        %% Linearization eigenvalues at a collinear point (y=z=0)

        function [lambda, details] = collinearPointEigenvalues(obj, x, mu)
            % COLLINEARPOINTEIGENVALUES  Eigen-structure near collinear equilibrium.
            % Inputs:
            %   x      : x-coordinate of the collinear point (nd), y=z=0 assumed
            %   mu     : mass parameter
            % Outputs:
            %   lambda : 6×1 eigenvalues (4 planar + 2 vertical)
            %   details: struct with Uxx,Uyy,Uzz,beta1,beta2sq,Λ1,Λ2,planar,vertical

            H   = obj.pseudoPotentialHessian(x, 0, 0, mu);
            Uxx = H(1,1);  Uyy = H(2,2);  Uzz = H(3,3);

            % Planar characteristic: λ^4 + (4 - Uxx - Uyy) λ^2 + Uxx*Uyy = 0
            beta1   = 2 - 0.5*(Uxx + Uyy);
            beta2sq = -Uxx*Uyy;
            disc    = beta1.^2 + beta2sq;

            Lambda1 = -beta1 + sqrt(disc);
            Lambda2 = -beta1 - sqrt(disc);

            lamPlanar = [ +sqrt(complex(Lambda1));
                -sqrt(complex(Lambda1));
                +sqrt(complex(Lambda2));
                -sqrt(complex(Lambda2)) ];

            % Vertical pair:  λ^2 - Uzz = 0  -> λ = ± i*sqrt(-Uzz)
            lamVert = [ 1i*sqrt(max(0, -Uzz));
                -1i*sqrt(max(0, -Uzz)) ];


            lambda = [lamPlanar; lamVert];

            if nargout > 1
                details = struct('Uxx',Uxx,'Uyy',Uyy,'Uzz',Uzz, ...
                    'beta1',beta1,'beta2sq',beta2sq, ...
                    'Lambda1',Lambda1,'Lambda2',Lambda2, ...
                    'planar',lamPlanar,'vertical',lamVert);
            end
        end

        %% Monodromy matrix over one period
        function [X_T_nd, Phi6_T, Phi4_T] = computeMonodromy(obj, X0_nd, T_nd, mu, n, opts)
            % COMPUTEMONODROMY  Integrate augmented CR3BP to obtain Φ(T).
            % Inputs:
            %   X0_nd(6×1) : initial state (nd)
            %   T_nd       : period (nd)
            %   mu, n      : CR3BP params
            %   opts       : ODE options (no Events)
            % Outputs:
            %   X_T_nd     : state at t=T (6×1)
            %   Phi6_T     : 6×6 monodromy
            %   Phi4_T     : 4×4 planar block ([x y vx vy] rows/cols)

            Phi0  = eye(6);
            Xaug0 = [X0_nd(:); Phi0(:)];

            [~, Yaug] = ode89(@(t,X) obj.augmentedDynamicsCR3BP(t,X,mu,n), [0 T_nd], Xaug0, opts);

            X_T_nd = Yaug(end,1:6).';
            Phi6_T = reshape(Yaug(end,7:end).', 6, 6);
            Phi4_T = Phi6_T([1 2 4 5],[1 2 4 5]);
        end





        %% ===============================================================
        %% 3.14) Initial Guess Generation

        function vy_guess = vy0PredictPolyfit(obj, x_hist, vy_hist, x_next, deg)
            % VY0PREDICTPOLYFIT
            % Predict vy0 at a new x0 using polynomial fit on recent corrected points.
            % Inputs:
            %   x_hist  : recent x0 values (vector)
            %   vy_hist : recent vy0 values (vector)
            %   x_next  : target x0 to predict vy0 for
            %   deg     : polynomial degree
            % Output:
            %   vy_guess : predicted vy0 at x_next
            p = polyfit(x_hist(:), vy_hist(:), deg);
            vy_guess = polyval(p, x_next);
        end




    end
end
