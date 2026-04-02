classdef PlotterLibrary
    % PLOTTERLIBRARY  Plotting & visualization utilities for MORPHO studies.
    %
    %   Provides figure helpers and high-level plotters to visualize
    %   asteroid-interception analyses and attitude motion. Includes:
    %     • 2BP trajectory plotting (static or animated, follow-cam)
    %     • MP4 movie export with frame skipping and fixed window size
    %     • Dark-mode styling, start/end markers, top-down orthographic views
    %     • Rotating 3D box visualization from quaternion history
    %     • Illuminated ellipsoid bodies with texture/terminator shading
    %
    % AUTHOR
    %   Moacir Fonseca Becker
    %   Purdue University
    %
    % LAST MODIFIED
    %   10/26/2025
    %
    % NOTES
    %   - Units follow [km, km/s, s] for orbital plots (labels reflect units).    
    %   - Depends on KeplerianOrbitalMechanicsLibrary for 2BP dynamics.
    %   - Attitude visualizations expect an AttitudeDeterminationLibrary
    %     (for quaternion↔DCM conversions).
    %   - Recommended MATLAB: R2021b+ for VideoWriter performance.

    properties (Access = private)
        orb   KeplerianOrbitalMechanicsLibrary
    end

    methods
        function obj = PlotterLibrary(orbInstance)
            if nargin == 0
                obj.orb = KeplerianOrbitalMechanicsLibrary();  % default
            else
                obj.orb = orbInstance;
            end
        end
    end

    methods  
        
        %% ADC Plotting
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
  

        %% Create a Celestial Body Based on Ellipsoid Parameters
        function hSurface = createCelestialBodyEllipsoid(obj, A, B, C, sunDirection, varargin)
            % createCelestialBodyEllipsoid Simulates an illuminated ellipsoid with specified axes and position.
            %z
            %   createCelestialBodyEllipsoid(A, B, C, sunDirection)
            %   createCelestialBodyEllipsoid(A, B, C, sunDirection, 'Color', colorValue)
            %   createCelestialBodyEllipsoid(A, B, C, sunDirection, 'Texture', textureImage)
            %   createCelestialBodyEllipsoid(A, B, C, sunDirection, 'Position', [x, y, z])
            %
            % Inputs:
            %   A - Semi-axis length along the X-axis
            %   B - Semi-axis length along the Y-axis
            %   C - Semi-axis length along the Z-axis
            %   sunDirection - 3-element vector [x, y, z] specifying the Sun's direction
            %s
            % Optional Name-Value Pair Arguments:
            %   'Color'    - 1x3 RGB vector or color string for the ellipsoid's color
            %   'Texture'  - Filename of the texture image to apply to the ellipsoid
            %   'Position' - 1x3 vector [x, y, z] specifying the ellipsoid's location
            %
            % Example:
            %   createCelestialBodyEllipsoid(1737.4, 1737.4, 1737.4, [1e5, 0, 0], 'Texture', 'moon.png', 'Position', [1000, 2000, 3000]);

            % Parse optional inputs (case-insensitive)
            p = inputParser;
            p.CaseSensitive = false;
            addParameter(p, 'Color', [0.5, 0.5, 0.5], @(x) ischar(x) || (isnumeric(x) && numel(x)==3));
            addParameter(p, 'Texture', '', @ischar);
            addParameter(p, 'Position', [0, 0, 0], @(x) isnumeric(x) && numel(x)==3);
            parse(p, varargin{:});

            colorValue = p.Results.Color;
            textureImage = p.Results.Texture;
            position = p.Results.Position;

            % Mesh resolution of the ellipsoid
            meshRes = 1200;

            % Generate the ellipsoid data
            [X, Y, Z] = ellipsoid(0, 0, 0, A, B, C, meshRes); 

            % Shift ellipsoid to the specified position
            X = X + position(1);
            Y = Y + position(2);
            Z = Z + position(3);

            if ~isempty(textureImage)
                % Read and prepare texture image
                img = imread(textureImage);
                if size(img,3) == 1
                    img = repmat(im2double(img), [1, 1, 3]); % Convert grayscale to RGB
                else
                    img = im2double(img);
                end

                % Flip the image vertically to correct orientation
                img = flipud(img);
                img = imresize(img, [size(X,1), size(X,2)]); % Match meshgrid size

                % Plot surface with texture
                hSurface = surf(X, Y, Z, ...
                    'FaceColor', 'texturemap', ...
                    'EdgeColor', 'none', ...
                    'CData', img, ...
                    'FaceLighting', 'gouraud');
            else
                % Plot surface with specified color
                hSurface = surf(X, Y, Z, ...
                    'FaceColor', colorValue, ...
                    'EdgeColor', 'none', ...
                    'FaceLighting', 'gouraud');
            end

            % Adjust lighting properties to darken shadows and enhance terminator
            set(hSurface, ...
                'AmbientStrength', 0.0, ...    % Very low ambient light
                'DiffuseStrength', 0.99, ...    % High diffuse light
                'SpecularStrength', 0.0, ...   % Minimal specular highlights
                'SpecularExponent', 20);        % Sharper specular reflections


            % Remove existing lights and add single infinite light (Sun)
            delete(findall(gcf, 'Type', 'light'));
            light('Position', sunDirection, 'Style', 'infinite', 'Color', [1,1,1]);

            % Apply lighting settings
            lighting gouraud;

            hold off;
        end



        %% Orbital Mechanics Plotters

        function [bodies, hFig] = plot_2BP_trajectories(obj, bodies, central, mu_central, animate, varargin)
            % plot_2BP_trajectories  Propagate (or accept pre-propagated histories) and
            % plot one-or-many trajectories in a two-body problem, with an optional
            % animated "follow-cam" movie.
            %
            % -------- Required Inputs -------------------------------------------------
            %   bodies{k}.name   – label string
            %   bodies{k}.IC     – 1×6 Cartesian state  [x y z vx vy vz]  (km | km s⁻¹)
            %                      **omit** when you pass X_hist / t_hist and set
            %                      'StatesProvided',true.
            %   bodies{k}.tspan  – scalar tf  *or*  vector of epochs (s)   (ignored
            %                      when X_hist is given or 'FixedTspan' is supplied)
            %   central          – 'Sun', 'Earth', …  (gives the focus its colour & marker)
            %   mu_central       – GM of the focus [km³ s⁻²]
            %   animate          – logical.  true  → live animated trace + MP4
            %
            % -------- Name-Value Pairs -------------------------------------------------
            %   'Title'          – figure title (default: "Trajectories about <central>")
            %   'FixedTspan'     – force the *same* propagation span for every body
            %   'OdeOptions'     – options structure for ODE45
            %                      (default: RelTol = AbsTol = 1 e-13)
            %   'DarkMode'       – true ⇒ black background, neon lines, white text
            %   'ShowStartEnd'   – true ⇒ put "start / end" markers + labels
            %   'StatesProvided' – true ⇒ caller already supplied
            %                      bodies{k}.X_hist  (N×6)  & (optionally) t_hist.
            %                      In this mode no ODE integration is performed.
            %
            %   --- NEW FOLLOW-CAM / MOVIE OPTIONS -------------------------------------
            %   'FollowBody'     – which body to keep centred during the animation:
            %                      • ''      (default)  ⇒ no follow-cam
            %                      • index   (integer)  ⇒ bodies{index}
            %                      • string  (name)     ⇒ first match to bodies{k}.name
            %   'Zoom'           – half-width of the fixed axis box [km] around the
            %                      FollowBody.  If empty (default) the first global
            %                      plot limits are used.
            %   'MaxFrames'      – maximum number of video frames to write
            %                      (skips steps automatically, default = 1 500)
            %
            % -------- Returns ----------------------------------------------------------
            %   bodies           – same cell array, but with fields .X_hist and .t_hist
            %                      filled in for any object that was propagated inside
            %                      the function.
            %


            %–––––––––––––––––––– 1. INPUT PARSING ––––––––––––––––––––––––––––
            p = inputParser;  p.CaseSensitive = false;
            addRequired( p,'bodies',     @(x) iscell(x) );
            addRequired( p,'central',    @(x) ischar(x)||isstring(x) );
            addRequired( p,'mu_central', @isnumeric );
            addRequired( p,'animate',    @islogical );
            addParameter(p,'Title',        '',  @(x) ischar(x)||isstring(x));
            addParameter(p,'FixedTspan',   [],  @isnumeric);
            addParameter(p,'OdeOptions',   odeset('RelTol',1e-13,'AbsTol',1e-13), @isstruct);
            addParameter(p,'DarkMode',     false, @islogical);
            addParameter(p,'ShowStartEnd', false, @islogical);
            addParameter(p,'StatesProvided',false,@islogical);
            addParameter(p,'FollowBody','', @(x) isnumeric(x) || ischar(x) || isstring(x));
            addParameter(p,'Zoom',[],@isnumeric);  
            addParameter(p,'MaxFrames',1500,@isnumeric);   % smoother skip logic
            addParameter(p,'TopDown',false,@islogical);   % new switch

        
            parse(p,bodies,central,mu_central,animate,varargin{:});
            

            figTitle    = p.Results.Title;
            fixed_tspan = p.Results.FixedTspan;
            odeOpt      = p.Results.OdeOptions;
            darkMode    = p.Results.DarkMode;
            showSE      = p.Results.ShowStartEnd;
            statesProvided = p.Results.StatesProvided;   
            followBody = p.Results.FollowBody;   % string | index | ''
            zoomBox    = p.Results.Zoom;         % scalar or []
            maxFrames  = p.Results.MaxFrames;
            topDown = p.Results.TopDown;



            if isempty(figTitle)
                figTitle = sprintf('Trajectories about %s',central);
            end

            nB = numel(bodies);

            %––– workout which body to follow –––––
            followIdx = [];
            if ~isempty(followBody)
                if isnumeric(followBody)
                    followIdx = followBody;
                else
                    followIdx = find(strcmpi(followBody, cellfun(@(b) b.name, bodies, 'uni',0)), 1);
                end
                if isempty(followIdx)
                    warning('FollowBody "%s" not found ‒ follow-cam disabled.',string(followBody));
                end
            end


            %––– choose colours & figure/text colours based on mode –––––
            if darkMode
                cols     = hsv(nB);   % neon palette
                figColor = 'k';
                fgColor  = 'w';
            else
                cols     = lines(nB);
                figColor = 'w';
                fgColor  = 'k';
            end

            %–––––––––––––––––––– 2. PROPAGATION ––––––––––––––––––––––––––––––
            for k = 1:nB


                %–––– fill in a dummy IC if histories are supplied but IC is absent ––––
                if statesProvided
                    assert(isfield(bodies{k},'X_hist') && ~isempty(bodies{k}.X_hist), ...
                        'StatesProvided==true but bodies{%d}.X_hist is missing/empty',k)

                    if ~isfield(bodies{k},'IC') || isempty(bodies{k}.IC)
                        bodies{k}.IC = bodies{k}.X_hist(1,1:6);   % first state = IC
                    end

                    % store t_hist if the caller forgot it
                    if ~isfield(bodies{k},'t_hist') || isempty(bodies{k}.t_hist)
                        bodies{k}.t_hist = NaN(size(bodies{k}.X_hist,1),1);
                    end

                    continue                                    % <-- skip integration
                end


                %–––– default path: integrate if no history present –––––
                assert(isfield(bodies{k},'IC') && ~isempty(bodies{k}.IC), ...
                       'Body %d needs an IC field when StatesProvided==false',k)

                IC = bodies{k}.IC(:);
                
                ts = obj.iff(~isempty(fixed_tspan), ...
                    fixed_tspan, ...
                    obj.iff(isfield(bodies{k},'tspan') && ~isempty(bodies{k}.tspan), ...
                    bodies{k}.tspan, 86400));
                if isscalar(ts), ts = [0 ts]; end
                
                if ~isfield(bodies{k},'X_hist') || isempty(bodies{k}.X_hist)
                    [tHist,Xhist]    = ode45(@(t,X) obj.orb.dynamics_2BP_cartesian(t,X,mu_central),ts,IC,odeOpt);
                    bodies{k}.t_hist = tHist;
                    bodies{k}.X_hist = Xhist;
                end

            end

            %–––––––––––––––––––– 3. BOUNDING BOX –––––––––––––––––––––––––––––
            xyzCells = cellfun(@(b) b.X_hist(:,1:3), bodies, 'UniformOutput', false);
            xyz      = vertcat(xyzCells{:});
            mn       = min(xyz,[],1);
            mx       = max(xyz,[],1);
            pad      = 0.15*(mx-mn);
            mn       = mn - pad;
            mx       = mx + pad;

            %–––––––––––––––––––– 4. FIGURE & STATIC PLOT –––––––––––––––––––––
            hFig = figure('Name',figTitle, ...
                'Units','normalized','OuterPosition',[0 0 1 1], ...
                'Color',figColor);
            baseFS = 12;
            baseLW = 2.0;
            baseMS = 5;
            set(hFig, ...
                'defaultAxesFontSize',   baseFS, ...
                'defaultTextFontSize',   baseFS, ...
                'defaultLineLineWidth',  baseLW, ...
                'defaultLineMarkerSize', baseMS, ...
                'defaultTextColor',      fgColor, ...
                'defaultAxesXColor',     fgColor, ...
                'defaultAxesYColor',     fgColor, ...
                'defaultAxesZColor',     fgColor);

            clf; hold on; grid on;

            ax = gca;
            
            if darkMode
                set(ax, 'Color',figColor, 'GridColor',fgColor);
            end

            if topDown
                view(0,90);          % look down the +Z axis
                set(ax,'ZLim',[-1 1])   % thin slab, never touched again
            end

            for k = 1:nB
                if animate
                    plot3(nan,nan,nan, ...
                        'Color',cols(k,:), ...
                        'DisplayName',bodies{k}.name, ...
                        'LineWidth',baseLW);
                else
                    X = bodies{k}.X_hist;
                    plot3(X(:,1), X(:,2), X(:,3), 'Color',cols(k,:), 'DisplayName',bodies{k}.name, 'LineWidth',baseLW);
                    if showSE
                        % start marker and label
                        plot3(X(1,1),X(1,2),X(1,3),'o','MarkerFaceColor',cols(k,:),'MarkerEdgeColor',fgColor,'MarkerSize',baseMS*1.3,'HandleVisibility','off');
                        text(   X(1,1),X(1,2),X(1,3),sprintf('%s start',bodies{k}.name), 'Color',cols(k,:),'FontSize',baseFS,...
                            'HorizontalAlignment','left','VerticalAlignment','bottom','Interpreter','none');
                        % end marker and label
                        plot3(X(end,1),X(end,2),X(end,3),'o','MarkerFaceColor','none','MarkerEdgeColor',cols(k,:),'LineWidth',baseLW,'MarkerSize',baseMS*1.3,'HandleVisibility','off');
                        text(   X(end,1),X(end,2),X(end,3),sprintf('%s end',bodies{k}.name),'Color',cols(k,:),'FontSize',baseFS,...
                            'HorizontalAlignment','left','VerticalAlignment','top','Interpreter','none');
                    end

                end
            end

            %–––––––––––––––––––– 5. DECORATION –––––––––––––––––––––––––––––––
            cCent = strcmpi(central,'sun')   * [1 0.85 0] + ...
                strcmpi(central,'earth')*[0 0.45 1] + ...
                (~strcmpi(central,'sun') & ~strcmpi(central,'earth')) * 0.7*[1 1 1];
            plot3(0,0,0,'o','MarkerFaceColor',cCent,'MarkerEdgeColor',fgColor,'DisplayName',central);

            axis equal;
            axis(ax,'equal');
            ax.BoxStyle = 'full';
            ax.Layer    = 'top';

            xlim([mn(1) mx(1)]);
            ylim([mn(2) mx(2)]);
            zlim([mn(3) mx(3)]);
            set(ax,'XLimMode','manual','YLimMode','manual','ZLimMode','manual');

            xlabel('X [km]','FontSize',baseFS,'Color',fgColor);
            ylabel('Y [km]','FontSize',baseFS,'Color',fgColor);
            zlabel('Z [km]','FontSize',baseFS,'Color',fgColor);

            set(ax,'Units','normalized','Position',[0.08 0.12 0.84 0.78]);

            annotation(hFig,'textbox',[0.19 0.8 0.35 0.01], ...
                'String',figTitle, ...
                'FitBoxToText','on', ...
                'Interpreter','tex', ...
                'FontSize',baseFS, ...
                'FontWeight','bold', ...
                'EdgeColor','none', ...
                'HorizontalAlignment','left', ...
                'VerticalAlignment','top', ...
                'Color',fgColor);

            lg = legend('Location','southwestoutside','FontSize',baseFS);
            if darkMode
                set(lg,'TextColor',fgColor,'Color','none');
            end

            %–––––––––––––––––––– 6. ANIMATION ––––––––––––––––––––––––––––––––
            if ~animate
                set(hFig,'Units','pixels','Position',[100 100 800 600]);  % 800×600 only for static plot
                camproj('orthographic'); view(0,90);
                return
            end

            camproj('perspective'); view(0,90);            
            set(hFig,'Units','pixels','Position',[100 100 1024 576]);   % << fixed size
            drawnow;

            % camproj('perspective'); view(0,90);
            % set(hFig,'Units','pixels','Position',[0 0 854 480]); drawnow;

            % -----------------------------------------------------------------
            %     VIDEO WRITER
            % -----------------------------------------------------------------
            v = VideoWriter('orbitAnimation.mp4','MPEG-4');
            v.FrameRate = 30;           % fps seen during playback
            v.Quality   = 100;          % (best)
            open(v);

            % -----------------------------------------------------------------
            %     INITIALIZE animatedline/markers/labels
            % -----------------------------------------------------------------
            al = gobjects(nB,1); mk = gobjects(nB,1); lbl = gobjects(nB,1);
            
            
            Nmax      = max(cellfun(@(b) size(b.X_hist,1), bodies));
            stepF     = max(1, ceil(Nmax / maxFrames));                     % keeps ≤ MaxFrames frames


            for k = 1:nB
                al(k) = animatedline('Color',cols(k,:),'LineWidth',baseLW,'HandleVisibility','off');
                P0    = bodies{k}.X_hist(1,1:3);
                mk(k) = plot3(P0(1),P0(2),P0(3),'o','MarkerFaceColor',cols(k,:), ...
                    'MarkerEdgeColor','none','MarkerSize',6,'HandleVisibility','off');
                lbl(k)= text(P0(1),P0(2),P0(3),bodies{k}.name, ...
                    'HorizontalAlignment','left','VerticalAlignment','bottom', ...
                    'HandleVisibility','off','Color',fgColor);
            end

            % -----------------------------------------------------------------
            % MAIN ANIMATION LOOP

            for i = 1:stepF:Nmax
                % -- update all bodies ------------------------------------------------
                for k = 1:nB
                    Xi = bodies{k}.X_hist;
                    if i <= size(Xi,1)
                        addpoints(al(k),Xi(i,1),Xi(i,2),Xi(i,3));
                        set(mk(k),'XData',Xi(i,1),'YData',Xi(i,2),'ZData',Xi(i,3));
                        set(lbl(k),'Position',Xi(i,1:3));
                    end
                end

                % ----- FOLLOW-CAM (optional) -------------------------------------------
                if ~isempty(followIdx)
                    
                    ctr = bodies{followIdx}.X_hist(min(i,size(bodies{followIdx}.X_hist,1)),1:3);

                    if isempty(zoomBox)      % auto-zoom = first computed half-span
                        zoomBox = max(diff(get(ax,'XLim'))/2,1);   % fallback
                    end
                    
                    set(ax,'XLim',ctr(1)+zoomBox*[-1 1], ...
                        'YLim',ctr(2)+zoomBox*[-1 1]);
      
                    if ~topDown
                        % only update Z when NOT in 2-D mode
                        set(ax,'ZLim',ctr(3)+zoomBox*[-1 1]);
                    end
                end
                % ------------------------------------------------------------

                % drawnow;
                % writeVideo(v,getframe(ax));

                drawnow limitrate nocallbacks;         % smoother & faster GUI update
                writeVideo(v,getframe(hFig));          % grab the whole window

            end

            close(v);
        end

        %–––––––––––––––––––– helper ––––––––––––––––––––––––––––––––––––––
        function y = iff(obj,c,a,b), if c, y=a; else, y=b; end
        end


    
    end
end 

    