classdef OpticalLibrary
    % OPTICALLIBRARY  Optical navigation & rendering utilities for vision-based GNC.
    %
    %   High-level tools to (a) synthesize camera views of a body (sphere/ellipsoid),
    %   (b) detect/fit the limb, (c) derive attitude/position using CRA-style
    %   methods, and (d) animate full imaging passes with optional photometrics.
    %   Designed to interoperate with CR3BPLibrary (for states/frames) and a
    %   plotting helper that can draw textured ellipsoids.
    %
    % -------------------------------------------------------------------------
    % KEY CAPABILITIES
    % -------------------------------------------------------------------------
    %   Image synthesis & annotation
    %     • simulateCameraImage
    %         Render an off-screen perspective view of a body from S/C pose with
    %         optional limb detection & conic fitting; returns image, limb points,
    %         apparent radius/diameter, and fitted conic parameters.
    %
    %   Scene/axes visualization
    %     • plotCameraAxesAtTimestep
    %         3-D plot of planet ellipsoid, S/C trajectory up to t_k, camera axes,
    %         and horizon vectors back-projected from image pixels.
    %
    %   Camera model
    %     • calculateFocalLength
    %         Pinhole model: compute (f_x, f_y) [pixels] from horizontal FOV and
    %         sensor resolution (W,H).
    %     • convert_pixelToNormalized_ImagePlane
    %         Pixel → normalized image-plane coordinates using intrinsics.
    %
    %   Limb & conic fitting
    %     • detectLimb
    %         Canny-based edge extraction + least-squares circle fit → limb points,
    %         apparent radius [px], limb center [px].
    %     • fitCircle
    %         Algebraic least-squares circle fit.
    %     • fitConic
    %         Algebraic fit of general conic A x^2 + B x y + C y^2 + D x + F y + G = 0
    %         → symmetric 3×3 matrix and coefficient vector.
    %
    %   CR3BP camera fly-by animation
    %     • animateCameraOrbit
    %         Walk along a provided state history, render frame-by-frame images
    %         (optionally with dynamic Sun direction), run limb detection, save
    %         video (.mp4) and a .mat bundle with all outputs and camera intrinsics.
    %
    %   Geometry / estimation
    %     • computeApparentAngularDiameter
    %         Theoretical apparent diameter 2*asin(R_p / ρ) [rad].
    %     • compute_T_CP
    %         Build camera axes pointing from S/C to body and return the DCM
    %         (synodic→camera) with [x_c y_c z_c] as columns.
    %     • estimateCRA_SpacecraftPosition
    %         Christian-Robinson Algorithm (CRA) position solution. Optionally
    %         propagates measurement uncertainty (σ_pix) to R_s (image cov),
    %         P_n (aux. var), and position covariance P_r via linearization.
    %     • computeCRA_Trajectory
    %         Apply CRA over a limb-point time series to produce an estimated S/C
    %         trajectory and per-epoch covariances (planet frame / synodic).
    %     • plotCRA_TrajectoryComparison
    %         Compare true vs CRA-estimated trajectory in 3-D and plot ΔX,ΔY,ΔZ.
    %     • computeAttitudeConicLocus
    %         Recover attitude candidates T_C_P from a horizon conic and known
    %         position r_P on a tri-axial ellipsoid (up to two chirality-valid
    %         solutions).
    %
    % -------------------------------------------------------------------------
    % UNITS, FRAMES & CONVENTIONS
    % -------------------------------------------------------------------------
    %   • Camera intrinsics (f_x,f_y,u_0,v_0) are in pixels. FOV is horizontal.
    %   • simulateCameraImage accepts body radius R_p or ellipsoid axes (A,B,C);
    %     SunDirection is a 3-vector (any frame consistent with the plotting lib).
    %   • compute_T_CP returns T_CP whose columns are camera axes [x_c y_c z_c];
    %     use r_syn = T_CP * r_c to rotate camera-frame vectors into synodic.
    %
    % -------------------------------------------------------------------------
    % DEPENDENCIES
    % -------------------------------------------------------------------------
    %   • plotterlib.createCelestialBodyEllipsoid(...) – textured ellipsoid mesh.
    %   • CR3BPLibrary (optional) – for sun-moon synodic vectors & state sampling.
    %   • Image Processing Toolbox (recommended) – edge detection, imadjust, etc.
    %
    % -------------------------------------------------------------------------
    % QUICK START
    % -------------------------------------------------------------------------
    %   % 1) Render a single frame with limb detection
    %   opt = OpticalLibrary();
    %   camFOV = 25; res = [800 800];
    %   Rm = 1737.4/384400;                            % Moon radius [nd]
    %   sc = [1.02 0.01 0.02];  moon = [1-0.01215 0 0];
    %   [img, limb, rpx, ddeg, fig] = opt.simulateCameraImage( ...
    %       plotterlib, Rm, sc, moon, camFOV, res, ...
    %       'DetectLimb', true, 'ShowFigure', true, 'Texture','moon.tif');
    %
    %   % 2) Convert pixels → normalized rays then to planet frame
    %   cam.f_x = res(1)/(2*tan(deg2rad(camFOV)/2));
    %   cam.f_y = cam.f_x * (res(2)/res(1));
    %   cam.u_0 = res(1)/2; cam.v_0 = res(2)/2;
    %   [T_CP,~,~,~] = opt.compute_T_CP(sc(:), moon(:));
    %   s_norm = opt.convert_pixelToNormalized_ImagePlane(limb, cam);
    %   raysP  = T_CP * [s_norm.'; ones(1,size(s_norm,1))];   % 3×N
    %
    %   % 3) CRA position at one epoch (ellipsoid a=b=c=Rm for sphere)
    %   A_P = diag([1/Rm^2 1/Rm^2 1/Rm^2]);
    %   r_c = opt.estimateCRA_SpacecraftPosition(A_P, limb, cam, T_CP);
    %
    %   % 4) Animate along a provided state history (saves .mp4/.mat if enabled)
    %   [LC, Rpx, tau, ddeg_all, tvec] = opt.animateCameraOrbit( ...
    %       cr3bplib, plotterlib, Rm, moon, T, Xhist, ...
    %       'NumFrames', 60, 'DetectLimb', true, 'ShowAnnotations', true);
    %
    % -------------------------------------------------------------------------
    % METHOD DETAILS (signatures)
    % -------------------------------------------------------------------------
    %   [image, limbXY, r_px, d_deg, fig, note, resUsed, Cmat, coeffs] =
    %       simulateCameraImage(plotterlib, R_p, scPos, bodyPos, camFOV_deg, resWH, ...);
    %
    %   plot = plotCameraAxesAtTimestep(plotterLib, A,B,C, sunDir, bodyPos, scXYZ_hist,
    %                                   limbCoordsAll, camParams, k, ...);
    %
    %   [f_x,f_y] = calculateFocalLength(FOV_deg, [W H]);
    %   [limbXY, r_px, ctr] = detectLimb(rgbImage);
    %   [ctr, r] = fitCircle(x,y);
    %   [Cmat, coeffs] = fitConic(x,y);
    %
    %   [LC_all, rpx_all, tau_all, ddeg_all, tvec, resUsed, camParams] =
    %       animateCameraOrbit(cr3bplib, plotterlib, R_p, bodyPos, T, Xhist, ...);
    %
    %   d_alpha = computeApparentAngularDiameter(R_p, scPos, bodyPos);  % [rad]
    %
    %   [r_c, R_s, P_r] = estimateCRA_SpacecraftPosition(A_P, limbXY, camParams, T_CP, ...);
    %   s_i = convert_pixelToNormalized_ImagePlane([u v], camParams);
    %   [T_CP, x_c, y_c, z_c] = compute_T_CP(scPos, bodyPos);
    %
    %   plotCRA_TrajectoryComparison(plotterLib, A,B,C, sunDir, bodyPos, scTrueXYZ, estXYZ, l_star);
    %
    %   [estXYZ, camAxes, P_r_all, R_s_all] =
    %       computeCRA_Trajectory(A_P, limbCoordsAll, camParams, scTrueXYZ, bodyPos, ...);
    %
    %   Tlist = computeAttitudeConicLocus(C_matrix, r_P, a,b,c);  % up to two solutions
    %
    % -------------------------------------------------------------------------
    % NOTES & CAVEATS
    % -------------------------------------------------------------------------
    %   • simulateCameraImage renders off-screen at requested resolution; MATLAB
    %     may return slightly different pixel dims (returned as adjustedResolution).
    %   • Limb detection uses fixed Canny thresholds by default; scenes with low
    %     contrast/terminator may require tuning or custom preprocessing.
    %   • CRA linearized covariance assumes small pixel noise and a well-conditioned
    %     limb point distribution (wide azimuthal coverage). Singular geometries
    %     (near-pole views, grazing) can inflate covariances.
    %   • Keep frames consistent: T_CP maps synodic→camera with columns [x_c y_c z_c].
    %     When converting positions/covariances between frames, apply T and T'.
    %   • Ellipsoid axes can be passed via 'EllipsoidAxes',[a b c] in rendering to
    %     support non-spherical bodies; CRA expects A_P = diag(1/a^2,1/b^2,1/c^2).
    %
    % AUTHOR
    %   Moacir Fonseca Becker
    %   Purdue University
    %
    % LAST MODIFIED
    %   03/11/2025

    methods

        %% Simulate a camera image
        function [imageData, limbCoordinates, apparentRadius, apparentAngularSize, fig, fullAnnotationText, adjustedResolution, C_matrix, coeffs] = simulateCameraImage(...
                obj, plotterlib, R_p, spacecraftPosition, bodyPosition, cameraFOV, resolution, varargin)
            % Simulates the celestial body as seen by the camera on the spacecraft
            % Optionally performs limb detection and annotations
            % Returns the image data and limb detection results

            % - - - - - - Parse optional parameters - - - - - -
            p = inputParser;
            p.KeepUnmatched = true; % Keep unmatched parameters
            addParameter(p, 'DetectLimb', false, @islogical);
            addParameter(p, 'ShowAnnotations', false, @islogical);
            addParameter(p, 'ShowFigure', false, @islogical);
            addParameter(p, 'SunDirection', [1, 0, 0], @(x) isnumeric(x) && numel(x)==3);
            addParameter(p, 'EllipsoidAxes', [], @(x) isnumeric(x) && numel(x)==3);

            parse(p, varargin{:});

            detectLimb = p.Results.DetectLimb;
            showAnnotations = p.Results.ShowAnnotations;
            showFigure = p.Results.ShowFigure;
            sunDirection = p.Results.SunDirection;
            ellipsoidAxes = p.Results.EllipsoidAxes;

            remainingParams = namedargs2cell(p.Unmatched); % Remaining parameters to pass to createCelestialBodyEllipsoid
            % - - - - - - Parse optional paramcaeters - - - - - -

            % Characteristic length #update
            l_star = 3.8475e5;

            % - - - - - - Determine Ellipsoid Axes - - - - - -
            if ~isempty(ellipsoidAxes)
                A = ellipsoidAxes(1);
                B = ellipsoidAxes(2);
                C = ellipsoidAxes(3);
            else
                A = R_p;
                B = R_p;
                C = R_p;
            end
            % - - - - - - Determine Ellipsoid Axes - - - - - -

            % Create an off-screen figure with specified resolution
            fig = figure('Units', 'pixels', 'Position', [0, 0, resolution(1), resolution(2)], ...
                'Color', 'black', 'MenuBar', 'none', 'ToolBar', 'none', 'Visible', 'off');
            set(fig, 'InvertHardCopy', 'off');
            ax = axes('Parent', fig, 'Units', 'normalized', 'Position', [0, 0, 1, 1]);
            hold(ax, 'on');
            axis(ax, 'off', 'equal','tight');
            set(ax, 'Color', 'black');

            % Set up camera properties
            camproj(ax, 'perspective');
            campos(ax, spacecraftPosition');
            camtarget(ax, bodyPosition');
            camva(ax, cameraFOV);
            camup(ax, [0, 0, 1]); % Fix the up vector to a stable direction % #test


            % Adjust camera properties to prevent auto-adjustment
            set(ax, 'CameraViewAngleMode', 'manual', ...
                'CameraPositionMode', 'manual', ...
                'CameraTargetMode', 'manual', ...
                'CameraUpVectorMode', 'manual');

            % Plot the celestial body with remaining parameters (e.g., 'Texture')
            plotterlib.createCelestialBodyEllipsoid(A, B, C, ...
                sunDirection, 'Position', bodyPosition, remainingParams{:});

            % Ensure aspect ratio is correct
            daspect(ax, [1 1 1]);
            hold(ax, 'off');

            % Capture the high-resolution image data using print
            tempFileName = [tempname, '.png'];
            % Set figure 'PaperPositionMode' to 'auto' to preserve the figure size
            set(fig, 'PaperPositionMode', 'auto');
            % Use print to save the figure at the specified resolution
            print(fig, tempFileName, '-dpng', sprintf('-r%d', 96)); % DPI to 96
            % Read the image back into MATLAB
            imageData = imread(tempFileName);
            % Delete the temporary file
            delete(tempFileName);
            % Close the off-screen figure
            close(fig);

            % Recompute the actual resolution used of the image (MATLAB changes it)
            [imageHeight, imageWidth, ~] = size(imageData);
            adjustedResolution = [imageHeight, imageWidth];

            % Initialize outputs
            limbCoordinates = [];
            apparentRadius = NaN;
            apparentAngularSize = NaN;

            % If limb detection is requested
            if detectLimb
                % Perform limb detection
                [limbCoordinates, apparentRadius, center] = obj.detectLimb(imageData);
                xlimbCoordinates = limbCoordinates(:,1); % [px]
                ylimbCoordinates = limbCoordinates(:,2); % [px]

                % Fit a conic to the edge points
                [C_matrix, coeffs] = obj.fitConic(xlimbCoordinates, ylimbCoordinates);

                % Compute apparent angular size
                N = imageHeight; % Use actual image height

                % Compute apparent angular size
                FOV_rad = deg2rad(cameraFOV);
                f_pixels = (N / 2) / tan(FOV_rad / 2);

                if ~isnan(apparentRadius) && ~isempty(apparentRadius)
                    % Compute apparent angular diameter from limb detection
                    apparentAngularDiameter_rad = 2 * atan2(apparentRadius, f_pixels);
                    apparentAngularSize = rad2deg(apparentAngularDiameter_rad); % In degrees
                end
            else
                C_matrix = [];
                coeffs = [];
            end


            % Compute the real distance to the body
            trueDistance_nd = norm(spacecraftPosition - bodyPosition);
            trueDistance_km = trueDistance_nd * l_star;

            % Compute theoretical apparent angular diameter
            d_alpha = obj.computeApparentAngularDiameter(R_p, spacecraftPosition, bodyPosition);
            d_alpha_deg = rad2deg(d_alpha);

            % If ShowFigure is true, display the image at a manageable size
            if showFigure
                % Create a new figure for display
                fig = figure('Name', 'Simulated Camera Image', 'NumberTitle', 'off','Color', 'black');

                % Display the image using imshow with 'InitialMagnification' set to fit
                imshow(imageData, 'InitialMagnification', 'fit');

                % If limb detection is requested, overlay the detections
                if detectLimb
                    hold on;
                    % Plot limb detection overlays
                    plot(limbCoordinates(:,1), limbCoordinates(:,2), 'r+', 'MarkerSize', 2.5);

                    if ~isempty(limbCoordinates)
                        % Draw convex hull
                        k = convhull(limbCoordinates(:,1), limbCoordinates(:,2));
                        plot(limbCoordinates(k,1), limbCoordinates(k,2), 'g-', 'LineWidth', 1.5, 'LineStyle', '--');

                        % Generate a grid over the image
                        [xGrid, yGrid] = meshgrid(1:imageWidth, 1:imageHeight);

                        % Evaluate the conic equation on the grid
                        conicEq = coeffs(1)*xGrid.^2 + coeffs(2)*xGrid.*yGrid + coeffs(3)*yGrid.^2 + ...
                            coeffs(4)*xGrid + coeffs(5)*yGrid + coeffs(6);

                        % Plot the zero-contour of the conic equation
                        contour(xGrid, yGrid, conicEq, [0 0], 'LineColor', 'cyan', 'LineWidth', 1.5, 'LineStyle', '-.');

                    end

                    hold off;
                end

                % If ShowAnnotations is true, add annotations on the figure
                if showAnnotations
                    % Combine annotation text
                    annotationText = sprintf('Theoretical Diameter: %.5f°', d_alpha_deg);
                    if detectLimb && ~isnan(apparentAngularSize)
                        annotationText = sprintf('%s\nDetected Diameter: %.5f°', annotationText, apparentAngularSize);
                    end

                    % Add FOV and Resolution information to the annotation text
                    fovResolutionText = sprintf('FOV: %d°\nResolution: %dx%d', cameraFOV, resolution(1), resolution(2));
                    trueDistanceText = sprintf('SC Distance: %.5f km', trueDistance_km);
                    fullAnnotationText = sprintf('%s\n%s\n%s', annotationText, trueDistanceText, fovResolutionText);

                    % Add text annotation directly to the axes
                    position = [10, 10]; % pixels from top-left corner
                    text(position(1), position(2), fullAnnotationText, ...
                        'Color', 'yellow', 'FontSize', 12, 'FontWeight', 'bold', ...
                        'HorizontalAlignment', 'left', 'VerticalAlignment', 'top');
                else
                    fullAnnotationText = '';
                end
            else
                fullAnnotationText = '';
            end
        end


        % Plot Camera and Planet Frame at Timestep
        function [plot, T_CP, x_c, y_c, z_c] = plotCameraAxesAtTimestep(obj, plotterLib, A, B, C, sunDirection, planetPosition, scTruePositions, limbCoordinatesAll, camParameters, timestep, varargin)
            % plotSpacecraftSnapshot: Plots the Moon, spacecraft trajectory up to a specific timestep,
            % camera orientation, and optionally horizon vectors at that timestep.
            %
            % Parameters:
            %   plotterLib          - Object or struct with method createCelestialBodyEllipsoid.
            %   opticalLib          - Object or struct with method pixelToNormalizedImageCoordinates.
            %   A, B, C             - Scalars representing the Moon's semi-principal axes.
            %   sunDirection        - 3-element vector indicating the Sun's direction.
            %   planetPosition        - 3-element vector specifying the Moon's position [nd].
            %   scTruePositions     - [numTimesteps x 3] array of spacecraft positions [nd].
            %   limbCoordinatesAll  - Cell array containing [N x 2] limb coordinates per timestep.
            %   camParameters       - Struct with camera intrinsic parameters:
            %                          .f_x, .f_y - Focal lengths in pixels.
            %                          .u_0, .v_0 - Principal point coordinates (pixels).
            %   timestep            - Integer specifying the timestep to plot.
            %
            % Optional Name-Value Pair Arguments:
            %   'PlotHorizonVectors' - (Boolean) Whether to plot horizon vectors. Default: true.
            %   'NumHorizonVectors'  - (Integer) Number of horizon vectors to plot. Default: 50.
            %
            % Example:
            %   plotSpacecraftSnapshot(plotterLib, opticalLib, A, B, C, sunDir, moonPos, ...
            %       scPos, limbCoordsAll, camParams, timestep, 'PlotHorizonVectors', true, 'NumHorizonVectors', 30)

            %% Parse Optional Inputs
            p = inputParser;
            addParameter(p, 'PlotHorizonVectors', true, @(x) islogical(x) || isnumeric(x));
            addParameter(p, 'NumHorizonVectors', 50, @(x) isnumeric(x) && x > 0 && floor(x) == x);
            parse(p, varargin{:});

            plotHorizonVectors = logical(p.Results.PlotHorizonVectors);
            numHorizonVectors = p.Results.NumHorizonVectors;

            Colors = [
                0.95  0.27  0.21;  % Colors(1,:) - Trajectory - Bright Coral Red
                1.00  0.00  0.00;  % Colors(2,:) - Moon X-axis - Strong Red
                0.00  1.00  0.00;  % Colors(3,:) - Moon Y-axis - Strong Green
                0.00  0.00  1.00;  % Colors(4,:) - Moon Z-axis - Strong Blue
                0.70  0.00  0.00;  % Colors(5,:) - Camera X-axis - Dark Red
                0.00  0.70  0.00;  % Colors(6,:) - Camera Y-axis - Dark Green
                0.00  0.00  0.70;  % Colors(7,:) - Camera Z-axis - Dark Blue
                0.00  0.80  0.44;  % Colors(8,:) - Horizon Vectors - Vibrant Green
                ];

            %% Begin Plotting
            plot =  figure('Name', 'Spacecraft Detected Snapshot', 'Color', 'w', 'Toolbar', 'figure');
            ax = gca;     hold(ax, 'on');     grid(ax, 'on');     axis(ax, 'equal');
            xlabel(ax, 'X (nd)', 'FontSize', 12);
            ylabel(ax, 'Y (nd)', 'FontSize', 12);
            zlabel(ax, 'Z (nd)', 'FontSize', 12);
            title(ax, ' Spacecraft Camera Frames and Horizon Detected ', 'FontSize', 14);

            % Plot the Moon using the plotter library
            planetPlot = plotterLib.createCelestialBodyEllipsoid(A, B, C, sunDirection, 'Position', planetPosition, 'Texture', 'moon.tif');
            set(planetPlot, 'DisplayName', 'Moon');

            hold(ax, 'on');

            % Scaling Factors
            scale = 0.02;           % For axes
            scale_horizon = 0.12;   % For horizon vectors

            %% Plot Moon's Principal Axes
            origin_planet = planetPosition;

            % Moon's X-Y-Z-axis
            quiver3(ax, origin_planet(1), origin_planet(2), origin_planet(3), scale, 0, 0, 'LineWidth', 2, 'Color', Colors(4,:), 'DisplayName', 'Moon X', 'MaxHeadSize', 0.8);
            quiver3(ax, origin_planet(1), origin_planet(2), origin_planet(3), 0, scale, 0, 'LineWidth', 2, 'Color', Colors(4,:), 'DisplayName', 'Moon Y', 'MaxHeadSize', 0.8);
            quiver3(ax, origin_planet(1), origin_planet(2), origin_planet(3), 0, 0, scale, 'LineWidth', 2, 'Color', Colors(4,:), 'DisplayName', 'Moon Z', 'MaxHeadSize', 0.8);
            text(ax, origin_planet(1) + scale, origin_planet(2), origin_planet(3), '  Moon X', 'FontSize', 12, 'Color', Colors(4,:));
            text(ax, origin_planet(1), origin_planet(2) + scale, origin_planet(3), '  Moon Y', 'FontSize', 12, 'Color', Colors(4,:));
            text(ax, origin_planet(1), origin_planet(2), origin_planet(3) + scale, '  Moon Z', 'FontSize', 12, 'Color', Colors(4,:));

            % Plot Spacecraft Trajectory Up to the Timestep
            sc_positions_upto_timestep = scTruePositions(1:timestep, :);
            plot3(ax, sc_positions_upto_timestep(:,1), sc_positions_upto_timestep(:,2), sc_positions_upto_timestep(:,3), ...
                'LineWidth', 1.5, 'Color', 'm', 'DisplayName', 'Trajectory');

            % Plot Spacecraft Position at Timestep
            scPosition = scTruePositions(timestep, :)';
            plot3(ax, scPosition(1), scPosition(2), scPosition(3), 'kp', 'MarkerSize', 10, 'MarkerFaceColor', 'k', 'DisplayName', 'Spacecraft');
            text(ax, scPosition(1), scPosition(2), scPosition(3), '  S/C', 'FontSize', 12, 'Color', 'k');

            % Compute Camera Axes at Timestep
            [T_CP, x_c, y_c, z_c] = obj.compute_T_CP(scPosition, planetPosition);

            % Plot Camera's Coordinate Axes
            % Camera X-Y-Z-axis
            quiver3(ax, scPosition(1), scPosition(2), scPosition(3), x_c(1)*scale, x_c(2)*scale, x_c(3)*scale, 'LineWidth', 2, 'Color', 'k', 'DisplayName', 'Camera X',  'MaxHeadSize', 0.8);
            quiver3(ax, scPosition(1), scPosition(2), scPosition(3), y_c(1)*scale, y_c(2)*scale, y_c(3)*scale, 'LineWidth', 2, 'Color', 'k', 'DisplayName', 'Camera Y', 'MaxHeadSize', 0.8);
            quiver3(ax, scPosition(1), scPosition(2), scPosition(3), z_c(1)*scale, z_c(2)*scale, z_c(3)*scale, 'LineWidth', 2, 'Color', 'k', 'DisplayName', 'Camera Z', 'MaxHeadSize', 0.8);
            text(ax, scPosition(1) + y_c(1)*scale, scPosition(2) + y_c(2)*scale, scPosition(3) + y_c(3)*scale, '  Cam Y', 'FontSize', 12, 'Color', 'k');
            text(ax, scPosition(1) + x_c(1)*scale, scPosition(2) + x_c(2)*scale, scPosition(3) + x_c(3)*scale, '  Cam X', 'FontSize', 12, 'Color', 'k');
            text(ax, scPosition(1) + z_c(1)*scale, scPosition(2) + z_c(2)*scale, scPosition(3) + z_c(3)*scale, '  Cam Z', 'FontSize', 12, 'Color', 'k');


            % Optionally Plot Horizon Vectors
            if plotHorizonVectors

                % Determine if limbCoordinatesAll is a cell array with one or multiple cells
                if iscell(limbCoordinatesAll) && numel(limbCoordinatesAll) == 1
                    % Only one set of limb coordinates provided
                    limbCoordinates = limbCoordinatesAll{1}; % Nx2 array
                elseif iscell(limbCoordinatesAll) && timestep <= numel(limbCoordinatesAll)
                    % Multiple sets of limb coordinates provided, access the desired timestep
                    limbCoordinates = limbCoordinatesAll{timestep}; % Nx2 array
                else
                    % Invalid input
                    error('limbCoordinatesAll must be a cell array with either one cell or at least %d cells.', timestep);
                end

                if isempty(limbCoordinates)
                    warning('Limb coordinates at timestep %d are empty. Skipping horizon vectors.', timestep);
                else
                    % Step 1: Convert pixel coordinates to normalized image plane coordinates (x, y)
                    s_i = obj.convert_pixelToNormalized_ImagePlane(limbCoordinates, camParameters); % Nx2 array

                    % Step 2: Convert to homogeneous coordinates (add third coordinate '1')
                    s_i_homogeneous = [s_i, ones(size(s_i,1),1)]';

                    % Step 3: Normalize the direction vectors to unit length
                    s_i_cam = s_i_homogeneous ./ vecnorm(s_i_homogeneous);

                    % Step 4: Transform s_i from camera frame to Planet frame using camera axes
                    s_i_planet = T_CP * s_i_cam;

                    % Reduce the number of horizon vectors being plotted
                    numVectors = size(s_i_planet, 2);
                    numSampled = min(numHorizonVectors, numVectors);
                    if numSampled < numVectors
                        sampleIndices = randperm(numVectors, numSampled);
                        reduced_s_i_planet = s_i_planet(:, sampleIndices);
                    else
                        reduced_s_i_planet = s_i_planet;
                    end

                    % Plot each horizon vector as a dashed line without arrow tip
                    for k = 1:size(reduced_s_i_planet, 2)
                        % Starting point is the spacecraft position
                        x_start = scPosition(1);
                        y_start = scPosition(2);
                        z_start = scPosition(3);

                        % Vector components
                        u = reduced_s_i_planet(1, k) * scale_horizon;
                        v = reduced_s_i_planet(2, k) * scale_horizon;
                        w = reduced_s_i_planet(3, k) * scale_horizon;

                        % Assign 'DisplayName' only to the first horizon vector
                        if k == 1
                            q =  quiver3(ax, x_start, y_start, z_start, u, v, w, ...
                                'Color', Colors(8,:), 'LineWidth', 0.8, 'ShowArrowHead', 'off', 'LineStyle', '--', 'DisplayName', 'Horizon Vectors');
                        else
                            q = quiver3(ax, x_start, y_start, z_start, u, v, w, ...
                                'Color', Colors(8,:), 'LineWidth', 0.8, 'ShowArrowHead', 'off', 'LineStyle', '--');
                            q.Annotation.LegendInformation.IconDisplayStyle = 'off';

                        end

                    end
                end
            end

            %% Finalize Visualization
            hold(ax, 'off');
            view(ax, 45, 45);
            legend(ax, 'show', 'Location', 'bestoutside');
            grid off;

        end


        %% Pinhole Camera Model Functions
        % Calculate the focal length of the camera in the X and Y directions

        function [f_x, f_y] = calculateFocalLength(obj, FOV_deg, resolutionPixels)
            % calculateFocalLength Computes the focal lengths f_x and f_y of a camera.
            %
            % This function calculates the focal distances in the x and y directions
            % (f_x and f_y) based on the camera's horizontal field of view (FOV) and the
            % sensor resolution in pixels. It handles non-square sensors by computing
            % f_x and f_y independently based on the aspect ratio of the sensor.
            %
            % Syntax:
            %   [f_x, f_y] = calculateFocalLength(FOV_deg, sensor_size_pixels)
            %
            % Inputs:
            %   FOV_deg            - HORIZONTAL field of view of the camera in degrees.
            %                        This is typically the FOV along the sensor's width.
            %   sensor_size_pixels - 1x2 vector specifying the sensor resolution in pixels.
            %                        [W, H], where W is the width in pixels and H is the height
            %                        in pixels.
            %
            % Outputs:
            %   f_x                - Focal length in the x-direction (pixels).
            %   f_y                - Focal length in the y-direction (pixels).
            %
            % Notes:
            %   - The function assumes that the horizontal FOV corresponds to the sensor's width.
            %   - The vertical FOV is computed based on the sensor's aspect ratio.
            %   - Ensure that the FOV is in degrees; if it's in radians, convert it before
            %     passing to this function.


            % Validate Inputs
            if ~isnumeric(FOV_deg) || ~isscalar(FOV_deg) || FOV_deg <= 0 || FOV_deg >= 180
                error('FOV_deg must be a positive scalar less than 180 degrees.');
            end

            if ~isnumeric(resolutionPixels) || ~isvector(resolutionPixels) || numel(resolutionPixels) ~= 2 || any(resolutionPixels <= 0)
                error('sensor_size_pixels must be a 1x2 positive vector [W, H].');
            end

            % Extract sensor width and height in pixels
            W = resolutionPixels(1);
            H = resolutionPixels(2);

            % Convert FOV from Degrees to Radians
            FOV_rad = deg2rad(FOV_deg);

            % This formula relates the sensor width and horizontal FOV to the focal length
            f_x = W / (2 * tan(FOV_rad / 2));

            % Compute Aspect Ratio of the Sensor
            aspect_ratio = H / W;

            FOV_y_rad = 2 * atan(tan(FOV_rad / 2) * aspect_ratio);

            % This formula relates the sensor height and vertical FOV to the focal length
            f_y = H / (2 * tan(FOV_y_rad / 2));

        end


        %% Limb Detection Function - Does not plot directly
        function [limbCoordinates, apparentRadius, center] = detectLimb(obj, imageData)
            % Detects the limb (edge) of a celestial body in an image.
            % Inputs:
            %   imageData - RGB image data
            % Outputs:
            %   limbCoordinates - Nx2 matrix of [x, y] coordinates of detected limb edge points
            %   apparentRadius  - Apparent radius of the limb in pixels
            %   using a circle fit to the points observed
            %   center          - [x, y] coordinates of the circle's center

            % Convert to grayscale
            grayImage = rgb2gray(imageData);

            % Enhance contrast
            enhancedImage = imadjust(grayImage);

            % - - - Apply edge detection with Canny Method - - -
            edges = edge(enhancedImage, 'Canny', [0.45, 0.99]);
            [edgeRows, edgeCols] = find(edges);

            limbCoordinates = [edgeCols, edgeRows]; % (x, y) coordinates

            % Initialize outputs
            apparentRadius = NaN;
            center = [NaN, NaN];

            % Fit a circle to the edge points to calculate the apparent radius
            if size(limbCoordinates, 1) >= 3 % Need at least 3 points for the circle fit
                % Fit circle to the edge points
                [center, radius] = obj.fitCircle(limbCoordinates(:,1), limbCoordinates(:,2));
                apparentRadius = radius;
            else
                warning('Not enough edge points to fit a circle.');
            end
        end


        % %% Limb Detection Function - Does not plot directly  v2
        % function [limbCoordinates, apparentRadius, center] = detectLimb(obj, imageData)
        %     % Convert to grayscale
        %     grayImage = rgb2gray(imageData);
        %
        %     % Enhance contrast using adaptive histogram equalization
        %     enhancedImage = adapthisteq(grayImage, 'ClipLimit', 0.01, 'NumTiles', [8 8]);
        %
        %     % Apply median filter for noise reduction
        %     filteredImage = medfilt2(enhancedImage, [3 3]);
        %
        %     % Normalize intensity
        %     normalizedImage = mat2gray(filteredImage);
        %
        %     % Use gradient-based sub-pixel edge detection
        %     [Gx, Gy] = imgradientxy(normalizedImage, 'sobel');
        %     [Gmag, Gdir] = imgradient(Gx, Gy);
        %
        %     % Normalize gradient magnitude
        %     GmagNormalized = Gmag / max(Gmag(:));
        %
        %     % Threshold gradient magnitude
        %     edgeThreshold = 0.55;
        %     edges = GmagNormalized > edgeThreshold;
        %
        %     % Sub-pixel edge localization
        %     [edgeRows, edgeCols] = find(edges);
        %     limbCoordinates = [edgeCols, edgeRows]; % Initial pixel-level coordinates
        %
        %     % Refine edge positions to sub-pixel accuracy
        %     % Implement sub-pixel refinement as discussed earlier
        %
        %     % Initialize outputs
        %     apparentRadius = NaN;
        %     center = [NaN, NaN];
        %     % Fit a circle to the edge points to calculate the apparent radius
        %     if size(limbCoordinates, 1) >= 3 % Need at least 3 points for the circle fit
        %         % Fit circle to the edge points
        %         [center, radius] = obj.fitCircle(limbCoordinates(:,1), limbCoordinates(:,2));
        %         apparentRadius = radius;
        %     else
        %         warning('Not enough edge points to fit a circle.');
        %     end
        % end
        %


        %% Circle Fit Using Least Squares
        function [center, radius] = fitCircle(obj, x, y)
            % Fits a circle to given x and y data points using least squares
            % Inputs:
            %   x, y - Vectors of x and y coordinates of points
            % Outputs:
            %   center - [x0, y0] coordinates of the circle's center
            %   radius - Radius of the circle

            % Prepare data
            A = [x(:), y(:), ones(size(x(:)))];
            b = -(x(:).^2 + y(:).^2);

            % Solve the linear system
            coeffs = A \ b;

            % Extract circle parameters
            xc = -0.5 * coeffs(1);
            yc = -0.5 * coeffs(2);
            r_squared = (coeffs(1)^2 + coeffs(2)^2)/4 - coeffs(3);

            center = [xc, yc];
            radius = sqrt(abs(r_squared));
        end


        %% Conic Fit Using Least Squares
        function [C_matrix, coeffs] = fitConic(obj, x, y)
            % FITCONIC Fits a conic section to a set of points.
            %   [C_matrix, coeffs] = fitConic(x, y) fits a general conic section
            %   to the points (x, y) and returns the symmetric matrix C_matrix
            %   and the coefficients [A, B, C, D, F, G] of the conic equation.
            %
            % Inputs:
            %   x - N x 1 vector of x coordinates
            %   y - N x 1 vector of y coordinates
            %
            % Outputs:
            %   C_matrix - 3 x 3 symmetric matrix of the conic
            %   coeffs   - 1 x 6 vector of conic coefficients [A, B, C, D, F, G]

            % Ensure x and y are column vectors
            x = x(:);
            y = y(:);
            N = length(x);

            % Build the design matrix
            Z = [x.^2, x.*y, y.^2, x, y, ones(N,1)];

            % Compute the SVD of Z
            [U, Sigma, V] = svd(Z, 0);

            % The solution is the last column of V (smallest singular value)
            C = V(:, end);

            % Normalize coefficients for stability #check
            coeffs = (C / norm(C))';

            % Extract coefficients
            A = coeffs(1);
            B = coeffs(2);
            C_ = coeffs(3);
            D = coeffs(4);
            F = coeffs(5);
            G = coeffs(6);

            % Build the symmetric matrix C_matrix
            C_matrix = [A,   B/2, D/2;
                B/2, C_,  F/2;
                D/2, F/2, G  ];
        end



        %% Animate Orbit
        function [limbCoordinatesAll, apparentRadiiAll, tauValuesAll, apparentAngularSizesAll, timeVector, adjustedResolution, camParameters] = ...
                animateCameraOrbit(obj, cr3bplib, plotterlib, R_p, bodyPosition, TimeVector, X_TrueStateHist, varargin)

            % - - - Parse optional parameters - - -
            p = inputParser;
            p.KeepUnmatched = true;
            addParameter(p, 'NumFrames', 50, @isnumeric);
            addParameter(p, 'InitialTau', 0.5, @isnumeric);
            addParameter(p, 'FinalTau', [], @isnumeric);
            addParameter(p, 'PropagationTime', [], @isnumeric);
            addParameter(p, 'CameraFOV', 25, @isnumeric);
            addParameter(p, 'Resolution', [800, 800], @(x) isnumeric(x) && numel(x) == 2);
            addParameter(p, 'SaveVideo', false, @islogical);
            addParameter(p, 'Filename', 'camera_animation', @ischar);
            addParameter(p, 'DetectLimb', false, @islogical);
            addParameter(p, 'ShowAnnotations', false, @islogical);
            addParameter(p, 'Texture', 'moon.tif', @ischar);
            addParameter(p, 'DynamicSunDirection', false, @islogical);

            parse(p, varargin{:});

            numFrames = p.Results.NumFrames;
            initialTau = p.Results.InitialTau;
            finalTau = p.Results.FinalTau;
            propagationTime = p.Results.PropagationTime;
            cameraFOV = p.Results.CameraFOV;
            resolution = p.Results.Resolution;
            saveVideo = p.Results.SaveVideo;
            filename = p.Results.Filename;
            detectLimb = p.Results.DetectLimb;
            showAnnotations = p.Results.ShowAnnotations;
            texture = p.Results.Texture;
            dynamicSunDirection = p.Results.DynamicSunDirection;
            % - - - Parse optional parameters - - -


            % CR3BP Characteristic Time
            t_star = 375704.306539867;           % [s]


            % - - - - Compute range of Tau Values - - - -
            % Determine finalTau if only PropagationTime is provided
            periodOrbit = max(TimeVector);
            if ~isempty(propagationTime)
                deltaTau = propagationTime / periodOrbit;
                finalTau = initialTau + deltaTau;
            elseif isempty(finalTau)
                error('Either FinalTau or PropagationTime must be provided.');
            end

            tauPoints = linspace(initialTau, finalTau, numFrames);
            timeVector = tauPoints - tauPoints(1);
            % - - - - Compute range of Tau Values - - - -


            % - - - - Generate Sun-Moon Vectors for Illumination - - - -
            if dynamicSunDirection
                deltaTau = finalTau - initialTau;           % Calculate tau range
                simTime = deltaTau * periodOrbit * t_star;  %[s] Get simulation time in seconds

                % Generate a time vector to compute the sun-moon vector position
                timeVectorDimensional = linspace(0, simTime, numFrames);

                % Compute history of Sun-Moon directions
                [r_sunMoon_B, ~, ~, ~] = cr3bplib.sunToMoonSynodic(timeVectorDimensional);
                r_sunMoon_B = r_sunMoon_B ./ vecnorm(r_sunMoon_B); % Normalize
            else
                r_sunMoon_B = [1; 0; 0];
            end
            % - - - - Generate Sun-Moon Vectors for Illumination - - - -


            % Initialize video writer
            if saveVideo
                v = VideoWriter(filename, 'MPEG-4');
                open(v);
            end

            % Initialize arrays to store outputs
            limbCoordinatesAll = cell(1, numFrames);
            apparentRadiiAll = zeros(1, numFrames);
            tauValuesAll = zeros(1, numFrames);
            apparentAngularSizesAll = zeros(1, numFrames);
            scTrueStateHistory = zeros(numFrames, 6);

            % - - - - Main Animation Loop - - - -
            for i = 1:numFrames
                tau = tauPoints(i);
                tauValuesAll(i) = tau;

                % Get spacecraft state at current tau
                scState = cr3bplib.getStateAtTau(TimeVector, X_TrueStateHist, tau, 'per');
                scTrueStateHistory(i,:) = scState';

                scPosition = scState(1:3);

                % Tau range text
                tauRangeText = sprintf('Tau Range: %.2f - %.2f', initialTau, finalTau);

                % Select the sun direction for this frame
                if dynamicSunDirection
                    currentSunDirection = r_sunMoon_B(:,i); % Dynamic case
                else
                    currentSunDirection = r_sunMoon_B;      % Static case
                end

                % Use simulateCameraImage to generate the figure with all content
                [~, limbCoordinates, apparentRadius, apparentAngularSize, fig, baseAnnotationText, adjustedResolution] = obj.simulateCameraImage(...
                    plotterlib, R_p, scPosition, bodyPosition, cameraFOV, resolution, ...
                    'DetectLimb', detectLimb, 'ShowAnnotations', showAnnotations, ...
                    'ShowFigure', true, 'Texture', texture, 'SunDirection', currentSunDirection);

                % Compute elapsed time in days
                elapsedTime_days = (tau - initialTau) * periodOrbit * t_star / 86400; % Elapsed time in days
                elapsedTimeText = sprintf('Elapsed Time: %.2f days', elapsedTime_days);

                % Combine base annotations with tau range and elapsed time
                updatedAnnotationText = sprintf('%s\n%s\n%s', baseAnnotationText, tauRangeText, elapsedTimeText);

                % Remove previous text annotations if necessary
                delete(findall(fig, 'type', 'text'));

                % Add combined annotation with tau range and elapsed time to the figure
                text(10, 10, updatedAnnotationText, ...
                    'Color', 'yellow', 'FontSize', 12, 'FontWeight', 'bold', ...
                    'HorizontalAlignment', 'left', 'VerticalAlignment', 'top');

                % Store outputs
                limbCoordinatesAll{i} = limbCoordinates;
                apparentRadiiAll(i) = apparentRadius;
                apparentAngularSizesAll(i) = apparentAngularSize;

                % Capture the entire figure for the video frame
                frame = getframe(fig);

                % Save video frame if required
                if saveVideo
                    writeVideo(v, frame);
                end

                % Close the figure to prevent memory issues
                close(fig);

                fprintf('Animating frame %d/%d\n', i, numFrames);

                % Control animation speed if desired
                pause(0.5);
            end
            % - - - - Main Animation Loop - - - -

            % - - - - Camera Parameters Computation - - - - -
            [f_x, f_y] = obj.calculateFocalLength(cameraFOV, adjustedResolution); % [px]
            camParameters.f_x = f_x;
            camParameters.f_y = f_y;
            camParameters.u_0 = adjustedResolution(1) / 2;
            camParameters.v_0 = adjustedResolution(2) / 2;
            camParameters.adjustedResolution = adjustedResolution;
            % - - - - Camera Parameters Computation - - - - -

            % Close Video Writer if used
            if saveVideo
                close(v);
                fprintf('Video saved as %s.mp4\n', filename);
            end

            % Compute total elapsed time in days
            totalElapsedTime_days = (finalTau - initialTau) * periodOrbit * t_star / 86400;

            % Save all information to a .mat file
            dataFilename = sprintf('AnimationData_Tau%.2f-%.2f_FOV%d_Res%dx%d_Frames%d_Time%.2fDays.mat', ...
                initialTau, finalTau, cameraFOV, adjustedResolution(1), adjustedResolution(2), numFrames, totalElapsedTime_days);

            % - - - - Save Parameters for MAT File - - - -
            saveData.limbCoordinatesAll = limbCoordinatesAll;
            saveData.apparentRadiiAll = apparentRadiiAll;
            saveData.tauValuesAll = tauValuesAll;
            saveData.apparentAngularSizesAll = apparentAngularSizesAll;
            saveData.timeVector = timeVector;
            saveData.initialTau = initialTau;
            saveData.finalTau = finalTau;
            saveData.cameraFOV = cameraFOV;
            saveData.userResolution = resolution;
            saveData.numFrames = numFrames;
            saveData.totalElapsedTime_days = totalElapsedTime_days;
            saveData.bodyPosition = bodyPosition;
            saveData.T_GW = TimeVector;
            saveData.X_GW = X_TrueStateHist;
            saveData.camParameters = camParameters;
            saveData.scStateHistory = scTrueStateHistory;
            saveData.scTruePositions = scTrueStateHistory(:,1:3);
            saveData.adjustedResolution = adjustedResolution;
            saveData.periodOrbit = periodOrbit;
            save(dataFilename, '-struct', 'saveData');  % Save the data to a .mat file
            fprintf('Data saved to %s\n', dataFilename);
            % - - - - Save Parameters for MAT File - - - -

        end


        %% Theoretical Aparent Angular Diameter
        function d_alpha = computeApparentAngularDiameter(obj, R_p, spacecraftPosition, bodyPosition)
            % Computes the apparent angular diameter of a celestial body
            % Inputs:
            %   R_p - Mean radius of the celestial body (scalar)
            %   spacecraftPosition - [x, y, z] position vector of the spacecraft
            %   bodyPosition - [x, y, z] position vector of the celestial body
            % Output:
            %   d_alpha - Apparent angular diameter in radians

            % Compute distance between spacecraft and celestial body
            rho = norm(spacecraftPosition - bodyPosition);

            % Ensure that R_p is less than rho to avoid invalid values
            if R_p >= rho
                error('The spacecraft is inside or on the surface of the celestial body.');
            end

            % Compute apparent angular diameter
            d_alpha = 2 * asin(R_p / rho);
        end

        %% Christian Robinson Algorithm
        function [r_c, R_s, P_r] = estimateCRA_SpacecraftPosition(obj, A_P, limbPointsCoords, camParameters, T_CP, varargin)
            % estimateSpacecraftPosition_CRA Estimates spacecraft position and computes covariance matrices.
            %
            % This function combines the estimation of the spacecraft's position using the
            % Christian Robinson Algorithm with the computation of covariance matrices
            % that quantify the uncertainty in the estimation.
            %
            % Inputs:
            %   obj               - Instance of the class containing this method.
            %   A_P               - Planet shape matrix in planet frame (3x3 matrix).
            %   limbPointsCoords  - Nx2 array of horizon points (pixel coordinates) [u, v].
            %   camParameters     - Struct with camera intrinsic parameters:
            %                        .f_x, .f_y - Focal lengths in pixels along x and y axes.
            %                        .u_0, .v_0 - Principal point coordinates (pixels).
            %   T_CP              - Rotation matrix from planet frame to camera frame (3x3 matrix).
            %   varargin          - Optional parameters (Name-Value pairs):
            %                        'sigma_pix' - Standard deviation of an observed horizon point in pixels.
            %
            % Outputs:
            %   r_c  - Estimated spacecraft position vector in the camera frame (3x1 vector).
            %   P_r  - (Optional) Covariance of the estimated position vector r_c (3x3 matrix).
            %   R_s  - (Optional) Covariance matrix of the horizon measurements (3x3 matrix).
            %
            % Usage:
            %   % Without covariance computation
            %   [r_c] = estimateSpacecraftPosition_CRA(obj, A_P, limbPointsCoords, camParameters, T_CP);
            %
            %   % With covariance computation
            %   [r_c, P_r, P_n, R_y, R_s] = estimateSpacecraftPosition_CRA(obj, A_P, limbPointsCoords, camParameters, T_CP, 'sigma_pix', 1.5);

            % Parse Optional Parameters
            p = inputParser;
            addParameter(p, 'sigma_pix', [], @(x) isempty(x) || (isscalar(x) && x > 0));
            parse(p, varargin{:});

            sigma_pix = p.Results.sigma_pix;            % Observed horizon point standard deviation in pixels
            compute_covariance = ~isempty(sigma_pix);

            %  - - - - Position Vector Estimation Start - - - -

            % Step 1: The shape matrix A in the camera frame is obtained by transforming A_P using T_CP
            A = T_CP * A_P * T_CP'; % #check

            % Step 2: Compute U via Cholesky factorization
            U = chol(A, 'upper');

            % Step 3: Transform horizon measurements to Cholesky factorized space
            % Convert pixel coordinates to normalized image plane coordinates (x, y).
            s_i = obj.convert_pixelToNormalized_ImagePlane(limbPointsCoords, camParameters);

            % Convert to homogeneous coordinates (image plane) by adding a third coordinate (1) for each point.
            s_i_homogeneous = [s_i, ones(size(s_i,1),1)]'; % 3xN matrix

            % Transform the measurements to Cholesky factorized space using matrix U.
            % This projects the image points into a space where the planet appears as a unit sphere.
            s_i_bar = U * s_i_homogeneous;

            % Step 4: Normalize s_tilde
            s_i_bar_norm = sqrt(sum(s_i_bar.^2,1));
            s_i_bar_prime = s_i_bar ./ s_i_bar_norm;

            % Form the measurement matrix H
            H = s_i_bar_prime';

            % Step 5: Solve the linear system H * n = 1
            % We are solving for vector n (3x1 vector) that best satifies the equation in a least-squares sense.
            n = H \ ones(size(H,1),1);

            % Step 6: Compute the spacecraft position vector r
            n_norm_sq = n' * n;
            k = -1 / sqrt(n_norm_sq - 1);
            y = U \ n;
            r_c = k * y; % SC r vector in the camera frame
            %  - - - - Position Vector Estimation End - - - -


            %  - - - - Covariances Estimation Start - - - -
            if compute_covariance
                % Validate sigma_pix
                if ~isscalar(sigma_pix) || sigma_pix <= 0
                    error('sigma_pix must be a positive scalar.');
                end

                % Step 2: Compute d_x (Focal Length over Pixel Pitch in Pixels per Radian)
                % #CHECK
                f_x = camParameters.f_x; % [px] Focal length in pixels
                mu_x = 1;                % [px] Distance between pixels
                d_x = f_x / mu_x;

                % Step 3: Compute R_s (Covariance of the Horizon Measurements)
                sigma_s = sigma_pix / d_x;         % Standard deviation in radians
                R_s = sigma_s^2 * diag([1, 1, 0]); % 3x3 matrix with zero in z-direction

                % Precompute C = U * R_s * U' : Eqn (52)
                C = U * R_s * U';

                % Step 4: Compute the Variance of Each Residual
                num_measurements = size(limbPointsCoords, 1);
                sigma_y_squared = zeros(num_measurements, 1); % Nx1 vector

                for i = 1:num_measurements
                    % Extract vectors for current measurement
                    s_bar_i = s_i_bar(:, i); % 3x1 vector
                    s_bar_prime_i = s_i_bar_prime(:, i); % 3x1 vector
                    norm_s_bar_i = norm(s_bar_i);

                    if norm_s_bar_i == 0
                        error('Normalization of s_i_bar_i resulted in zero.');
                    end

                    % Compute J_i : Eqn (50)
                    J_i = (1 / norm_s_bar_i) * (n') * (eye(3) - s_bar_prime_i * s_bar_prime_i'); % 1x3 vector

                    % Compute sigma^2_{y_i} : Eqn (52)
                    sigma_y_squared(i) = J_i * C * J_i'; % Scalar
                end

                % Step 5: Construct R_y (Covariance of the Residuals) : Eqn (47)
                R_y = diag(sigma_y_squared); % NxN diagonal matrix

                % Step 6: Compute P_n (Covariance of n)
                try
                    inv_R_y = diag(1 ./ sigma_y_squared); % NxN diagonal matrix
                    P_n = inv(H' * inv_R_y * H);          % 3x3 matrix
                catch ME
                    error('Computation of P_n failed: %s', ME.message);
                end

                % Step 7: Compute Matrix F
                scalar_factor = -1 / sqrt(n_norm_sq - 1); % Scalar
                temp_matrix = eye(3) - (n * n') / (n_norm_sq - 1); % 3x3 matrix
                F = scalar_factor * (U \ temp_matrix); % 3x3 matrix

                % Step 8: Compute P_r (Covariance of r_c)
                P_r = F * P_n * F'; % 3x3 matrix
            else
                % If covariance computation is not requested, assign empty matrices
                P_r = [];
                P_n = [];
                R_y = [];
                R_s = [];
            end
            %  - - - - Covariances Estimation End - - - -

        end



        function s_i = convert_pixelToNormalized_ImagePlane(obj, limbPointsCoords, camParameters)
            % Convert pixel coordinates of horizon points to normalized image plane coordinates.
            %
            % Inputs:
            %   limbPointsCoords     - Nx2 array of horizon points (pixel coordinates) [u, v]
            %   camParameters  - Struct with camera intrinsic parameters:
            %                       .f_x, .f_y - Focal lengths in pixels along x and y axes
            %                       .u_0, .v_0 - Principal point coordinates (pixels)
            %
            % Output:
            %   s_i               - Nx2 array of normalized image plane coordinates [x, y]
            %
            % Description:
            %   The function converts the pixel coordinates of the horizon points to normalized
            %   image plane coordinates using the pinhole camera model. This mapping transforms
            %   the pixel measurements into a coordinate system where the camera's optical
            %   center is at the origin, and distances are measured in units of focal length.
            %
            %   The conversion is performed as follows:
            %     x = (u - u_0) / f_x
            %     y = (v - v_0) / f_y
            %   where:
            %     (u, v)   - Pixel coordinates of the horizon point
            %     (u_0, v_0) - Pixel coordinates of the principal point (optical center)
            %     f_x, f_y - Focal lengths in pixels along the x and y axes
            %
            %   This conversion accounts for any offset of the optical center from the image center
            %   and scales the coordinates by the focal length, resulting in a dimensionless
            %   representation suitable for further geometric computations.

            % Extract pixel coordinates from the horizonPoints array
            u = limbPointsCoords(:,1); % Pixel coods on the image's x-axis
            v = limbPointsCoords(:,2); % Pixel coods on the image's y-axis

            % Apply the pinhole camera model equations to convert pixel coordinates to
            % normalized image plane coordinates
            x = (u - camParameters.u_0) / camParameters.f_x;
            y = (v - camParameters.v_0) / camParameters.f_y;

            % Combine the normalized coordinates into an Nx2 array
            s_i = [x, y];
        end

        % Rotation Matrix from Camera to Synodic Frame #CHECK
        function [T_CP, x_c, y_c, z_c] = compute_T_CP(obj, spacecraftPosition, bodyPosition)
            % COMPUTE_T_CP Computes the rotation matrix and camera axes for a spacecraft
            % pointing toward a celestial body.
            %
            % This function calculates the rotation matrix that transforms vectors
            % from the camera frame (attached to the spacecraft) to the CR3BP synodic frame
            % It also computes the camera's right, up, and forward axes.
            %
            % Parameters:
            %   spacecraftPosition - (3x1 vector) The spacecraft's position in the
            %                        synodic frame [X; Y; Z].
            %   bodyPosition       - (3x1 vector) The celestial body's position in the
            %                        synodic frame [X; Y; Z].
            %
            % Returns:
            %   T_CP - (3x3 matrix) Rotation matrix from the synodic frame to the camera frame.
            %          Columns of T_CP correspond to the camera axes:
            %          [x_c, y_c, z_c], where:
            %          x_c = Camera's right vector
            %          y_c = Camera's true up vector
            %          z_c = Camera's forward vector
            %
            % Notes:
            %   - The function ensures the computed axes are orthogonal and normalized.
            %   - In case the initial up vector U is parallel to z_c, a fallback
            %     up vector is used to avoid singularities.

            % Ensure column vectors
            C = spacecraftPosition(:); % Spacecraft position as a column vector
            T = bodyPosition(:);       % Celestial body position as a column vector

            % Calculate the forward vector (z_c)
            % This is the normalized line-of-sight vector pointing from the spacecraft
            % toward the target body.
            z_c = (T - C) / norm(T - C);

            % Define the initial up vector U (e.g., positive Z-axis in the synodic frame)
            U = [0; 0; 1];

            % Compute the right vector (x_c) as the cross product of z_c and U
            % x_c = cross(U, z_c);
            x_c = cross(z_c, U);      % #check

            % Check for singularity: if U is parallel to z_c, redefine U
            if norm(x_c) < 1e-6
                % When U is parallel to z_c; use a fallback up vector (e.g., positive X-axis)
                U = [1; 0; 0];
                x_c = cross(z_c, U);
            end

            % Normalize the right vector (x_c)
            x_c = x_c / norm(x_c);

            % Compute the true up vector (y_c) as the cross product of z_c and x_c
            y_c = cross(z_c, x_c);

            % Construct the rotation matrix T_CP from synodic to camera frame % #check
            % The columns of T_CP are the camera's right (x_c), up (y_c), and forward (z_c) axes.
            T_CP = [x_c, y_c, z_c];
        end





        % Plot Christian Robinson Estimated Trajectory vs Real
        function plotCRA_TrajectoryComparison(obj, plotterLib, A, B, C, sunDirection, planetPosition, scTruePositions, estimatedPositions, l_star)
            % PLOT_TRAJECTORYCOMPARISON_CRA Compares true and estimated spacecraft trajectories.
            %
            % This function plots:
            %   1. The Planet with its principal axes.
            %   2. The true spacecraft trajectory.
            %   3. The estimated spacecraft trajectory (valid positions only).
            %   4. The differences between the true and estimated trajectory components (X, Y, Z) in kilometers.
            %
            % Parameters:
            %   obj               - Instance of the class containing this method.
            %   plotterLib        - Object with method createCelestialBodyEllipsoid.
            %   A, B, C           - Scalars representing the Planet's semi-principal axes.
            %   sunDirection      - 3-element vector indicating the Sun's direction.
            %   planetPosition    - 3-element vector specifying the Planet's position [nd].
            %   scTruePositions   - [Nx3] matrix of true spacecraft positions [nd].
            %   estimatedPositions- [Nx3] matrix of estimated spacecraft positions [nd].
            %   l_star            - Scalar representing the scaling factor from nondimensional units to kilometers (km/nd).
            %
            % Example Usage:
            %   plot_TrajectoryComparison_CRA(obj, plotterLib, 1.0, 1.0, 1.0, [1; 0; 0], [0; 0; 0], scTruePositions, estimatedPositions, 1000);

            % Input Validation
            if nargin < 10
                error('All 10 input arguments must be provided, including l_star.');
            end

            if size(scTruePositions, 1) ~= size(estimatedPositions, 1)
                error('scTruePositions and estimatedPositions must have the same number of rows.');
            end

            if size(scTruePositions, 2) ~= 3 || size(estimatedPositions, 2) ~= 3
                error('scTruePositions and estimatedPositions must have three columns each.');
            end

            if ~isscalar(l_star) || l_star <= 0
                error('l_star must be a positive scalar representing the scaling factor from nd to km.');
            end

            % Moon and Principal Axes Setup
            scale = 0.05;                      % Scaling factor for Planet's axes
            origin_Planet = planetPosition;    % Planet's position

            % Filter Valid Estimated Positions
            validIndices = ~any(isnan(estimatedPositions), 2); % Rows without NaNs
            estimatedPositionsValid = estimatedPositions(validIndices, :);

            % Align true positions to valid indices
            truePositionsValid = scTruePositions(validIndices, :);

            % Plot 3D Trajectory Comparison
            figure('Name', 'Spacecraft Trajectory Comparison', 'Color', 'w');

            % Plot the Planet
            planetPlot = plotterLib.createCelestialBodyEllipsoid(A, B, C, sunDirection, ...
                'Position', origin_Planet, 'Texture', 'moon.tif');
            set(planetPlot, 'DisplayName', 'Planet');
            hold on;

            % Plot the Planet's principal axes
            quiver3(origin_Planet(1), origin_Planet(2), origin_Planet(3), scale, 0, 0, ...
                'r', 'LineWidth', 2, 'DisplayName', 'Planet X');
            quiver3(origin_Planet(1), origin_Planet(2), origin_Planet(3), 0, scale, 0, ...
                'm', 'LineWidth', 2, 'DisplayName', 'Planet Y');
            quiver3(origin_Planet(1), origin_Planet(2), origin_Planet(3), 0, 0, scale, ...
                'k', 'LineWidth', 2, 'DisplayName', 'Planet Z');

            % Plot the true trajectory
            plot3(scTruePositions(:,1), scTruePositions(:,2), scTruePositions(:,3), ...
                'r-', 'LineWidth', 1.5, 'DisplayName', 'True Trajectory');

            % Plot the estimated trajectory
            plot3(estimatedPositionsValid(:,1), estimatedPositionsValid(:,2), estimatedPositionsValid(:,3), ...
                'b.-', 'LineWidth', 1, 'DisplayName', 'Estimated Trajectory');

            grid on;
            axis equal;
            xlabel('X [nd]');
            ylabel('Y [nd]');
            zlabel('Z [nd]');
            title('True vs. Estimated Spacecraft Trajectory');
            legend('show');

            % Compute and Plot Differences in Trajectory Components
            figure('Name', 'Trajectory Component Differences', 'Color', 'w');

            % Compute position differences
            if size(truePositionsValid, 1) == size(estimatedPositionsValid, 1)
                positionDifferences_nd = truePositionsValid - estimatedPositionsValid; % [Nx3] in nd
                positionDifferences_km = positionDifferences_nd * l_star;             % [Nx3] in km
            else
                error('Mismatch in sizes of valid true positions and estimated positions.');
            end

            % Plot X component differences in km
            subplot(3, 1, 1);
            plot(1:size(positionDifferences_km, 1), positionDifferences_km(:, 1), 'r-', 'LineWidth', 1.5);
            grid on;
            xlabel('Time Step');
            ylabel('\Delta X [km]');
            title('Difference in X Component');

            % Plot Y component differences in km
            subplot(3, 1, 2);
            plot(1:size(positionDifferences_km, 1), positionDifferences_km(:, 2), 'm-', 'LineWidth', 1.5);
            grid on;
            xlabel('Time Step');
            ylabel('\Delta Y [km]');
            title('Difference in Y Component');

            % Plot Z component differences in km
            subplot(3, 1, 3);
            plot(1:size(positionDifferences_km, 1), positionDifferences_km(:, 3), 'k-', 'LineWidth', 1.5);
            grid on;
            xlabel('Time Step');
            ylabel('\Delta Z [km]');
            title('Difference in Z Component');

            % Finish Visualization
            sgtitle('Component-Wise Differences Between True and Estimated Positions (in km)');
        end

        % Compute history of Spacecraft Trajectories Based on CRA
        function [estimatedPositions, cameraAxes, estimatedCovariances, R_s_all] = computeCRA_Trajectory(obj, A_P, limbCoordinatesAll, camParameters, scTruePositions, planetPosition, varargin)
            % computeSCTrajectoryCRA Computes the spacecraft trajectory in the planet frame using CRA.
            %
            % This function calculates the spacecraft's estimated positions based on
            % the CRA algorithm and outputs the position vectors in the body frame.
            %
            % Parameters:
            %   A_P               - Horizon radius (principal semi-axis) of the Moon.
            %   limbCoordinatesAll- Cell array of horizon coordinates at each timestep.
            %   camParameters     - Struct containing the camera parameters:
            %                        .f_x, .f_y - Focal lengths (pixels)
            %                        .u_0, .v_0 - Principal point coordinates (pixels)
            %   scTruePositions   - [Nx3] Array of true spacecraft positions in CR3BP.
            %   planetPosition    - [1x3] Position of the Moon in the CR3BP frame.
            %   sigma_pix           - Standard deviation of an observed horizon point in pixels.
            %
            % Outputs:
            %   estimatedPositions   - [numTimesteps x 3] Estimated spacecraft positions in the synodic frame.
            %   cameraAxes           - Cell array of camera axes (x_c, y_c, z_c) at each timestep.
            %   estimatedCovariances - [numTimesteps x 3 x 3] Position covariance matrices in the synodic frame.
            %   R_s_all              - [numTimesteps x 3 x 3] Horizon measurement covariance matrices.


            % Parse Optional Parameters
            p = inputParser;
            addParameter(p, 'sigma_pix', [], @(x) isempty(x) || (isscalar(x) && x > 0));
            parse(p, varargin{:});
            sigma_pix = p.Results.sigma_pix;
            compute_covariance = ~isempty(sigma_pix);

            % Calculate the number of timesteps in the simulation
            numTimesteps = size(scTruePositions, 1);

            % Initialize outputs
            estimatedPositions = zeros(numTimesteps, 3);
            estimatedCovariances = zeros(numTimesteps, 3, 3);
            R_s_all = zeros(numTimesteps, 3, 3);
            cameraAxes = cell(numTimesteps, 1);

            % Loop through each timestep to compute the estimated positions
            for i = 1:numTimesteps
                % Extract the limb coordinates for the current timestep
                limbCoordinates = limbCoordinatesAll{i};

                % Handle empty limbCoordinates
                if isempty(limbCoordinates)
                    estimatedPositions(i, :) = [NaN, NaN, NaN];
                    estimatedCovariances(i, :, :) = NaN(3);
                    R_s_all(i, :, :) = NaN(3);
                    continue;
                end

                % Extract the spacecraft and body positions
                spacecraftPosition = scTruePositions(i, :)';   % [3 x 1]
                bodyPosition = planetPosition;  % [3 x 1]

                % Compute T_CP : rotates from synodic to camera! #?
                [T_CP, x_c, y_c, z_c] = obj.compute_T_CP(spacecraftPosition, bodyPosition);

                % Store the camera axes
                cameraAxes{i} = struct('x_c', x_c, 'y_c', y_c, 'z_c', z_c);

                % Estimate spacecraft position and compute covariance with CRA
                % T_CP' (transposed) rotates to the synodic frame ?
                if compute_covariance
                    [r_c, P_r, R_s] = obj.estimateCRA_SpacecraftPosition(A_P, limbCoordinates, camParameters, T_CP, 'sigma_pix', sigma_pix);
                else
                    [r_c, ~, ~] = obj.estimateCRA_SpacecraftPosition(A_P, limbCoordinates, camParameters, T_CP);
                end

                % Transform estimated position to synodic frame
                r_synodic = T_CP * r_c; % [3 x 1]
                % r_synodic = T_CP' * r_c; % #test
                estimatedPositions(i, :) = r_synodic' + bodyPosition'; % [1 x 3]

                if compute_covariance
                    % Transform covariance matrix to synodic frame
                    P_r_synodic = T_CP * P_r * T_CP';              % #check
                    estimatedCovariances(i, :, :) = P_r_synodic;

                    % Store R_s
                    R_s_all(i, :, :) = R_s;
                end

            end
        end


        %% Attitude Determination Based On Conic Locus
        function [T_C_P_solutions] = computeAttitudeConicLocus(obj, C_matrix, r_P, a, b, c)
            % computeAttitude Determines the rotation matrix T_C_P using the conic locus method.
            %
            % Inputs:
            %   obj      - Object handle (since it's a class method)
            %   C_matrix - 3x3 symmetric matrix defining the conic (from fitConic)
            %   r_P      - 3x1 vector of spacecraft position with respect to the planet, in the planet frame
            %   a, b, c - Semi-principal axes of the planet's ellipsoid
            %
            % Output:
            %   T_C_P_solutions - Cell array of 3x3 rotation matrices (up to two solutions)

            % Step 1: Compute A_P (Planet shape matrix)
            A_P = diag([1/a^2, 1/b^2, 1/c^2]); % Equation (24)

            % Step 2: Compute M_P
            M_P = A_P * (r_P * r_P') * A_P - (r_P' * A_P * r_P - 1) * A_P; % Equation (141)

            % Step 3: Compute SVD of C_matrix
            [V_L, S_C, V_R] = svd(C_matrix); % Equation (146)

            % Step 4: Compute SVD of M_P
            [W_L, S_M, W_R] = svd(M_P); % Equation (147)

            % Step 5: Loop over P_i matrices and compute possible T matrices
            P_list = {
                diag([1, 1, 1]),    % P1
                diag([-1, 1, 1]),   % P2
                diag([1, -1, 1]),   % P3
                diag([1, 1, -1])    % P4
                };

            T_C_P_solutions = {}; % Initialize empty cell array to store solutions
            j = 1;
            for i = 1:4
                P_i = P_list{i};
                T = V_L * P_i * W_L'; % Equation (150)
                T = det(T) * T;       % Ensure proper rotation matrix

                % Chirality test (Equation below Eq. (150))
                k = [0; 0; 1]; % k vector
                if k' * T * r_P > 0
                    % Keep this T
                    T_C_P_solutions{j} = T;
                    j = j + 1;
                end
            end

            % If no solutions passed the chirality test, return empty
            if isempty(T_C_P_solutions)
                warning('No rotation matrices passed the chirality test.');
            end
        end





    end
end

