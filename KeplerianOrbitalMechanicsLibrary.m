classdef KeplerianOrbitalMechanicsLibrary

    % KEPLERIANORBITALMECHANICSLIBRARY  Two-body/perturbed astrodynamics toolbox.
    %
    %   General-purpose routines for Keplerian mechanics, reference-frame
    %   conversions, anomaly/geometry utilities, ΔV helpers, and multiple
    %   propagators (Cartesian, Keplerian, Equinoctial, Milankovitch) with
    %   optional J2 and SRP. Includes CR3BP dynamics for quick three-body tests.
    %
    % FEATURES
    %   • Frames & DCMs
    %       - Orbital elements → DCM313, RTN/ECI vector transforms
    %       - Euler-sequence DCM builders (via Attitude library)
    %   • State converters
    %       - coe_to_cartesian, keplerian↔equinoctial↔Milankovitch
    %   • Geometry & kinematics
    %       - Orbit radii (r(ν), p, rp, ra), a/b from {rp,ra,ε,h}, circular speed
    %       - r/θ/h unit vectors; inclination/RAAN recovery from ĥ, r̂, θ̂
    %   • Anomalies & angles
    %       - Mean/Eccentric/True anomaly relations and solvers (Newton-Raphson & symbolic)
    %       - Flight-path angle fpa(e,ν) and fpa(h,r,v)
    %       - Phase angle, flyby turning angle for hyperbolas
    %   • Energetics & momentum
    %       - ε(v,r,μ), h and |h|, mean motion n, orbital period
    %   • ΔV helpers
    %       - ΔV components in VNB or RTN given azimuth/elevation
    %       - Departure ΔV from parking orbit given v∞
    %   • F–G functions
    %       - Position/velocity mapping for ellipses, hyperbolas, and general conics
    %   • Propagators (2-body)
    %       - Cartesian EOM: dynamics_2BP_cartesian (baseline), +J2, +J2+SRP
    %       - Element-space EOM: Keplerian / Equinoctial / Milankovitch with J2 [±SRP]
    %   • 3-body
    %       - CR3BP synodic 6-state dynamics
    %   • Perturbation models (ECI/RTN)
    %       - J2 acceleration, cannon-ball SRP
    %   • Helpers (Lambert-style triangle relations)
    %       - Semiperimeter/α/β and time-of-flight formulas for ellipse/hyperbola/parabola
    %
    % UNITS & FRAMES
    %   • Default dimensions: km, s, rad.  μ must be consistent with km^3/s^2.
    %   • Frames: ECI (inertial), RTN (orbital rotating), VNB.  DCM313 maps RTN→ECI.
    %   • Angles are radians unless a function name or docstring says otherwise.
    %
    % DEPENDENCIES
    %   AttitudeDeterminationLibrary   – Euler/DCM utilities
    %   CelestialBody                  – constants (Earth/Sun) used by some helpers
    %
    % QUICK START
    %   orb = KeplerianOrbitalMechanicsLibrary();
    %
    %   % 1) Elements → Cartesian state
    %   muE = 398600.0;
    %   coe = [7000, 0.01, deg2rad(28.5), deg2rad(40), deg2rad(30), deg2rad(10)];
    %   xECI = orb.coe_to_cartesian(coe, muE);           % [x y z vx vy vz]
    %
    %   % 2) Basic geometry/energetics
    %   vCirc = orb.v_circ_ur(muE, 7000);
    %   P     = orb.period_ua(muE, 7000);                % orbital period
    %
    %   % 3) Solve anomalies
    %   E = orb.E_Me2(0.5, 0.01);                        % solve Kepler's eqn
    %   nu = orb.ta_eE(0.01, E);
    %
    %   % 4) RTN/ECI transforms
    %   D = orb.OrbitalElementsToDCM(deg2rad(40), deg2rad(28.5), deg2rad(30+rad2deg(nu)));
    %   r_RTN = [norm(xECI(1:3)); 0; 0];                 % [r 0 0] in RTN
    %   r_ECI = orb.vec_rot_to_cart(D, r_RTN).';         % RTN→ECI
    %
    %   % 5) 2-body propagation (Cartesian)
    %   f  = @(t,X) orb.dynamics_2BP_cartesian(t, X, muE);
    %   tspan = [0, 2*P];
    %   [T,Y] = ode45(f, tspan, xECI, odeset('RelTol',1e-12,'AbsTol',1e-12));
    %
    %   % 6) Add J2 (and SRP if desired)
    %   fJ2  = @(t,X) orb.dynamics_2BP_cartesian_J2(t, X, muE, 1.0826e-3, 6378.1);
    %   [T2,Y2] = ode45(fJ2, tspan, xECI);
    %
    %   % 7) Propagate in element space (example: Keplerian + J2)
    %   Xkep0 = [coe(1:5), orb.M_eE(coe(2), E)];         % [a e i Ω ω M]
    %   fkJ2  = @(t,X) orb.dynamics_2BP_keplerian_J2(t, X);
    %   [Tk,Xk] = ode113(fkJ2, [0 P], Xkep0);
    %
    %   % 8) CR3BP quick test
    %   muCR = 0.01215058; n = 1;
    %   x0 = [1.02 0 0 0 0.15 0]';
    %   [Tc,Yc] = ode45(@(t,X) orb.CR3BP(X, muCR, n), [0 3], x0);
    %    
    % NOTES
    %   - Some routines assume e<1 (ellipse) or e>1 (hyperbola); check function
    %     comments and branch your logic accordingly.
    %   - Angles: several helpers accept/return degrees internally when noted
    %     by the variable name or the original function; keep units consistent.
    %   - Element-space propagators rely on small-perturbation Gauss equations;
    %     accuracy degrades for very high J2/SRP or near singular elements.
    %
    % AUTHOR
    %   Moacir Fonseca Becker
    %   Purdue University
    %
    % LAST MODIFIED
    %   08/13/2025

    properties (Access = private)
        orb        KeplerianOrbitalMechanicsLibrary   % self-handle or user-supplied
        adc        AttitudeDeterminationLibrary       % attitude library (optional)
        bodyEarth  CelestialBody                      % constant data – Earth
        bodySun    CelestialBody                      % constant data – Sun
    end

    methods
        function obj = KeplerianOrbitalMechanicsLibrary(orbInstance, adcInstance)
            %KEPLERIANORBITALMECHANICSLIBRARY  Constructor.
            %
            % If the caller passes another K.O.M.L. object, keep it.
            % Otherwise   →   use *this* object as the internal reference
            % (avoids infinite recursion).

            % --- handle ORB instance ------------------------------------
            if nargin >= 1 && isa(orbInstance, 'KeplerianOrbitalMechanicsLibrary')
                obj.orb = orbInstance;           % caller-supplied
            else
                obj.orb = obj;                   % point to self  (no recursion!)
            end

            % --- handle ADC instance ------------------------------------
            if nargin == 2 && isa(adcInstance, 'AttitudeDeterminationLibrary')
                obj.adc = adcInstance;
            else
                obj.adc = AttitudeDeterminationLibrary();   % one-off construction, no loop
            end

            % --- cache Earth & Sun constants ----------------------------
            obj.bodyEarth = CelestialBody("Earth");
            obj.bodySun   = CelestialBody("Sun");
        end
    end

    methods

%% --- 3D Orbital Functions ---
    % --- Coordinate Transformation Functions ---
    
        % --- Converts a vector from ECI frame to RTN frame using the inverse
        % DCM ---
        function [r_vec_rtn] = vec_cart_to_rot(DCMinv, r_vec_eci)
            % Inputs:
            %   DCMinv: Inverse Direction Cosine Matrix (DCM) for the
            %   transformation r_vec_eci: Vector in ECI frame to be converted
            % Outputs:
            %   r_vec_rtn: Vector in RTN frame after conversion
            r = dot(DCMinv(1,:), r_vec_eci);
            theta = dot(DCMinv(2,:), r_vec_eci);
            h = dot(DCMinv(3,:), r_vec_eci);      
            r_vec_rtn = [r, theta, h];
        end
        
        % --- Transforms a vector from the RTN frame to the ECI frame ---
        function [r_xyz] = vec_rot_to_cart(obj, DCM313, vec_rot)
            % Inputs:
            %   DCM313: Direction Cosine Matrix for 3-1-3 rotation sequence
            %   in column
            %   vec_rot: Vector in RTN frame to be converted
            % Outputs:
            %   r_xyz: Vector in ECI frame after conversion
            rx = dot(DCM313(1,:), vec_rot);
            ry = dot(DCM313(2,:), vec_rot);
            rz = dot(DCM313(3,:), vec_rot);      
            r_xyz = [rx, ry, rz];
        end
        
        % --- Generates the Direction Cosine Matrix (DCM) from orbital elements
        % ---
        function [DCM313] = OrbitalElementsToDCM(obj,RAAN, i, theta)
            % Inputs:
            %   RAAN: Right Ascension of the Ascending Node (radians) i:
            %   Inclination (degrees) theta: Argument of periapsis + True
            %   Anomaly (radians)
            % Outputs:
            %   DCM313: Direction Cosine Matrix for the specified orbital
            %   elements
            X_hat  = [(cos(RAAN)*cos(theta))-(sin(RAAN)*cos(i)*sin(theta)), ...
                    -cos(RAAN)*sin(theta)-(sin(RAAN)*cos(i)*cos(theta)), ...
                    sin(RAAN)*sin(i)];
                
            Y_hat = [(sin(RAAN)*cos(theta))+(cos(RAAN)*cos(i)*sin(theta)), ...
                    -sin(RAAN)*sin(theta)+(cos(RAAN)*cos(i)*cos(theta)), ...
                    -cos(RAAN)*sin(i)];
                
            Z_hat = [(sin(i)*sin(theta)), (sin(i)*cos(theta)), cos(i)];
            
            DCM313 = [X_hat; Y_hat; Z_hat];
        end
    
    % --- Orbital Inclination and Angle Functions ---
    
        % --- Finds the orbital inclination given the h vector ---
        function [i] = i_h_xyz(h_vec_norm)
            % Inputs:
            %   h_vec_norm: Normalized angular momentum vector (h)
            % Outputs:
            %   i: Inclination of the orbit (radians)
            i = acos(h_vec_norm(3));
        end
        
        % --- Finds possible theta values given unit vectors of r and theta ---
        function [theta] = theta_3D_rth_hat(r_eci_unit, theta_eci_unit, i)
            % Inputs:
            %   r_eci_unit: Unit vector of position in ECI frame
            %   theta_eci_unit: Unit vector of velocity direction in ECI frame
            %   i: Orbital inclination (radians)
            % Outputs:
            %   theta: Possible values of theta (true anomaly)
            
            s_thetas = r_eci_unit(3) / sin(i); 
            c_thetas = theta_eci_unit(3) / sin(i);
            
            % Find possible theta solutions for cosine
            cos_thetas = [round(acos(c_thetas)), -round(acos(c_thetas))];
            
            % Find possible theta solutions for sine
            if asin(s_thetas) ~= 90 && asin(s_thetas) ~= 0 && asin(s_thetas) ~= 180 
                sin_thetas = [round(asin(s_thetas)), 180 - round(asin(s_thetas))];
            elseif asin(s_thetas) == 90
                sin_thetas = [round(asin(s_thetas)), -270];
            else
                sin_thetas = [0, 180];
            end
            
            % Find the common theta angle
            theta = intersect(cos_thetas, sin_thetas);
        end
        
        % --- Finds the Right Ascension of the Ascending Node (RAAN) given h
        % vector ---
        function [raan] = RAAN_find_use_h_hat(h_vec_norm_eci, i)
            % Inputs:
            %   h_vec_norm_eci: Normalized angular momentum vector (h) in ECI
            %   frame i: Orbital inclination (radians)
            % Outputs:
            %   raan: Possible values of RAAN (radians)
            
            s_raan = h_vec_norm_eci(1) / sin(i); 
            c_raan = -h_vec_norm_eci(2) / sin(i);
            
            % Find possible RAAN solutions for cosine
            cos_raans = [round(acos(c_raan)), -round(acos(c_raan))];
            
            % Find possible RAAN solutions for sine
            if asin(s_raan) ~= 90 && asin(s_raan) ~= 0 && asin(s_raan) ~= 180 
                sin_raans = [round(asin(s_raan)), 180 - round(asin(s_raan))];
            elseif asin(s_raan) == 90
                sin_raans = [round(asin(s_raan)), -270];
            else
                sin_raans = [0, 180];
            end
            
            % Find the common RAAN angle
            raan = intersect(cos_raans, sin_raans);
        end
        
        % --- Generates unit vectors for r, theta, and h ---
        function [theta_hat, r_hat, h_hat, XYZ_hat] = rotational_hat(RAAN, i, theta)
            % Inputs:
            %   RAAN: Right Ascension of the Ascending Node (radians) i:
            %   Inclination (radians) theta: Argument of periapsis + True
            %   Anomaly (radians)
            % Outputs:
            %   theta_hat: Unit vector in theta direction (RTN frame) r_hat:
            %   Unit vector in r direction (RTN frame) h_hat: Unit vector in h
            %   direction (angular momentum direction) XYZ_hat: Combined matrix
            %   of unit vectors
            
            r_hat = [(cos(RAAN)*cos(theta)) - (sin(RAAN)*cos(i)*sin(theta)), ...
                    (sin(RAAN)*cos(theta)) + (cos(RAAN)*cos(i)*sin(theta)), ...
                    sin(i)*sin(theta)];
                
            theta_hat = [-cos(RAAN)*sin(theta) - (sin(RAAN)*cos(i)*cos(theta)), ...
                        -sin(RAAN)*sin(theta) + (cos(RAAN)*cos(i)*cos(theta)), ...
                        sin(i)*cos(theta)];
                    
            h_hat = [sin(RAAN)*sin(i), -cos(RAAN)*sin(i), cos(i)];
            
            XYZ_hat = [r_hat', theta_hat', h_hat'];
        end

       
%% --- Position And Velocity Functions ---    
    % --- Position Functions ---
    
        % --- Converts position vector from the rotational frame to the VNB
        % frame ---
        function [r_vnb] = r_vec_rot_to_vnb(fpa, r_mag)
            % Inputs:
            %   fpa: Flight Path Angle (degrees) r_mag: Magnitude of the radius
            % Outputs:
            %   r_vnb: Position vector in the Velocity-Normal-Binormal (VNB)
            %   frame [r_v, r_n, r_b]
            r_vnb = [r_mag * sin(fpa), 0, r_mag * cos(fpa)];
        end
        
        % --- Position vector from Eccentric Anomaly in elliptical orbits ---
        function [r] = r_vec_ep_aEeb(a, E, e, b)
            % Inputs:
            %   a: Semi-major axis E: Eccentric Anomaly (degrees) e:
            %   Eccentricity b: Semi-minor axis
            % Outputs:
            %   r: Position vector [r_e, r_p, 0] in the e-p frame
            r = [a * (cos(E) - e), b * sin(E), 0];
        end
        
        % --- Generates a position vector in the rotating frame (r-theta-h) ---
        function [r] = r_vec_rot_frame(r_mag)
            % Inputs:
            %   r_mag: Magnitude of the radius
            % Outputs:
            %   r: Position vector [r, 0, 0] in the rotating frame
            r = [r_mag, 0, 0];
        end
        
        % --- Position vector in the e-p (Perifocal) reference frame ---
        function [r] = r_vec_ep_tar(ta, r_mag)
            % Inputs:
            %   ta: True Anomaly (degrees) r_mag: Radius magnitude
            % Outputs:
            %   r: Position vector [r_e, r_p, 0] in the e-p frame
            r_e = r_mag * cos(ta); 
            r_p = r_mag * sin(ta);
            r = [r_e, r_p, 0];
        end
        
        % --- Distance to body for elliptical and hyperbolic orbits ---
        function [r] = r_aeta(obj,a, e, ta)
            % Inputs:
            %   a: Semi-major axis e: Eccentricity ta: True Anomaly (degrees)
            % Outputs:
            %   r: Orbital radius
            if e > 1
                r = (abs(a) * (e^2 - 1)) / (1 + (e * cos(ta)));
            elseif e < 1
                r = (a * (1 - e^2)) / (1 + (e * cos(ta)));
            end
        end
        
        % --- Distance to body from semi-latus rectum, eccentricity, and true
        % anomaly ---
        function [r] = r_Peta(p, e, ta)
            % Inputs:
            %   p: Semi-latus rectum e: Eccentricity ta: True Anomaly (degrees)
            % Outputs:
            %   r: Orbital radius
            r = p / (1 + (e * cos(ta)));
        end
        
        % --- Distance to body for orbital mechanics calculations ---
        function [r] = r_hueta(h, u, e, ta)
            % Inputs:
            %   h: Specific angular momentum u: Gravitational parameter e:
            %   Eccentricity ta: True Anomaly (degrees)
            % Outputs:
            %   r: Orbital radius
            r = (h^2 / u) / (1 + e * cos(ta));
        end
        
    % --- Velocity Functions ---
    
        % --- Velocity at infinity for hyperbolic orbits ---
        function [v] = vel_inf_ua(u, a)
            % Inputs:
            %   u: Gravitational parameter a: Semi-major axis
            % Outputs:
            %   v: Velocity at infinity
            v = sqrt(u / abs(a));
        end
        
        % --- Velocity vector in the rotating frame for orbital motion ---
        function [v] = v_vec_rot_frame_fpav(obj,fpa, vmag)
            % Inputs:
            %   fpa: Flight Path Angle (degrees) vmag: Velocity magnitude
            %   (km/s)
            % Outputs:
            %   v: Velocity vector [v_r, v_theta, v_h] in the rotating frame
            v = [vmag * sin(fpa), vmag * cos(fpa), 0];
        end
        
        % --- Velocity vector components in the e-p frame from orbital
        % parameters ---
        function [v] = v_vec_ep_rEban(r, E, b, a, n)
            % Inputs:
            %   r: Position vector magnitude E: Eccentric Anomaly (degrees) b:
            %   Semi-minor axis a: Semi-major axis n: Mean motion
            % Outputs:
            %   v: Velocity vector in the e-p frame
            r_mag = norm(r);
            v = [((-a^2 * n) / r_mag) * sin(E), cos(E) * ((a * b * n) / r_mag), 0];
        end
        
        % --- Velocity vector in e-p frame from flight path angle and true
        % anomaly ---
        function [v] = v_vec_ep_frame(fpa, ta, vmag)
            % Inputs:
            %   fpa: Flight Path Angle (degrees) ta: True Anomaly (degrees)
            %   vmag: Velocity magnitude (km/s)
            % Outputs:
            %   v: Velocity vector [v_e, v_p, 0] in the e-p frame
            v_e = vmag * sin(fpa) * cos(ta) + vmag * cos(fpa) * -sin(ta);
            v_p = vmag * sin(fpa) * sin(ta) + vmag * cos(fpa) * cos(ta);
            v = [v_e, v_p, 0];
        end
        
        % --- Velocity magnitude from gravitational parameter, orbital radius,
        % and semi-major axis ---
        function [v] = vel_ura(obj, u, r, a, E_or_H)
            % Inputs:
            %   u: Gravitational parameter r: Orbital radius a: Semi-major axis
            %   E_or_H: Orbit type ('E' for Ellipse, 'H' for Hyperbola)
            % Outputs:
            %   v: Velocity magnitude (km/s)
            if E_or_H == "E" || E_or_H == "e"
                v = sqrt(2 * (u / r) - (u / a));
            elseif E_or_H == "H" || E_or_H == "h"
                v = sqrt(2 * (u / r) + (u / abs(a)));
            end
        end
        
        % --- Circular orbit velocity from gravitational parameter and radius
        % ---
        function [v] = v_circ_ur(obj, u, r)
            % Inputs:
            %   u: Gravitational parameter r: Orbital radius
            % Outputs:
            %   v: Circular orbit velocity
            v = sqrt(u / r);
        end
        
        % --- Velocity magnitude for elliptical and hyperbolic orbits ---
        function [v] = v_mag_ura(u, r, a, E_or_H)
            % Inputs:
            %   a: semimajor axis
            %   u: Gravitational parameter r: Orbital radius a: Semi-major axis
            %   E_or_H: Orbit type ('E' for Ellipse, 'H' for Hyperbola)
            % Outputs:
            %   v: Velocity magnitude (km/s)
            if E_or_H == "E" || E_or_H == "e"
                v = sqrt(2 * (u / r) - (u / a));
            elseif E_or_H == "H" || E_or_H == "h"
                v = sqrt(2 * (u / r) + (u / abs(a)));
            end
        end

                        
%% --- Orbital Geometrical Variables ---
    % --- Semi-Major and Minor Axis, Periapsis/Apoapsis Radius  ---
    
        % --- Semi-Major Axis from Periapsis Radius and Eccentricity ---
        function [a] = a_rpe(obj, rp, e)
            % Inputs:
            %   rp: Periapsis radius e: Eccentricity
            % Outputs:
            %   a: Semi-major axis
            % Distinguishes between hyperbolic orbits (e > 1) and elliptical
            % orbits (e < 1).
            if e > 1
                a = rp / (e - 1);  % Hyperbolic orbit
            elseif e < 1
                a = rp / (1 - e);  % Elliptical orbit
            end
        end
        
        % --- Semi-Major Axis from Apoapsis Radius and Eccentricity ---
        function [a] = a_rae(obj, ra, e)
            % Inputs:
            %   ra: Apoapsis radius e: Eccentricity
            % Outputs:
            %   a: Semi-major axis
            % Applicable for all conic sections.
            a = ra / (1 + e);
        end
        
        % --- Semi-Major Axis from Orbital Radius, Gravitational Parameter, and
        % Velocity ---
        function [a] = a_ruv(obj, r, u, v)
            % Inputs:
            %   r: Orbital radius u: Gravitational parameter v: Velocity
            %   magnitude
            % Outputs:
            %   a: Semi-major axis
            % Note: Assumes elliptical orbits.
            a = abs((u * r) / ((r * v^2) - (2 * u)));
        end
        
        % --- Semi-Major Axis from Periapsis and Apoapsis Radii ---
        function [a] = a_rpra(obj, rp, ra)
            % Inputs:
            %   rp: Periapsis radius ra: Apoapsis radius
            % Outputs:
            %   a: Semi-major axis
            % Suitable for all conic sections.
            a = (rp + ra) / 2;
        end
        
        % --- Semi-Major Axis from Specific Orbital Energy and Gravitational
        % Parameter ---
        function [a] = a_espu(obj, esp, u)
            % Inputs:
            %   esp: Specific orbital energy u: Gravitational parameter
            % Outputs:
            %   a: Semi-major axis
            % Applicable for elliptical orbits.
            a = u / (2 * abs(esp));
        end
        
        % --- Semi-Minor Axis from Semi-Major Axis and Eccentricity ---
        function [b] = b_ae(obj, a, e)
            % Inputs:
            %   a: Semi-major axis e: Eccentricity
            % Outputs:
            %   b: Semi-minor axis
            % Differentiates between hyperbolic (e > 1) and elliptical (e < 1)
            % orbits.
            if e > 1
                b = a * sqrt(e^2 - 1);  % Hyperbolic orbit
            elseif e < 1
                b = a * sqrt(1 - e^2);  % Elliptical orbit
            end
        end
        
        % --- Apoapsis Radius from Semi-Major Axis and Eccentricity ---
        function [ra] = ra_ae(obj, a, e)
            % Inputs:
            %   a: Semi-major axis e: Eccentricity
            % Outputs:
            %   ra: Apoapsis radius
            ra = a * (1 + e);
        end
        
        % --- Periapsis Radius from Semi-Major Axis and Eccentricity ---
        function [rp] = rp_ae(obj,a, e)
            % Inputs:
            %   a: Semi-major axis e: Eccentricity
            % Outputs:
            %   rp: Periapsis radius
            % Handles hyperbolic (e > 1) and elliptical (e < 1) orbits
            % distinctly.
            if e > 1
                rp = a * (e - 1);  % Hyperbolic orbit
            elseif e < 1
                rp = a * (1 - e);  % Elliptical orbit
            end
        end
    
    % --- Semi-Latus Rectum Functions ---
    
        % --- Semi-Latus Rectum for Hyperbolic Orbits from Semi-Major Axis and
        % Eccentricity ---
        function [p] = p_hybl_ae(obj, a, e)
            % Inputs:
            %   a: Semi-major axis e: Eccentricity
            % Outputs:
            %   p: Semi-latus rectum
            % Applicable for hyperbolic orbits.
            p = abs(a) * (1 - e^2);
        end
        
        % --- Semi-Latus Rectum from Angular Momentum and Gravitational
        % Parameter ---
        function [p] = p_hu(obj, h, u)
            % Inputs:
            %   h: Angular momentum u: Gravitational parameter
            % Outputs:
            %   p: Semi-latus rectum
            % Applicable to all conic sections.
            p = h^2 / u;
        end
        
        % --- Semi-Latus Rectum from Semi-Major Axis and Eccentricity ---
        function [p] = p_ae(obj, a, e)
            % Inputs:
            %   a: Semi-major axis e: Eccentricity
            % Outputs:
            %   p: Semi-latus rectum
            % Differentiates between hyperbolic (e > 1) and other conic
            % sections.
            if e > 1
                p = a * (e^2 - 1);  % Hyperbolic orbit
            else
                p = a * (1 - e^2);  % Elliptical orbit
            end
        end

%% --- 3D Maneuvers ---
    
    % --- Delta V in VNB (Velocity-Normal-Binormal) Frame ---
    function [DeltaV_VNB] = delta_v_vnb_alpha_beta(obj, delta_v_mag, beta, alpha)
        % Inputs:
        %   delta_v_mag: Magnitude of the velocity change (Delta V) beta:
        %   Elevation angle (radians) alpha: Azimuth angle (radians)
        % Outputs:
        %   DeltaV_VNB: Velocity change vector in the VNB frame [v_v, v_n,
        %   v_b]
        % The function calculates the Delta V components in the VNB frame
        % using beta and alpha angles.
        DeltaV_VNB = delta_v_mag .* [cos(beta) * cos(alpha), sin(beta), cos(beta) * sin(alpha)];
    end
    
    % --- Delta V in Rotational Frame ---
    function [DeltaV_rot] = delta_v_rot_beta_phi(delta_v_mag, beta, phi)
        % Inputs:
        %   delta_v_mag: Magnitude of the velocity change (Delta V) beta:
        %   Elevation angle (radians) phi: Azimuth angle (radians)
        % Outputs:
        %   DeltaV_rot: Velocity change vector in the rotational frame
        %   [v_r, v_theta, v_h]
        % The function calculates the Delta V components in the rotational
        % (r-theta-h) frame using beta and phi angles.
        DeltaV_rot = delta_v_mag .* [cos(beta) * sin(phi), cos(beta) * cos(phi), sin(beta)];
    end


%% --- F and G Functions for Position and Velocity in Conic Sections ---

    % --- Hyperbolas ---
    
    % --- F + G Function for Position in Hyperbolic Orbits ---
    function [rx] = rx_fg_hyperbola(a, ro, Hdelta, vo, t_delta, u)
        % Inputs:
        %   a: Semi-major axis (negative for hyperbolas) ro: Initial
        %   position vector Hdelta: Hyperbolic anomaly difference vo:
        %   Initial velocity vector t_delta: Time difference u:
        %   Gravitational parameter
        % Outputs:
        %   rx: Final position vector in the e-p frame
        % Computes the final position vector using F and G functions for
        % hyperbolic orbits.
        f = (1 - (abs(a) / norm(ro) * (cosh(Hdelta) - 1))) * ro;
        g = (t_delta - (sqrt(abs(a)^3 / u) * (sinh(Hdelta) - Hdelta))) * vo;
        rx = f + g;
    end
    
    % --- Conics (General) ---
    
    % --- F + G Function for Position in Conic Orbits ---
    function [rx] = rx_fg_conic(p, rx, ro, ta_delta, vo, u)
        % Inputs:
        %   p: Semi-latus rectum rx: Final position vector magnitude ro:
        %   Initial position vector ta_delta: True anomaly difference vo:
        %   Initial velocity vector u: Gravitational parameter
        % Outputs:
        %   rx: Final position vector in the e-p frame
        % Computes the final position vector using F and G functions for
        % general conic orbits.
        f = (1 - (rx / p) * (1 - cos(ta_delta))) * ro;
        g = ((rx * norm(ro)) / sqrt(u * p)) * sin(ta_delta) * vo;
        rx = f + g;
    end
    
    % --- F Function for Conic Orbits ---
    function [f] = f_conic(p, rx, ta_delta)
        % Inputs:
        %   p: Semi-latus rectum rx: Final position vector magnitude
        %   ta_delta: True anomaly difference
        % Outputs:
        %   f: F function value
        % Computes the F function for conic orbits.
        f = 1 - (rx / p) * (1 - cos(ta_delta));
    end
    
    % --- G Function for Conic Orbits ---
    function [g] = g_conic(rx, ro_vec, u, p, ta_delta)
        % Inputs:
        %   rx: Final position vector magnitude ro_vec: Initial position
        %   vector u: Gravitational parameter p: Semi-latus rectum
        %   ta_delta: True anomaly difference
        % Outputs:
        %   g: G function value
        % Computes the G function for conic orbits.
        g = (rx * norm(ro_vec) / sqrt(u * p)) * sin(ta_delta);
    end
    
    % --- Ellipses ---
    
    % --- F + G Function for Position in Elliptical Orbits ---
    function [rx] = rx_fg_ellipse(a, ro_vec, Edelta, vo, t_delta, u)
        % Inputs:
        %   a: Semi-major axis ro_vec: Initial position vector Edelta:
        %   Eccentric anomaly difference vo: Initial velocity vector
        %   t_delta: Time difference u: Gravitational parameter
        % Outputs:
        %   rx: Final position vector in the e-p frame
        % Computes the final position vector using F and G functions for
        % elliptical orbits.
        f = (1 - (a / norm(ro_vec) * (1 - cos(Edelta)))) * ro_vec;
        g = (t_delta - (sqrt(a^3 / u) * (deg2rad(Edelta) - sin(deg2rad(Edelta))))) * vo;
        rx = f + g;
    end
    
    % --- F Function for Elliptical Orbits ---
    function [f] = f_rx_ellipse(a, ro_vec, Edelta)
        % Inputs:
        %   a: Semi-major axis ro_vec: Initial position vector Edelta:
        %   Eccentric anomaly difference
        % Outputs:
        %   f: F function value
        % Computes the F function for elliptical orbits.
        f = (1 - (a / norm(ro_vec) * (1 - cos(Edelta)))) * ro_vec;
    end
    
    % --- G Function for Elliptical Orbits ---
    function [g] = g_rx_ellipse(vo_vec, t_delta, E_delta, a, u)
    % Inputs:
    %   vo_vec: Initial velocity vector t_delta: Time difference E_delta:
    %   Eccentric anomaly difference a: Semi-major axis u: Gravitational
    %   parameter
    % Outputs:
    %   g: G function value
    % Computes the G function for elliptical orbits.
    g = (t_delta - (sqrt(a^3 / u) * (deg2rad(E_delta) - sin(deg2rad(E_delta))))) * vo_vec;
    end


%% --- F and G Functions that Relate Velocity (vi) with Position (ro) and Velocity (vo) ---

    % --- Hyperbolas ---
    
    % --- F DOT + G DOT Function for Velocity in Hyperbolic Orbits ---
    function [vx] = vx_fg_hyperbola(a, ro_vec, rx_mag, u, H_delta, vo_vec)
        % Inputs:
        %   a: Semi-major axis (negative for hyperbolas) ro_vec: Initial
        %   position vector rx_mag: Magnitude of the final position vector
        %   u: Gravitational parameter H_delta: Hyperbolic anomaly
        %   difference vo_vec: Initial velocity vector
        % Outputs:
        %   vx: Final velocity vector in the e-p frame
        % Computes the final velocity vector using F and G functions for
        % hyperbolic orbits.
        f = -((sqrt(u * abs(a)) / (rx_mag * norm(ro_vec))) * sinh(H_delta)) * ro_vec;
        g = (1 - (abs(a) / rx_mag) * (cosh(H_delta) - 1)) * vo_vec;
        vx = f + g;
    end
    
    % --- Conics (General) ---
    
    % --- F DOT + G DOT Function for Velocity in Conic Orbits ---
    function [vx] = vx_fg_conics(p, ro_vec, vo_vec, u, ta_delta)
        % Inputs:
        %   p: Semi-latus rectum ro_vec: Initial position vector vo_vec:
        %   Initial velocity vector u: Gravitational parameter ta_delta:
        %   True anomaly difference
        % Outputs:
        %   vx: Final velocity vector in the e-p frame
        % Computes the final velocity vector using F and G dot functions
        % for general conic orbits.
        f = (((dot(ro_vec, vo_vec) / (p * norm(ro_vec))) * (1 - cos(ta_delta))) - ...
             ((1 / norm(ro_vec)) * (sqrt(u / p) * sin(ta_delta)))) * ro_vec;
        g = (1 - (norm(ro_vec) / p) * (1 - cos(ta_delta))) * vo_vec;
        vx = f + g;
    end
    
    % --- F DOT Function for Conic Orbits ---
    function [f] = f_dot_conics(p, ro_vec, vo_vec, u, ta_delta)
        % Inputs:
        %   p: Semi-latus rectum ro_vec: Initial position vector vo_vec:
        %   Initial velocity vector u: Gravitational parameter ta_delta:
        %   True anomaly difference
        % Outputs:
        %   f: F dot function value
        % Computes the F dot function for conic orbits.
        f = (((dot(ro_vec, vo_vec) / (p * norm(ro_vec))) * (1 - cos(ta_delta))) - ...
             ((1 / norm(ro_vec)) * (sqrt(u / p) * sin(ta_delta))));
    end
    
    % --- G DOT Function for Conic Orbits ---
    function [g] = g_dot_conics(ro, p, ta_delta)
        % Inputs:
        %   ro: Initial position vector p: Semi-latus rectum ta_delta: True
        %   anomaly difference
        % Outputs:
        %   g: G dot function value
        % Computes the G dot function for conic orbits.
        g = (1 - (norm(ro) / p) * (1 - cos(ta_delta)));
    end
    
    % --- Ellipses ---
    
    % --- F DOT + G DOT Function for Velocity in Elliptical Orbits ---
    function [vx] = vx_fg_ellipse(a, ro, rx_mag, u, E_delta, vo)
        % Inputs:
        %   a: Semi-major axis ro: Initial position vector rx_mag:
        %   Magnitude of the final position vector u: Gravitational
        %   parameter E_delta: Eccentric anomaly difference vo: Initial
        %   velocity vector
        % Outputs:
        %   vx: Final velocity vector in the e-p frame
        % Computes the final velocity vector using F and G dot functions
        % for elliptical orbits.
        f = -((sqrt(u * a) / (rx_mag * norm(ro))) * sin(deg2rad(E_delta))) * ro;
        g = (1 - (a / rx_mag) * (1 - cos(deg2rad(E_delta)))) * vo;
        vx = f + g;
    end
    
    % --- F DOT Function for Elliptical Orbits ---
    function [f] = f_vi_ellipse(a, ro, r_mag, u, E_delta)
        % Inputs:
        %   a: Semi-major axis ro: Initial position vector r_mag: Magnitude
        %   of the final position vector u: Gravitational parameter
        %   E_delta: Eccentric anomaly difference
        % Outputs:
        %   f: F dot function value
        % Computes the F dot function for elliptical orbits.
        f = -((sqrt(u * a) / (r_mag * norm(ro))) * sin(deg2rad(E_delta))) * ro;
    end
    
    % --- G DOT Function for Elliptical Orbits ---
    function [g] = g_vi_ellipse(a, vo, r_mag, E_delta)
        % Inputs:
        %   a: Semi-major axis vo: Initial velocity vector r_mag: Magnitude
        %   of the final position vector E_delta: Eccentric anomaly
        %   difference
        % Outputs:
        %   g: G dot function value
        % Computes the G dot function for elliptical orbits.
        g = (1 - (a / r_mag) * (1 - cos(deg2rad(E_delta)))) * vo;
    end

%% --- Anomalies ---

    % --- Eccentric Anomaly via Newton-Raphson Method ---
    function E_c = E_Me2(M, e)
        % Inputs:
        %   M: Mean Anomaly (radians) e: Eccentricity
        % Outputs:
        %   E_c: Eccentric Anomaly (radians)
        % Uses numerical methods (Newton-Raphson) to solve for the
        % Eccentric Anomaly.
        
        % Define the Kepler equation as an anonymous function
        keplerEq = @(E) E - e * sin(E) - M;
        
        % Initial guess for E. M is a reasonable starting point.
        E0 = M;
        
        % Solve the equation numerically using fzero
        E_c = fzero(keplerEq, E0);
    end
    
    % --- Eccentric Anomaly using Symbolic Solver ---
    function E_c = E_Me(obj, M, e)
        % Inputs:
        %   M: Mean Anomaly (radians) e: Eccentricity
        % Outputs:
        %   E_c: Eccentric Anomaly (radians)
        % Uses a symbolic solver to compute the Eccentric Anomaly.
        syms E
        E_c_sol = vpasolve(E - e * sin(E) == M, E);
        E_c = double(E_c_sol);  % Convert symbolic solution to double precision
    end
    
    % --- Mean Anomaly from Time Since Periapsis ---
    function [M] = M_ttp_ua(t_tp, u, a)
        % Inputs:
        %   t_tp: Time since periapsis passage u: Gravitational parameter
        %   a: Semi-major axis
        % Outputs:
        %   M: Mean Anomaly (radians)
        M = t_tp * sqrt(u / a^3);
    end
    
    % --- Mean Anomaly from Eccentric Anomaly ---
    function [M] = M_eE(obj, e, E)
        % Inputs:
        %   e: Eccentricity E: Eccentric Anomaly (radians)
        % Outputs:
        %   M: Mean Anomaly (radians)
        M = E - e * sin(E);
    end
    
    % --- True Anomaly from Orbital Radius, Velocity, Gravitational
    % Parameter, and Flight Path Angle ---
    function [ta] = ta_rvufpa(obj, r, v, u, fpa)
        % Inputs:
        %   r: Orbital radius v: Velocity u: Gravitational parameter fpa:
        %   Flight path angle
        % Outputs:
        %   ta: True Anomaly (degrees)
        % Calculates True Anomaly from orbital parameters.
        ta = atan((((r * v^2) / u) * cos(fpa) * sin(fpa)) / (((r * v^2) / u) * cos(fpa)^2 - 1));
        ta = [ta, ta - 180];  % Returns two possible true anomalies
    end
    
    % --- True Anomaly from Eccentricity and Eccentric Anomaly ---
    function [ta] = ta_eE(obj,e, E)
        % Inputs:
        %   e: Eccentricity E: Eccentric Anomaly (radians)
        % Outputs:
        %   ta: True Anomaly (radians)
        ta = 2 * atan(sqrt((1 + e) / (1 - e)) * tan(E / 2));
    end
    
    % --- True Anomaly from Orbital Radius, Eccentricity, and Semi-Latus
    % Rectum ---
    function [ta] = ta_rep(obj,r, e, p)
        % Inputs:
        %   r: Orbital radius e: Eccentricity p: Semi-latus rectum
        % Outputs:
        %   ta: True Anomaly (degrees)
        % Calculates two possible true anomaly values.
        ta = [acos((1 / e) * ((p / r) - 1)), -acos((1 / e) * ((p / r) - 1))];
    end
    
    % --- True Anomaly at Infinity for Hyperbolas ---
    function [ta] = ta_inf_e(e)
        % Inputs:
        %   e: Eccentricity
        % Outputs:
        %   ta: True Anomaly at infinity (degrees)
        ta = acos(1 / e);
    end
    
    % --- Eccentric Anomaly for Keplerian Orbits ---
    function [EccA] = EccA_are(obj, a, r, e)
        % Inputs:
        %   a: Semi-major axis r: Orbital radius e: Eccentricity
        % Outputs:
        %   EccA: Eccentric Anomaly (radians)
        % Calculates Eccentric Anomaly for ellipses (e < 1) and hyperbolas
        % (e > 1).
        if e < 1
            EccA = acos(max(min((a - r) / (a * e), 1), -1));  % Clamp value to [-1, 1]
        elseif e > 1
            EccA = acosh((abs(a) + r) / (abs(a) * e));
        end
    end
    
    % --- Eccentric Anomaly from True Anomaly ---
    function [EccA] = EccA_areta(obj, a, r, e, ta)
        % Inputs:
        %   a: Semi-major axis r: Orbital radius e: Eccentricity ta: True
        %   anomaly (radians)
        % Outputs:
        %   EccA: Eccentric Anomaly (radians)
        % Calculates Eccentric Anomaly given the True Anomaly.
        if e < 1
            cos_E = (a - r) / (a * e);
            cos_E = max(min(cos_E, 1), -1);  % Clamp the value to avoid numerical issues
            EccA = acos(cos_E);
            if ta > pi
                EccA = 2 * pi - EccA;  % Adjust for true anomalies in third and fourth quadrants
            end
        elseif e > 1
            EccA = acosh((abs(a) + r) / (abs(a) * e));
        end
    end


%% --- Flyby And Phase Angle Functions ---
    
    % --- Phase Angle (phi) Calculation ---
    function [phi] = phi_n_tof(obj,trans_angle, n_deg, tof_sec)
        % Inputs:
        %   trans_angle: Transfer angle (radians or degrees) n_deg: Mean
        %   motion (degrees per second) tof_sec: Time of flight (seconds)
        % Outputs:
        %   phi: Phase angle (radians or degrees)
        % Calculates the phase angle using the transfer angle, mean motion,
        % and time of flight.
        phi = trans_angle - n_deg * tof_sec;
    end
    
    % --- Flyby Angle for Hyperbolas ---
    function [flyby] = flyby_e(e)
        % Inputs:
        %   e: Eccentricity
        % Outputs:
        %   flyby: Flyby angle (radians)
        % Calculates the flyby angle for hyperbolic orbits based on the
        % eccentricity.
        flyby = 2 * asin(1 / e);
    end
    
    % --- Flight Path Angle as a Function of Eccentricity and True Anomaly
    % ---
    function [fpa] = fpa_eta(obj,e, ta)
        % Inputs:
        %   e: Eccentricity ta: True Anomaly (radians)
        % Outputs:
        %   fpa: Flight path angle (radians)
        % Calculates the flight path angle as a function of eccentricity
        % and true anomaly.
        fpa = atan((e * sin(ta)) / (1 + e * cos(ta)));
    end
    
    % --- Flight Path Angle as a Function of Angular Momentum, Orbital
    % Radius, and Velocity ---
    function [fpa] = fpa_hrv(obj, h, r, v)
        % Inputs:
        %   h: Specific angular momentum r: Orbital radius v: Velocity
        %   magnitude
        % Outputs:
        %   fpa: Flight path angle (radians)
        % Calculates the flight path angle based on angular momentum,
        % orbital radius, and velocity. Returns two possible values (one
        % positive, one negative).
        fpa = [acos(h / (r * v)), -acos(h / (r * v))];
    end

        
        
%% --- Eccentricities ---

    % --- Eccentricity for Hyperbolic Orbits from Flyby Angle ---
    function [e] = e_flyby(obj, flyby)
        % Inputs:
        %   flyby: Flyby angle (radians)
        % Outputs:
        %   e: Eccentricity
        % Calculates eccentricity based on the flyby angle for hyperbolic
        % orbits.
        e = 1 / (sin(flyby / 2));
    end
    
    % --- Eccentricity as a Function of Two Orbital Radii and True
    % Anomalies ---
    function [e] = e_r1r2_ta1_ta2(obj, r1, r2, ta1, ta2)
        % Inputs:
        %   r1: First orbital radius r2: Second orbital radius ta1: First
        %   true anomaly (radians) ta2: Second true anomaly (radians)
        % Outputs:
        %   e: Eccentricity
        % Calculates eccentricity based on two positions (radii) and two
        % true anomalies.
        e = (r1 - r2) / (r2 * cos(ta2) - r1 * cos(ta1));
    end
    
    % --- Eccentricity for Ellipses from Semi-Major Axis and Periapsis ---
    function [e] = e_ellps_arp(obj, a, rp)
        % Inputs:
        %   a: Semi-major axis rp: Periapsis radius
        % Outputs:
        %   e: Eccentricity
        % Calculates eccentricity for elliptical orbits.
        e = (a - rp) / a;
    end
    
    % --- Eccentricity for Hyperbolas from Semi-Major Axis and Periapsis
    % ---
    function [e] = e_hyp_arp(a, rp)
        % Inputs:
        %   a: Semi-major axis (negative for hyperbolas) rp: Periapsis
        %   radius
        % Outputs:
        %   e: Eccentricity
        % Calculates eccentricity for hyperbolic orbits.
        e = (rp / abs(a)) + 1;
    end
    
    % --- Eccentricity as a Function of Specific Energy, Angular Momentum,
    % and Gravitational Parameter ---
    function [e] = e_esphu(obj, esp, h, u)
        % Inputs:
        %   esp: Specific orbital energy h: Specific angular momentum u:
        %   Gravitational parameter
        % Outputs:
        %   e: Eccentricity
        % Calculates eccentricity based on specific energy, angular
        % momentum, and gravitational parameter.
        e = sqrt(1 + ((2 * esp * h^2) / u^2));
    end
    
    % --- Eccentricity as a Function of Radius, Velocity, Gravitational
    % Parameter, and Flight Path Angle ---
    function [e] = e_ruvfpa(obj, r, u, v, fpa)
        % Inputs:
        %   r: Orbital radius u: Gravitational parameter v: Velocity fpa:
        %   Flight path angle (radians)
        % Outputs:
        %   e: Eccentricity
        % Calculates eccentricity based on radius, velocity, gravitational
        % parameter, and flight path angle.
        e = sqrt((((((r * v^2) / u) - 1)^2) * (cos(fpa))^2) + (sin(fpa))^2);
    end
    
    % --- Eccentricity as a Function of Angular Momentum, Specific Energy,
    % and Gravitational Parameter ---
    function [e] = e_concs_hespu(obj, h, esp, u)
        % Inputs:
        %   h: Specific angular momentum esp: Specific orbital energy u:
        %   Gravitational parameter
        % Outputs:
        %   e: Eccentricity
        % General formula for calculating eccentricity for all conics.
        e = sqrt(1 + (2 * esp * h^2) / u^2);
    end
    
    % --- Eccentricity for Hyperbolas from Periapsis and Semi-Major Axis
    % ---
    function [e] = e_hypbl_arp(obj, rp, a)
        % Inputs:
        %   rp: Periapsis radius a: Semi-major axis (negative for
        %   hyperbolas)
        % Outputs:
        %   e: Eccentricity
        % Calculates eccentricity for hyperbolic orbits based on periapsis
        % and semi-major axis.
        e = (rp / abs(a)) + 1;
    end
    
    % --- Eccentricity for Ellipses from Semi-Latus Rectum and Semi-Major
    % Axis ---
    function [e] = e_ellps_pa(obj, p, a)
        % Inputs:
        %   p: Semi-latus rectum a: Semi-major axis
        % Outputs:
        %   e: Eccentricity
        % Calculates eccentricity for elliptical orbits.
        e = sqrt(1 - (p / a));
    end
    
    % --- Eccentricity for Hyperbolas from Semi-Latus Rectum and Semi-Major
    % Axis ---
    function [e] = e_hyper_pa(obj, p, a)
        % Inputs:
        %   p: Semi-latus rectum a: Semi-major axis (negative for
        %   hyperbolas)
        % Outputs:
        %   e: Eccentricity
        % Calculates eccentricity for hyperbolic orbits.
        e = sqrt(1 + (p / abs(a)));
    end
    
    % --- Eccentricity as a Function of Periapsis and Apoapsis Radii ---
    function [e] = e_rpra(obj, rp, ra)
        % Inputs:
        %   rp: Periapsis radius ra: Apoapsis radius
        % Outputs:
        %   e: Eccentricity
        % Calculates eccentricity based on periapsis and apoapsis radii.
        e = (ra - rp) / (ra + rp);
    end
    
    % --- Mean Motion as a Function of Gravitational Parameter and
    % Semi-Major Axis ---
    function [n] = n_ua(obj, u, a)
        % Inputs:
        %   u: Gravitational parameter a: Semi-major axis
        % Outputs:
        %   n: Mean motion (radians per second)
        % Calculates mean motion based on gravitational parameter and
        % semi-major axis.
        n = sqrt(u / a^3);
    end


%% --- Specific Energy And Angular Momentum ---
    
    % --- Specific Angular Momentum Magnitude ---
    function [h] = h_mag_vprp(obj, vp, rp)
        % Inputs:
        %   vp: Velocity at periapsis rp: Radius at periapsis
        % Outputs:
        %   h: Specific angular momentum magnitude
        % Calculates specific angular momentum magnitude from velocity and
        % radius at periapsis.
        h = vp * rp;
    end
    
    % --- Specific Angular Momentum ---
    function [h] = h_uae(obj, u, a, e)
        % Inputs:
        %   u: Gravitational parameter a: Semi-major axis e: Eccentricity
        % Outputs:
        %   h: Specific angular momentum
        % Calculates specific angular momentum for conic orbits.
        h = sqrt(u * a * (1 - e^2));
    end
    
    % --- Specific Energy for General Conic Orbits ---
    function [esp] = esp_vur(obj, v, u, r)
        % Inputs:
        %   v: Velocity magnitude u: Gravitational parameter r: Orbital
        %   radius
        % Outputs:
        %   esp: Specific orbital energy
        % Calculates specific orbital energy for general conic sections.
        esp = (v^2 / 2) - (u / r);
    end
    
    % --- Specific Energy for Elliptical and Hyperbolic Orbits ---
    function [esp] = esp_ua(obj, u, a, E_or_H)
        % Inputs:
        %   u: Gravitational parameter a: Semi-major axis E_or_H: Orbit
        %   type ('E' for Ellipse, 'H' for Hyperbola)
        % Outputs:
        %   esp: Specific orbital energy
        % Calculates specific orbital energy for elliptical and hyperbolic
        % orbits.
        if E_or_H == "E" || E_or_H == "e"
            esp = -u / (2 * abs(a));  % Elliptical
        elseif E_or_H == "H" || E_or_H == "h"
            esp = u / (2 * abs(a));   % Hyperbolic
        end
    end


%% --- Time-Related Quantities ---

    % --- Orbital Period as a Function of Gravitational Parameter and
    % Semi-Major Axis ---
    function [p] = period_ua(obj, u, a)
        % Inputs:
        %   u: Gravitational parameter a: Semi-major axis
        % Outputs:
        %   p: Orbital period
        % Calculates the orbital period for elliptical orbits.
        p = (2 * pi) / sqrt(u / a^3);
    end
    
    % --- Time to Periapsis as a Function of Eccentric Anomalies,
    % Eccentricity, and Orbital Parameters ---
    function [tx_tp] = txtp_EccAn_eua(obj,EccAn, e, u, a)
        % Inputs:
        %   EccAn: Eccentric Anomaly (radians) e: Eccentricity u:
        %   Gravitational parameter a: Semi-major axis
        % Outputs:
        %   tx_tp: Time to periapsis passage
        % Calculates time to periapsis passage for elliptical and
        % hyperbolic orbits.
        if e > 1  % Hyperbolic
            tx_tp = (e * sinh(EccAn) - EccAn) / sqrt(u / abs(a)^3);
        elseif e < 1  % Elliptical
            tx_tp = (EccAn - e * sin(EccAn)) / sqrt(u / abs(a)^3);
        end
    end


%% --- Maneuvers ---

    % --- Delta-V for Departure in a Parking Orbit ---
    function [Delta_v] = delta_v_dept_prk(v_inf_dept, u_planet, rad_prk_orb, v_prk_orb)
        % Inputs:
        %   v_inf_dept: Excess velocity at departure (km/s) u_planet:
        %   Gravitational parameter of the planet (km^3/s^2) rad_prk_orb:
        %   Radius of the parking orbit (km) v_prk_orb: Orbital velocity in
        %   the parking orbit (km/s)
        % Outputs:
        %   Delta_v: Required delta-v for the departure (km/s)
        % Assumes a coplanar, circular, Hohmann transfer maneuver.
        Delta_v = sqrt(v_inf_dept^2 + ((2 * u_planet / rad_prk_orb))) - v_prk_orb;
    end


%% --- Space Triangle Functions And Lambert ---

    % --- Hypotenuse of a Triangle from Two Sides (r1 and r2) ---
    function [hyp] = trngl_hyp_r1r2(r1, r2)
        % Inputs:
        %   r1: Length of the first side r2: Length of the second side
        % Outputs:
        %   hyp: Hypotenuse of the triangle
        hyp = sqrt(r1^2 + r2^2);
    end
    
    % --- Semi-Perimeter of the Triangle ---
    function [semi_per] = trngl_semiper_r1r2c(r1, r2, chord)
        % Inputs:
        %   r1: Length of the first side r2: Length of the second side
        %   chord: Length of the chord
        % Outputs:
        %   semi_per: Semi-perimeter of the triangle
        semi_per = (r1 + r2 + chord) / 2;
    end
    
    % --- Minimum Semi-Major Axis ---
    function [a_min] = trngl_a_min_semiper(semiper)
        % Inputs:
        %   semiper: Semi-perimeter of the triangle
        % Outputs:
        %   a_min: Minimum semi-major axis
        a_min = semiper / 2;
    end
    
    % --- Distance to the Focus from the Minimum Energy Ellipse ---
    function [dist_FQ_fmin] = trngl_dist_FQ_fmin(a_min, r1_or_r2)
        % Inputs:
        %   a_min: Minimum semi-major axis r1_or_r2: Orbital radius (r1 or
        %   r2)
        % Outputs:
        %   dist_FQ_fmin: Distance to the focus from the minimum energy
        %   ellipse
        dist_FQ_fmin = 2 * a_min - r1_or_r2;
    end
    
    % --- Alpha Angle for Ellipses ---
    function [alpha] = trngl_alpha(a, semiper)
        % Inputs:
        %   a: Semi-major axis semiper: Semi-perimeter of the triangle
        % Outputs:
        %   alpha: Alpha angle (radians)
        s = semiper;
        alpha = 2 * asin(sqrt(s / (2 * a)));
    end
    
    % --- Beta Angle for Ellipses ---
    function [beta] = trngl_beta(a, semiper, chord)
        % Inputs:
        %   a: Semi-major axis semiper: Semi-perimeter of the triangle
        %   chord: Chord length
        % Outputs:
        %   beta: Beta angle (radians)
        s = semiper;
        c = chord;
        beta = 2 * asin(sqrt((s - c) / (2 * a)));
    end
    
    % --- Alpha Angle for Hyperbolas ---
    function [alpha] = trngl_alpha_Hyp(a, semiper)
        % Inputs:
        %   a: Semi-major axis (negative for hyperbolas) semiper:
        %   Semi-perimeter of the triangle
        % Outputs:
        %   alpha: Alpha angle for hyperbolic orbits (radians)
        s = semiper;
        alpha = 2 * asinh(sqrt(s / (2 * a)));
    end
    
    % --- Beta Angle for Hyperbolas ---
    function [beta] = trngl_beta_Hyp(a, semiper, chord)
        % Inputs:
        %   a: Semi-major axis (negative for hyperbolas) semiper:
        %   Semi-perimeter of the triangle chord: Chord length
        % Outputs:
        %   beta: Beta angle for hyperbolic orbits (radians)
        s = semiper;
        c = chord;
        beta = 2 * asinh(sqrt((s - c) / (2 * a)));
    end
    
    % --- Elliptical Transfer Function (P) ---
    function [P] = trngl_P_ellipse(a, semiper, chord, r1, r2, alpha, beta)
        % Inputs:
        %   a: Semi-major axis semiper: Semi-perimeter of the triangle
        %   chord: Chord length r1: Orbital radius 1 r2: Orbital radius 2
        %   alpha: Alpha angle (radians) beta: Beta angle (radians)
        % Outputs:
        %   P: Elliptical transfer function
        s = semiper;
        c = chord;
        P = [((4 * a * (s - r1) * (s - r2)) / c^2) * (sin((alpha + beta) / 2))^2, ...
             ((4 * a * (s - r1) * (s - r2)) / c^2) * (sin((alpha - beta) / 2))^2];
    end
    
    % --- Hyperbolic Transfer Function (P) ---
    function [P] = trngl_P_hyperb(a, semiper, chord, r1, r2, alpha, beta)
        % Inputs:
        %   a: Semi-major axis (negative for hyperbolas) semiper:
        %   Semi-perimeter of the triangle chord: Chord length r1: Orbital
        %   radius 1 r2: Orbital radius 2 alpha: Alpha angle (radians)
        %   beta: Beta angle (radians)
        % Outputs:
        %   P: Hyperbolic transfer function
        s = semiper;
        c = chord;
        P = [((4 * abs(a) * (s - r1) * (s - r2)) / c^2) * (sinh((alpha + beta) / 2))^2, ...
             ((4 * abs(a) * (s - r1) * (s - r2)) / c^2) * (sinh((alpha - beta) / 2))^2];
    end
    
    % --- Time of Flight for Parabolic Orbits ---
    function [TOF_parab] = trngl_TOF_parab(u, semiper, chord, type)
        % Inputs:
        %   u: Gravitational parameter semiper: Semi-perimeter of the
        %   triangle chord: Chord length type: Type of transfer (1 or 2)
        % Outputs:
        %   TOF_parab: Time of flight for parabolic orbits
        s = semiper;
        c = chord;
        if type == 1
            TOF_parab = (1 / 3) * sqrt(2 / u) * (s^(3/2) - (s - c)^(3/2));
        elseif type == 2
            TOF_parab = (1 / 3) * sqrt(2 / u) * (s^(3/2) + (s - c)^(3/2));
        end
    end
    
    % --- Time of Flight for Elliptical and Hyperbolic Orbits ---
    function [TOF] = trngl_TOF(u, a, semiper, chord, type)
        % Inputs:
        %   u: Gravitational parameter a: Semi-major axis semiper:
        %   Semi-perimeter of the triangle chord: Chord length type: Transfer
        %   type ('1A', '1B', '2A', '2B', '1H')
        % Outputs:
        %   TOF: Time of flight (seconds)
        s = semiper;
        c = chord;
        alpha_0 = 2 * asin(sqrt(s / (2 * a)));  % Radians
        beta_0  = 2 * asin(sqrt((s - c) / (2 * a)));  % Radians

        if type == "1A"
            TOF = (sqrt(a^3 / u)) * ((alpha_0 - sin(alpha_0)) - (beta_0 - sin(beta_0)));
        elseif type == "1B"
            TOF = (sqrt(a^3 / u)) * (2 * pi - (alpha_0 - sin(alpha_0)) - (beta_0 - sin(beta_0)));
        elseif type == "2A"
            TOF = (sqrt(a^3 / u)) * ((alpha_0 - sin(alpha_0)) + (beta_0 - sin(beta_0)));
        elseif type == "2B"
            TOF = (sqrt(a^3 / u)) * (2 * pi - (alpha_0 - sin(alpha_0)) + (beta_0 - sin(beta_0)));
        elseif type == "1H"
            alpha_0 = 2 * asinh(sqrt(s / (2 * abs(a))));
            beta_0  = 2 * asinh(sqrt((s - c) / (2 * abs(a))));
            TOF = (sqrt(a^3 / u)) * (sinh(alpha_0) - alpha_0 - (sinh(beta_0) - beta_0));
        end
    end


%% ====== 2BP Propagator No Perturbation ======
    function [x_dot] = dynamics_2BP_cartesian(obj, t, X, mu)
        % DYNAMICS_2BP_CARTESIAN Computes the derivative of the state vector
        % for a spacecraft under a simple 2-body gravitational model (no J2).
        %
        % Usage:
        %   x_dot = dynamics_2BP_cartesian(t, X, mu)
        %
        % INPUTS:
        %   t   - Time [s] (unused; included for ODE solver compatibility)
        %   X   - State vector [6x1]: [x; y; z; vx; vy; vz] (position in km, velocity in km/s)
        %   mu  - Gravitational parameter [km^3/s^2]
        %
        % OUTPUT:
        %   x_dot - Derivative of the state vector [6x1]
    
        % Extract position components
        x = X(1); y = X(2); z = X(3);
    
        % Extract velocity components
        vx = X(4); vy = X(5); vz = X(6);
    
        % Compute distance from the central body
        r = sqrt(x^2 + y^2 + z^2);
    
        % Compute gravitational acceleration (2-body problem)
        ax = -mu * x / r^3;
        ay = -mu * y / r^3;
        az = -mu * z / r^3;
    
        % Assemble the derivative of the state vector
        x_dot = [vx; vy; vz; ax; ay; az];
    end

   
    %% ====== 2BP Jacobian (Cartesian, no perturbations) ======
    function A = jacobian_2BP_cartesian(obj,t, X, mu)
        % JACOBIAN_2BP_CARTESIAN  Returns the state Jacobian ∂f/∂X for the
        % unperturbed 2-body problem in Cartesian coordinates.
        %
        % Usage:
        %   A = jacobian_2BP_cartesian(t, X, mu)
        %
        % INPUTS:
        %   t   - Time [s] (unused; included for ODE/STM interface compatibility)
        %   X   - State vector [6x1]: [x; y; z; vx; vy; vz] (km, km/s)
        %   mu  - Gravitational parameter [km^3/s^2]
        %
        % OUTPUT:
        %   A   - Jacobian matrix ∂f/∂X evaluated at (t, X) [6x6]
        %
        % Dynamics recap:
        %   f(X) = [ v ;
        %            -mu * r / ||r||^3 ]
        %   with r = [x; y; z], v = [vx; vy; vz]
        %
        %   ∂f/∂X =
        %     [ 0_3   I_3 ;
        %       ∂a/∂r 0_3 ]
        %
        %   where  ∂a/∂r = μ * ( 3*r*rᵀ/||r||^5 - I_3/||r||^3 )

        % 't' is unused by design

        % Extract position
        x = X(1); y = X(2); z = X(3);

        % Distance and powers
        r2 = x*x + y*y + z*z;
        r  = sqrt(r2);

        % Protect against singularity at r ≈ 0
        if r < 1e-12
            error('jacobian_2BP_cartesian: singular state (||r|| ≈ 0).');
        end

        r3 = r2 * r;
        r5 = r2 * r3;

        % 3x3 block: ∂a/∂r = μ * (3 rrᵀ / r^5 - I / r^3)        
        % Build rrᵀ explicitly
        rrT = [x*x, x*y, x*z;
            y*x, y*y, y*z;
            z*x, z*y, z*z];

        I3   = eye(3);
        dadr = mu * ( 3.0 * rrT / r5 - I3 / r3 );

        % Assemble full 6x6 Jacobian
        A = zeros(6,6);
        A(1:3,4:6) = I3;      % ∂(ṙ)/∂v = I
        A(4:6,1:3) = dadr;    % ∂(v̇)/∂r
        % ∂(ṙ)/∂r = 0, ∂(v̇)/∂v = 0 already

    end

    %% ====== STATE TRANSITION MATRIX Propagators ======
    function [Phi_T, X_T] = propagateSTM_2BP(obj, X0, mu, tspan, opts)
        Phi0  = eye(6);
        Xaug0 = [X0(:); Phi0(:)];
        rhs   = @(t,X) augmentedDynamics2BP_STM(obj, t, X, mu);
        [~, Xaug] = ode89(rhs, tspan, Xaug0, opts);
        Xaug_T = Xaug(end,:).';
        X_T    = Xaug_T(1:6);
        Phi_T  = reshape(Xaug_T(7:end), 6, 6);
    end


    function dXaug = augmentedDynamics2BP_STM(obj,t, Xaug, mu)
        % State + STM dynamics for the unperturbed 2-body problem (Cartesian).
        % Xaug = [x;y;z;vx;vy;vz; vec(Phi)], where Phi is 6x6 (column-stacked).

        %  split state and STM
        X   = Xaug(1:6);
        Phi = reshape(Xaug(7:end), 6, 6);

        %  state dynamics
        x = X(1); y = X(2); z = X(3);
        r3 = (x*x + y*y + z*z)^(3/2);
        ax = -mu*x / r3;
        ay = -mu*y / r3;
        az = -mu*z / r3;
        dX = [X(4); X(5); X(6); ax; ay; az];

        %  Jacobian A = ∂f/∂X
        A = obj.jacobian_2BP_cartesian(0, X, mu);        % t unused

        %  STM dynamics
        dPhi = A * Phi;

        %  pack
        dXaug = [dX; dPhi(:)];
    end


    %% ====== 2BP Propagators With Perturbation ======

    function [x_dot] = dynamics_2BP_cartesian_J2(t,X, mu, J2, ro)
            % DYNAMICS_2BP_CARTESIAN_J2 Calculates the state vector
            % derivative for a spacecraft in a perturbed two-body problem
            % in Cartesian coordinates considering J2 perturbation.
            %
            % Usage:
            %   x_dot = dynamics_2BP_cartesian_J2(t, X)
            %
            % INPUTS:
            %   t   - Time variable [s]. Included for compatibility with ODE solvers
            %         but not used in the computation as the system is time-invariant.
            %   X   - State vector [6x1] comprising position and velocity components:
            %         X = [x; y; z; vx; vy; vz], where:
            %           x, y, z   - Position components in the inertial frame [km]
            %           vx, vy, vz - Velocity components in the inertial frame [km/s]
            %   mu  - Gravitational parameter of the central body [km^3/s^2]
            %   J2  - Second zonal harmonic coefficient (dimensionless), representing
            %         the oblateness of the central body
            %   ro  - Equatorial radius of the central body [km]
            %
            % Where:
            %   t is the current time (unused as the problem is autonomous)
            %   X is the state vector [x; y; z; vx; vy; vz] x_dot is the
            %   derivative of the state vector
            %
            % The function computes the gravitational acceleration
            % including the J2 perturbation from Earth's oblateness and
            % returns the derivative of the position and velocity of the
            % spacecraft.
                               
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
            factor = -1.5 * J2 * mu * (ro^2) / (r^5);
        
            aj2_x = factor * (x / r) * (1 - 5 * z2_r2);
            aj2_y = factor * (y / r) * (1 - 5 * z2_r2);
            aj2_z = factor * (z / r) * (3 - 5 * z2_r2);
           
            % Include the J2 perturbation in the acceleration components
            dvxdt = dvxdt + aj2_x;
            dvydt = dvydt + aj2_y;
            dvzdt = dvzdt + aj2_z;
            
            % Assemble the derivative of the state vector including the J2
            % perturbation
            x_dot = [vx; vy; vz; dvxdt; dvydt; dvzdt];
    end
        
    function [x_dot] = dynamics_2BP_cartesian_J2_SRP(t,X)
        % DYNAMICS_2BP_CARTESIAN_J2 Calculates the state vector derivative
        % for a spacecraft in a perturbed two-body problem in Cartesian
        % coordinates considering J2 and SRP perturbation.
        %
        % Usage:
        %   x_dot = dynamics_2BP_cartesian_J2(t, X)
        %
        % Where:
        %   t is the current time (unused as the problem is autonomous) X
        %   is the state vector [x; y; z; vx; vy; vz] x_dot is the
        %   derivative of the state vector
        %
        % The function computes the gravitational acceleration including
        % the J2 perturbation from Earth's oblateness and returns the
        % derivative of the position and velocity of the spacecraft.
    
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
        
        % Calculate the gravitational acceleration components 2BP based f0
        % rate of change of state vector
        f0 = [v_vec_eci ; -mu * x / r^3; -mu * y / r^3 ; -mu * z / r^3];
               
        aj2_vec_eci = calc_J2_accel_cartesian(r_vec_eci);
        
        % Calculate cannon ball perturbation in cartesian
        aSRP_vec_eci = calc_SRP_accel_cartesian(t, r_vec_eci);
        
        % B matrix
        B = [zeros(3,3);eye(3)];
        
        % Assemble the derivative of the state vector including the J2
        % perturbation
        x_dot = f0 + B * (aj2_vec_eci + aSRP_vec_eci);    
    end
        
    function [x_dot,r_vec_eci] = dynamics_2BP_milankovitch_J2(t, X)
    
        % DYNAMICS_2BP_MILANKOVITCH_J2 Propagates the state of an orbiting
        % object under the influence of J2 perturbations using Milankovitch
        % orbital elements. This function computes the time derivative of
        % the state vector and the position vector in the ECI frame for a
        % given set of Milankovitch elements.
        %
        % Inputs:
        %   t - Time variable (
        % not used in this function, included for ode45 compatibility).
        %   X - State vector in Milankovitch elements:
        %       X(1:3) - Specific angular momentum vector (h_vec) in ECI
        %       frame. X(4:6) - Eccentricity vector (e_vec) in ECI frame.
        %       X(7)   - True longitude (L), in radians.
        %   mu - Stanard gravitational parameter of the central body.
        %   libCall - Library call for additional functions like
        %   'OrbitalElementsToDCM'.
        %
        % Outputs:
        %   x_dot - Time derivative of the state vector. r_vec_eci -
        %   Position vector in the ECI frame.
        %
        % Example call:
        %   [x_dot, r_vec_eci] = dynamics_2BP_milankovitch_J2(t, X, mu,
        %   AAE590Functions);
        % Where:
        %   - t is the time (scalar) - X is the current state vector (7x1
        %   vector) - mu is the gravitational parameter (scalar)
        % The function returns the time derivative of the state vector
        % (x_dot) and the
        
        libCall =  KeplerianOrbitalMechanicsLibrary();
        
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
        i = acos(hz/h_mag);                   % Recover inclination from Milankovitch X0     
        n_vec_eci = cross(z_vec,h_hat_eci);    % Find Line of Nodes unit vector 
        a = p / (1 - e_mag^2);                 % Calculate the semimajor axis
    
        % Normalize the line of nodes vector to get the LoN unit vector
        n_vec_hat = n_vec_eci / norm(n_vec_eci);
    
        RAAN = acos(dot(x_vec,n_vec_hat));         % RAAN angle in degrees is angle between LINE OF NODES and X+
        omega = acos(dot(n_vec_hat,e_vec_hat));    % Argument of periapsis angle
        nu = rad2deg(L) - RAAN - omega;             % True anomaly angle
        omega_plus_nu = rad2deg(L) - RAAN;  % Calculate longitude of periapsis nu+omega
    
        if n_vec_hat(2) < 0 % Correct the quadrant for RAAN
            RAAN = 360 - RAAN;
        end
        
        % Calculate DCM313 at current point using angles found above
        DCM313 = libCall.OrbitalElementsToDCM(RAAN,i,omega_plus_nu); 
    
        % Compute the position vector in ECI coordinates with new DCM313 We
        % start from the position vector in rotating frame of ref
        r_mag = (a*(1-e_mag^2)) / (1+(e_mag*cos(nu)));    
        r_vec_rot = [r_mag;0;0];
    
        % Now convert r vector in RTN (rotating) to ECI frame
        r_vec_eci  = libCall.vec_rot_to_cart(DCM313,r_vec_rot)';
    
        % Calculate the J2 perturbation acceleration in ECI coordinates
        aj2_vec_eci = calc_J2_accel_cartesian(r_vec_eci);
    
        % Calculate the velocity vector in rotation frame (for ease) for
        % Gaussian planetary elements, matrix B
        
        % Find flight path angle as a function of e and nu first
        fpa = atan((e_mag*sin(nu)/(1+e_mag*cos(nu))));
        
        % Find speed magnitude as a funtion for mu, r, a
        v_mag = sqrt(2*(mu/r_mag)-(mu/a));
    
        % Velocity as vector RTN (rotating) frame
        v_vec_rot = [v_mag*sin(fpa),v_mag*cos(fpa),0]; 
    
        % Velocity vector from RTN (rotating) to ECI frame
        v_vec_eci  = libCall.vec_rot_to_cart(DCM313,v_vec_rot)';
        
        % Unperturbed Component for rate of change of state vector f_0(x)
        % computation
        f0_milankovitch = [0; 0; 0; 0; 0; 0; h_mag/r_mag^2]; 
                          
        % Define the B matrix using Gaussian planetary equations for
        % Milankovitch
        
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
    
    function [x_dot,r_vec_eci] = dynamics_2BP_milankovitch_J2_SRP(t, X)
    
        % DYNAMICS_2BP_MILANKOVITCH_J2 Propagates the state of an orbiting
        % object under the influence of J2 and SRP perturbations using
        % Milankovitch orbital elements. This function computes the time
        % derivative of the state vector and the position vector in the ECI
        % frame for a given set of Milankovitch elements.
        %
        % Inputs:
        %   t - Time variable (
        % not used in this function, included for ode45 compatibility).
        %   X - State vector in Milankovitch elements:
        %       X(1:3) - Specific angular momentum vector (h_vec) in ECI
        %       frame. X(4:6) - Eccentricity vector (e_vec) in ECI frame.
        %       X(7)   - True longitude (L), in radians.
        %   mu - Stanard gravitational parameter of the central body.
        %   libCall - Library call for additional functions like
        %   'OrbitalElementsToDCM'.
        %
        % Outputs:
        %   x_dot - Time derivative of the state vector. r_vec_eci -
        %   Position vector in the ECI frame.
        %
        % Example call:
        %   [x_dot, r_vec_eci] = dynamics_2BP_milankovitch_J2(t, X, mu,
        %   AAE590Functions);
        % Where:
        %   - t is the time (scalar) - X is the current state vector (7x1
        %   vector) - mu is the gravitational parameter (scalar)
        % The function returns the time derivative of the state vector
        % (x_dot) and the
        
        libCall =  KeplerianOrbitalMechanicsLibrary();
            
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
        i = acos(hz/h_mag);                   % Recover inclination from Milankovitch X0     
        n_vec_eci = cross(z_vec,h_hat_eci);    % Find Line of Nodes unit vector 
        a = p / (1 - e_mag^2);                 % Calculate the semimajor axis
    
        % Normalize the line of nodes vector to get the LoN unit vector
        n_vec_hat = n_vec_eci / norm(n_vec_eci);
    
        RAAN = acos(dot(x_vec,n_vec_hat));         % RAAN angle in degrees is angle between LoN and X+
        omega = acos(dot(n_vec_hat,e_vec_hat));    % Argument of periapsis angle
        nu = rad2deg(L) - RAAN - omega;                      % True anomaly angle
        omega_plus_nu = rad2deg(L) - RAAN;  % Calculate longitude of periapsis nu+omega
    
        if n_vec_hat(2) < 0 % Correct the quadrant for RAAN
            RAAN = 360 - RAAN;
        end
        
        % Calculate DCM313 at current point using angles found above
        DCM313 = libCall.OrbitalElementsToDCM(RAAN,i,omega_plus_nu); 
    
        % Compute the position vector in ECI coordinates with new DCM313 We
        % start from the position vector in rotating frame of ref
        r_mag = (a*(1-e_mag^2)) / (1+(e_mag*cos(nu)));    
        r_vec_rot = [r_mag;0;0];
    
        % Now convert r vector in RTN (rotating) to ECI frame
        r_vec_eci  = libCall.vec_rot_to_cart(DCM313,r_vec_rot)';
    
        % Calculate the J2 perturbation acceleration in ECI coordinates
        aj2_vec_eci = calc_J2_accel_cartesian(r_vec_eci);
    
        % Calculate SRP cannonball perturbation in ECI
        aSRP_vec_eci = calc_SRP_accel_cartesian(t, r_vec_eci);
    
        % Calculate the velocity vector in rotation frame (for ease) for
        % Gaussian planetary elements, matrix B
        
        % Find flight path angle as a function of e and nu first
        fpa = atan((e_mag*sin(nu)/(1+e_mag*cos(nu))));
        
        % Find speed magnitude as a funtion for mu, r, a
        v_mag = sqrt(2*(mu/r_mag)-(mu/a));
    
        % Velocity as vector RTN (rotating) frame
        v_vec_rot = [v_mag*sin(fpa),v_mag*cos(fpa),0]; 
    
        % Velocity vector from RTN (rotating) to ECI frame
        v_vec_eci  = libCall.vec_rot_to_cart(DCM313,v_vec_rot)';
        
        % Unperturbed Component for rate of change of state vector f_0(x)
        % computation
        f0_milankovitch = [0; 0; 0; 0; 0; 0; h_mag/r_mag^2]; 
                          
        % Define the B matrix using Gaussian planetary equations for
        % Milankovitch
        
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
        L = X(6);       %   [rads]Ø
        
        % Unperturbed Component for rate of change of state vector f_0(x)
        % computation
        
        q = 1 + f*cos(L) + g*sin(L);        % L is in [rad]    
        s_sq = 1 + h^2 + k^2;
        L_dot = sqrt(mu*p) * (q/p)^2;       %in rad/s, uses mean motion
        
        f0_equinoctial = [0; 0; 0; 0; 0; L_dot]; % Only L_dot is non-zero for unperturbed motion
    
        % Recover COE angles from modified equinoctial state
        RAAN = atan2d(k, h);                % RAAN [deg]
        i = 2 * atan(sqrt(h^2 + k^2));     % Inclination [deg]
        omega_plus_nu = rad2deg(L) - RAAN;  % [deg]       
                   
        % Compute the position vector in ECI coordinates with new DCM313
        r = p / q;                     % (Walker, 1985. pg4) 
    
    
        % Calculate DCM313 Matrix - RTN (rotating) to ECI
        DCM313 =  libCall.OrbitalElementsToDCM(RAAN,i,omega_plus_nu);
    
        % Compute the position vector in ECI coordinates with new DCM313
        r_vec_rot = [r; 0; 0];  % [km]
        r_vec_eci = libCall.vec_rot_to_cart(DCM313,r_vec_rot)';
     
        % Calculate the J2 perturbation in the RTN frame (rotating)
        % components as per document shared in Brightspace
        C = - (mu*J2*r0^2)/(r^4);
        aj2R = 1.5 * C * (1 - (12 * (h * sin(L) - k * cos(L))^2) / (1 + h^2 + k^2)^2);
        aj2T = 12 * C * ((h * sin(L) - k * cos(L)) * (h * cos(L) + k * sin(L))) / ((1 + h^2 + k^2)^2);
        aj2N = 6 * C * ((1 - h^2 - k^2) * (h * sin(L) - k * cos(L))) / ((1 + h^2 + k^2)^2);
        
        % Perturbation J2 vector in RTN direction
        aj2_vec_rtn = [aj2R; aj2T; aj2N];
    
        % Define the B matrix from the image provided
        B = sqrt(p/mu) * ...
            [0,        2*p/q,                    0;
             sin(L),  ((q+1)*cos(L)+f)/q,     -g*(h*sin(L)-k*cos(L))/q;
             -cos(L), ((q+1)*sin(L)+g)/q,     f*(h*sin(L)-k*cos(L))/q;
             0,        0,                       s_sq*cos(L)/(2*q);
             0,        0,                       s_sq*sin(L)/(2*q);
             0,        0,                       (h*sin(L) - k*cos(L))/q];
    
        % Compute the perturbed rate of change of the equinoctial elements
        x_dot = f0_equinoctial + B * aj2_vec_rtn;
    end     
    
    function [x_dot,r_vec_eci] = dynamics_2BP_equinoctial_J2_SRP(t, X)
        
        libCall =  KeplerianOrbitalMechanicsLibrary();
        
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
        
        % Unperturbed Component for rate of change of state vector f_0(x)
        % computation
        
        q = 1 + f*cos(L) + g*sin(L);        % L is in [rad]    
        s_sq = 1 + h^2 + k^2;
        L_dot = sqrt(mu*p) * (q/p)^2;       %in rad/s, uses mean motion
        
        f0_equinoctial = [0; 0; 0; 0; 0; L_dot]; % Only L_dot is non-zero for unperturbed motion
    
        % Recover COE angles from modified equinoctial state
        RAAN = atan2d(k, h);                % RAAN [deg]
        i = 2 * atan(sqrt(h^2 + k^2));     % Inclination [deg]
        omega_plus_nu = rad2deg(L) - RAAN;  % [deg]       
                   
        % Compute the position vector in ECI coordinates with new DCM313
        r = p / q;                     % (Walker, 1985. pg4) 
     
        % Calculate the J2 perturbation in the RTN frame (rotating)
        % components as per document shared in Brightspace - directly in
        % RTN frame
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
        r_vec_eci = libCall.vec_rot_to_cart(DCM313,r_vec_rot)';
    
        % Calculate SRP cannonball perturbation in ECI
        aSRP_vec_eci = calc_SRP_accel_cartesian(t, r_vec_eci);
    
        % Convert SRP Cannoball acceleration from ECI back to RTN
        % (rotating)
        aSRP_vec_rtn = vec_cart_to_rot(obj,DCM313inv, aSRP_vec_eci);    
    
        % Define the B matrix from the image provided
        B = sqrt(p/mu) * ...
            [0,        2*p/q,                    0;
             sin(L),  ((q+1)*cos(L)+f)/q,     -g*(h*sin(L)-k*cos(L))/q;
             -cos(L), ((q+1)*sin(L)+g)/q,     f*(h*sin(L)-k*cos(L))/q;
             0,        0,                       s_sq*cos(L)/(2*q);
             0,        0,                       s_sq*sin(L)/(2*q);
             0,        0,                       (h*sin(L) - k*cos(L))/q];
    
        % Compute the perturbed rate of change of the equinoctial elements
        x_dot = f0_equinoctial + B * (aj2_vec_rtn + aSRP_vec_rtn);
        x_dot = f0_equinoctial + B * (aj2_vec_rtn);
        
    end     
    
    function [x_dot,r_vec_eci] = dynamics_2BP_keplerian_J2(t,X)
    
        libCall =  KeplerianOrbitalMechanicsLibrary();
        
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
        
        % Compute true anomaly (ν) from Eccentric Anomaly (E) and the
        % radius
        nu = libCall.ta_eE(e,E);  
        r = libCall.r_aeta(a,e,nu);  
          
        % Calculate New DCM313 Matrix - RTN (rotating) to ECI
        DCM313 =  libCall.OrbitalElementsToDCM(RAAN,i,omega+nu);
        DCM313inv = inv(DCM313);
    
        % Compute the position vector in ECI coordinates with new DCM313
        r_vec_rot = [r; 0; 0];  % [km]
        r_vec_eci = libCall.vec_rot_to_cart(DCM313,r_vec_rot)';
        
        % Calculate the J2 perturbation acceleration in ECI coordinates
        aj2_vec_eci = calc_J2_accel_cartesian(r_vec_eci);
    
        % Convert J2 acceleration from ECI back to RTN (rotating)
        aj2_vec_rot = libCall.vec_cart_to_rot(DCM313inv, aj2_vec_eci);
    
        % Pre-compute common terms for B(X) matrix
        sin_nu = sin(nu);
        cos_nu = cos(nu);
        sin_i = sin(i);
        tan_i = tan(i);
        
        % B matrix construction
        B = (1/h) * ...
            [2*a^2*e*sin_nu,               2*a^2*p/r,                   0;
             p*sin_nu,                     (p+r)*cos_nu + r*e,          0;
             0,                            0,                           r*cos(nu + omega);
             0,                            0,                           r*sin(nu + omega)/sin_i;
             -p*cos_nu/e,                  (p+r)*sin_nu/e,              -r*sin(nu + omega)/tan_i;
             b*p*cos_nu/(a*e) - 2*b*r/a,  -b*(p+r)*sin_nu/(a*e),                        0];
    
       % Compute the perturbed rate of change of the Keplerian elements
        x_dot = f0_coe + B * aj2_vec_rot';
    end
    
    function [x_dot,r_vec_eci] = dynamics_2BP_keplerian_J2_SRP(t,X)

        libCall =  KeplerianOrbitalMechanicsLibrary();

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
        r_vec_eci = libCall.vec_rot_to_cart(DCM313,r_vec_rot)';

        % Calculate the J2 perturbation acceleration in ECI coordinates
        aj2_vec_eci = calc_J2_accel_cartesian(r_vec_eci);

        % Convert J2 acceleration from ECI back to RTN (rotating)
        aj2_vec_rot = libCall.vec_cart_to_rot(DCM313inv, aj2_vec_eci);

        % Calculate SRP cannonball perturbation in cartesian
        aSRP_vec_eci = calc_SRP_accel_cartesian(t, r_vec_eci);

        % Convert SRP Cannoball acceleration from ECI back to RTN (rotating)
        aSRP_vec_rot = libCall.vec_cart_to_rot(DCM313inv, aSRP_vec_eci);

        % Pre-compute common terms for B(X) matrix
        sin_nu = sin(nu);
        cos_nu = cos(nu);
        sin_i = sin(i);
        tan_i = tan(i);

        % B matrix construction
        B = (1/h) * ...
        [2*a^2*e*sin_nu,               2*a^2*p/r,                   0;
        p*sin_nu,                     (p+r)*cos_nu + r*e,          0;
        0,                            0,                           r*cos(nu + omega);
        0,                            0,                           r*sin(nu + omega)/sin_i;
        -p*cos_nu/e,                  (p+r)*sin_nu/e,              -r*sin(nu + omega)/tan_i;
        b*p*cos_nu/(a*e) - 2*b*r/a,  -b*(p+r)*sin_nu/(a*e),                        0];

        % Compute the perturbed rate of change of the Keplerian elements
        x_dot = f0_coe + B * (aj2_vec_rot' + aSRP_vec_rot');
    end

%% ====== 2BP Perturbations Calculation ECI ======
    
    % --- J2 Perturbation Acceleration in Cartesian Coordinates ---
    function [aj2_vec_eci] = calc_J2_accel_cartesian(r_vec_eci)
        % Inputs:
        %   r_vec_eci: Position vector in ECI coordinates [x, y, z] (km)
        % Outputs:
        %   aj2_vec_eci: J2 perturbation acceleration vector in ECI
        %   coordinates [aj2_x, aj2_y, aj2_z] (km/s^2)
        % Calculates the J2 perturbation acceleration based on the
        % spacecraft's position.
    
        % Constants
        mu = 398600.0;  % Earth's gravitational parameter (km^3/s^2)
        J2 = 0.0010826; % J2 coefficient
        ro = 6378.1;    % Earth's equatorial radius (km)
    
        % Extract position components from the state vector
        x = r_vec_eci(1);  % Pos X (km)
        y = r_vec_eci(2);  % Pos Y (km)
        z = r_vec_eci(3);  % Pos Z (km)
    
        % Calculate the distance from the center of the Earth
        r = sqrt(x^2 + y^2 + z^2);
    
        % Calculate the J2 perturbation acceleration components
        z2_r2 = (z^2) / (r^2);  % Ratio of z^2 to r^2
        factor = -1.5 * J2 * mu * (ro^2) / (r^5);
    
        aj2_x = factor * x * (1 - 5 * z2_r2);
        aj2_y = factor * y * (1 - 5 * z2_r2);
        aj2_z = factor * z * (3 - 5 * z2_r2);
    
        % J2 perturbation acceleration vector in ECI
        aj2_vec_eci = [aj2_x; aj2_y; aj2_z];
    end
    
    % --- Solar Radiation Pressure (SRP) Acceleration in Cartesian
    % Coordinates ---
    function [aSRP_vec_eci] = calc_SRP_accel_cartesian(t, r_vec_eci)
        % Inputs:
        %   t: Time (seconds) r_vec_eci: Position vector in ECI coordinates
        %   [x, y, z] (km)
        % Outputs:
        %   aSRP_vec_eci: SRP perturbation acceleration vector in ECI
        %   coordinates [aSRP_x, aSRP_y, aSRP_z] (km/s^2)
        % Calculates the Solar Radiation Pressure (SRP) perturbation
        % acceleration.
    
        % Constants
        A_m_ratio = 5.4e-6;       % Area-to-mass ratio of the spacecraft (km^2/kg)
        a = 149597898;            % Average Earth-Sun distance (km) (1 AU)
        mu = 132712440017.99;     % Sun's gravitational constant (km^3/s^2)
        n = sqrt(mu / a^3);       % Mean motion of the Earth around the Sun (rad/s)
        r_earth_sun = a;          % Earth-Sun distance (km)
        G0 = 1.02e14;             % Solar flux constant (kg*km/s^2)
    
        % Compute the angle theta representing Earth's position in its
        % orbit
        theta = n * t;  % Angle in radians (Earth's position around the Sun)
    
        % Unit vectors for the ECI coordinate system (assuming the Sun lies
        % on the x-axis initially)
        x_hat = [1; 0; 0];
        y_hat = [0; 1; 0];
    
        % Compute the spacecraft's position relative to the Sun
        sun_vec = r_earth_sun * (cos(theta) * x_hat - sin(theta) * y_hat);  % Sun's position in ECI
    
        % Calculate the SRP perturbation acceleration
        aSRP_num = A_m_ratio * G0 * (r_vec_eci + sun_vec);  % Numerator
        aSRP_den = norm(r_vec_eci + sun_vec)^3;             % Denominator (distance cubed)
    
        % SRP perturbation acceleration vector in ECI
        aSRP_vec_eci = aSRP_num / aSRP_den;
    end



%% ====== Orbital Mechanics 3BP Propagators ======

        
            function dxdt = CR3BP(X, mu, n)
                % CR3BP (Circular Restricted Three Body Problem) Dynamics
                % Function Inputs:
                %   X - State vector [x, y, z, vx, vy, vz] mu - Mass
                %   parameter (ratio of secondary mass to total system
                %   mass) n - Mean motion (average angular velocity)
                % Outputs:
                %   dxdt - Derivative of the state vector (rate of change)
                
                % Unpack the state vector into position and velocity
                % components x, y, z - Position coordinates vx, vy, vz -
                % Velocity components
                
                % Calculate distances 'd' and 'r' for gravitational effect
                % calculations 'd' - Distance to the secondary body 'r' -
                % Distance to the primary body
                
                % Compute the derivatives of position (velocity components)
                % dxdt, dydt, dzdt - Derivatives of x, y, z (same as vx,
                % vy, vz)
                
                % Compute the derivatives of velocity using CR3BP equations
                % dvxdt, dvydt, dvzdt - Accelerations in x, y, z directions
                
                % Unpack the state vector
                x = X(1);
                y = X(2); % Correct variable name
                z = X(3);
                vx = X(4);
                vy = X(5);
                vz = X(6);
            
                % CR3BP Equations
                d = sqrt((x + mu)^2 + y^2 + z^2); % Corrected term
                r = sqrt((x - 1 + mu)^2 + y^2 + z^2); % Corrected term
            
                % CR3BP Differential Equations
                dxdt = vx;
                dydt = vy;
                dzdt = vz;
                dvxdt = 2*n*vy + n^2*x - (1 - mu)*(x + mu)/d^3 - mu*(x - 1 + mu)/r^3;
                dvydt = -2*n*vx + n^2*y - (1 - mu)*y/d^3 - mu*y/r^3; % Corrected term
                dvzdt = -((1 - mu)*z/d^3) - mu*z/r^3;
            
                % Return the rate of change
                dxdt = [dxdt; dydt; dzdt; dvxdt; dvydt; dvzdt];
            end


            %% ====== State Converter Functions ======
                     


            % ================== State–converters =========================

            function coe = cart2kep(Xcart, mu)
                % cart2kep  Classical elements from Cartesian state (elliptic only).
                % Uses h-hat 3–1–3 identities and your dualTrigInverse() to get RAAN.
                % Inputs:
                %   Xcart : [r; v] (6x1 or 1x6) in one inertial frame
                %   mu    : gravitational parameter
                % Output:
                %   coe   : struct with a,e,i,Omega,omega,f,M,n,p,h and vectors

                Xcart = Xcart(:);
                if numel(Xcart) < 6
                    error('kep_from_state: Xcart must have 6 elements [r; v].');
                end
                r = Xcart(1:3);  v = Xcart(4:6);

                % Magnitudes
                rnorm = norm(r);  vnorm = norm(v);

                % Angular momentum and its unit vector
                hvec = cross(r, v);
                h    = norm(hvec);
                hhat = hvec / h;

                % Eccentricity vector & scalar
                evec = ( (vnorm^2 - mu/rnorm)*r - dot(r,v)*v )/mu;
                e    = norm(evec);

                % Energy -> a, and semilatus rectum p (elliptic assumption)
                epsE = vnorm^2/2 - mu/rnorm;
                a    = -mu/(2*epsE);
                p    = h^2/mu;

                % Inclination (0 ≤ i ≤ π)
                cz = max(-1, min(1, hhat(3)));
                i  = acos(cz);
                si = sin(i);       % = sin(i)

                % ==================== RAAN Ω from h-hat ====================
                % ĥx =  sinΩ sin i,   ĥy = -cosΩ sin i

                % All candidates in degrees from your helper
                candS = dualTrigInverse('sin', hhat(1)/si);   % [-360,360] deg
                candC = dualTrigInverse('cos', -hhat(2)/si);   % [-360,360] deg

                % Pick the common angle (equal modulo 360 within tolerance)
                Omega_deg = pickCommonAngleDeg(candS, candC, 1e-6);
                Omega = mod(deg2rad(Omega_deg), 2*pi);

                % ===========================================================

                % Argument of periapsis (ω): varpi - Ω with varpi = atan2(e_y, e_x)
                varpi = atan2(evec(2), evec(1));
                omega = mod(varpi - Omega, 2*pi);

                % True anomaly (f)
                if e > 1e-14
                    x = dot(evec, r) / max(e*rnorm, eps);
                    x = max(-1, min(1, x));
                    y = dot(cross(evec, r), hvec) / max(h*e*rnorm, eps);
                    f = atan2(y, x);
                else
                    f = atan2(r(2), r(1));
                end
                f = mod(f, 2*pi);

                % Mean anomaly (M) and mean motion (n) — elliptic only
                cosE = (1 - rnorm/a) / max(e, eps); cosE = max(-1, min(1, cosE));
                sinE = (dot(r, v) / sqrt(mu*a)) / max(e, eps);
                E    = atan2(sinE, cosE);
                M    = mod(E - e*sin(E), 2*pi);
                n    = sqrt(mu/a^3);

                % Pack
                coe = struct('a',a,'e',e,'i',i,'Omega',Omega,'omega',omega, ...
                    'f',f,'M',M,'n',n,'p',p,'h',h, ...
                    'r',rnorm,'v',vnorm, ...
                    'evec',evec,'hvec',hvec,'hhat',hhat);

                function ang = pickCommonAngleDeg(A, B, tolDeg)
                    ang = NaN;
                    A = A(:).'; B = B(:).';
                    for a = A
                        % difference wrapped to [-180,180]
                        d = mod(a - B + 180, 360) - 180;
                        if any(abs(d) <= tolDeg)
                            ang = a; return;
                        end
                    end
                end

                % ---------- small helper (uses modulo-360 equality) ----------
                function angles = dualTrigInverse(trigFunc, value)
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
            end


            function X_cart = coe_to_cartesian(obj, X_coe, mu, varargin)
                % COE_TO_CARTESIAN  Convert Keplerian elements to Cartesian state vectors.
                % X_coe: [a e i RAAN omega M_or_nu], angles in rad. If 'useTA'==true, last col is nu.

                % ---- parse ----
                p = inputParser;  p.CaseSensitive = false;
                addRequired (p,'X_coe',@(x) isnumeric(x) && (size(x,2)==6 || isvector(x)));
                addRequired (p,'mu',    @isnumeric);
                addParameter(p,'useTA',false,@islogical);
                parse(p,X_coe,mu,varargin{:});
                useTA = p.Results.useTA;

                if isvector(X_coe), X_coe = reshape(X_coe,1,[]); end
                [N,~] = size(X_coe);
                X_cart = zeros(N,6);

                for k = 1:N
                    % unpack
                    a     = X_coe(k,1);
                    e     = X_coe(k,2);
                    inc   = X_coe(k,3);
                    RAAN  = X_coe(k,4);
                    omega = X_coe(k,5);

                    % anomalies & radius
                    if useTA
                        nu = X_coe(k,6);
                        r_mag = obj.r_aeta(a,e,nu);
                    else
                        M  = X_coe(k,6);
                        E  = obj.E_Me(M,e);          % your solver
                        nu = obj.ta_eE(e,E);
                        r_mag = obj.r_aeta(a,e,nu);
                    end

                    % perifocal vectors (column form)
                    p_slr = a*(1 - e^2);
                    r_pqw = [ r_mag*cos(nu);  r_mag*sin(nu);  0 ];
                    v_pqw = sqrt(mu/p_slr)*[ -sin(nu);  e+cos(nu);  0 ];

                    % Standard PQW->IJK direction cosine matrix:
                    % C_I_P = R3(RAAN) * R1(inc) * R3(omega)
                    cO = cos(RAAN);  sO = sin(RAAN);
                    ci = cos(inc);   si = sin(inc);
                    cw = cos(omega); sw = sin(omega);
                    C_I_P = [ ...
                        cO*cw - sO*sw*ci,   -cO*sw - sO*cw*ci,   sO*si; ...
                        sO*cw + cO*sw*ci,   -sO*sw + cO*cw*ci,  -cO*si; ...
                        sw*si,               cw*si,               ci    ];

                    % transform to inertial (column), then store as row
                    r_ijk = C_I_P * r_pqw;
                    v_ijk = C_I_P * v_pqw;

                    X_cart(k,:) = [r_ijk.'  v_ijk.'];
                end
            end


            function [r_vec_eci_matrix, COE] = convertEquinoctialToECI(obj, X_eqn)
                    
                % CONVERT EQUINOCTIAL TO ECI Convert Modified Equinoctial
                % orbital elements to ECI coordinates.
                %
                % Usage:
                %   [X_pos_equinoctial, COE_Equinoctial] =
                %   convertEquinoctialToECI(T_eqn, X_eqn, calc)
                %
                % Where:
                %   T_eqn - Time array for Equinoctial elements. X_eqn -
                %   Matrix of Equinoctial elements at each time step.
                %   obj - Instance of AAE590Functions class with
                %   necessary orbital mechanics functions.
                %
                % Returns:
                %   X_pos_equinoctial - Matrix of positions in ECI
                %   coordinates. COE_Equinoctial - Matrix of Classical
                %   Orbital Elements [a, e, i, RAAN].
                
                % Initialize the matrix to store position in ECI
                % coordinates and COE
                r_vec_eci_matrix = zeros(length(X_eqn), 3);
                COE = zeros(length(X_eqn), 5);  % [a, e, i, RAAN, omega + nu]
            
                % Loop through each time step to convert Equinoctial
                % elements to ECI coordinates
                for idx = 1:length(X_eqn)
                    % Extract the Equinoctial elements
                    p = X_eqn(idx,1);
                    f = X_eqn(idx,2);
                    g = X_eqn(idx,3);
                    h = X_eqn(idx,4);
                    k = X_eqn(idx,5);
                    L = X_eqn(idx,6);
            
                    % Compute classical orbital elements from modified
                    % equinoctial elements
                    q = 1 + f*cos(L) + g*sin(L);
                    RAAN_ith = atan2(k, h);             % Right Ascension of the Ascending Node in radians
                    i_ith = 2 * atan(sqrt(h^2 + k^2));  % Inclination in radians
                    e = sqrt(f^2 + g^2);                % Eccentricity
                    a = p / (1 - e^2);                  % Semi-major axis assuming elliptical orbit
                    omega_plus_nu = L - RAAN_ith;       % Argument of latitude (arg of peri + true anomaly) aka true longitude
            
                    % Calculate the Direction Cosine Matrix (DCM) for RTN
                    % (rotating) to ECI frame
                    DCM313 = obj.OrbitalElementsToDCM(rad2deg(RAAN_ith), rad2deg(i_ith), rad2deg(L - RAAN_ith)); 
            
                    % Compute the position vector in ECI coordinates
                    r = p / q;
                    r_vec_rot = [r; 0; 0];  % Position vector in the orbital plane
                    r_vec_eci = obj.vec_rot_to_cart(DCM313, r_vec_rot); % Convert to ECI
                    
                    % Store the computed ECI coordinates and classical
                    % orbital elements
                    r_vec_eci_matrix(idx,:) = r_vec_eci;
                    COE(idx, :) = [a, e, rad2deg(i_ith), rad2deg(RAAN_ith),rad2deg(omega_plus_nu)];
                end
            end
            
            function [r_vec_eci_matrix, COE] = convertMilankovitchToECI(X, mu, libCall)
               
              % This function converts a matrix of state vectors in
              % Milankovitch elements
                % to Earth-Centered Inertial (ECI) position vectors and
                % corresponding Common Orbital Elements (COE).
                %
                % Parameters:
                %   X        - A matrix of state vectors, each row is a
                %   state vector
                %              in Milankovitch elements (nx7 matrix where n
                %              is the number of state vectors).
                %   mu       - The stanard gravitational parameter (mu) of
                %   the central body. libCall  - An object that provides
                %   the function `OrbitalElementsToDCM`
                %              which computes the Direction Cosine Matrix
                %              (DCM) from orbital elements.
                %
                % Returns:
                %   r_vec_eci_matrix - A matrix where each row is an ECI
                %   position vector
                %                      corresponding to a state vector from
                %                      `X`.
                %   COE_matrix       - A matrix where each row contains the
                %   COEs
                %                      [a, e, i, RAAN, omega] for each
                %                      state vector.
            
                % Initialize matrices to store results
                num_states = size(X,1);
                r_vec_eci_matrix = zeros(num_states, 3);
                COE = zeros(num_states, 5);
            
                for idx = 1:num_states
                    % Extract the Milankovitch elements from the current
                    % state vector
                    h_vec_eci = X(idx, 1:3);
                    e_vec_eci = X(idx, 4:6);
                    L = X(idx, 7); % True longitude in radians
            
                    % Compute magnitudes and unit vectors for h and e
                    h_mag = norm(h_vec_eci);
                    e_mag = norm(e_vec_eci);
                    h_hat = h_vec_eci / h_mag;
                    e_hat = e_vec_eci / e_mag;
            
                    % Compute semilatus rectum and semimajor axis
                    p = h_mag^2 / mu;
                    a = p / (1 - e_mag^2);
            
                    % Compute inclination
                    i = acos(h_vec_eci(3) / h_mag);
            
                    % Compute RAAN using the node line vector
                    n_vec = cross([0; 0; 1], h_hat); % Cross product of z_hat and h_hat
                    n_hat = n_vec / norm(n_vec); % Node line unit vector
                    RAAN = acos(n_hat(1)); % RAAN is the angle between n_hat and x_hat
            
                    if n_hat(2) < 0
                        RAAN = 360 - RAAN;
                    end
            
                    % Compute argument of periapsis and true anomaly
                    omega = acos(dot(e_hat, n_hat));
                    nu = rad2deg(L) - RAAN - omega;
            
                    % Compute position vector in rotating frame
                    r = p / (1 + e_mag * cos(nu));
                    r_vec_rot = [r; 0; 0];
            
                    % Compute the DCM for the conversion from rotating
                    % frame to ECI
                    DCM313 = libCall.OrbitalElementsToDCM(RAAN, i, omega + nu);
                    r_vec_eci = DCM313 * r_vec_rot;
            
                    % Store the results in the matrices
                    r_vec_eci_matrix(idx, :) = r_vec_eci';
                    COE(idx, :) = [a, e_mag, i, RAAN, omega];
                end
            end
            
            function x0_equinoctial = keplerianToEquinoctial(a, e, i, RAAN, omega, nu)
                % Convert Keplerian elements to Equinoctial elements
                %
                % Parameters:
                %   a     - Semi-major axis e     - Eccentricity i     -
                %   Inclination (in degrees) RAAN  - Right Ascension of the
                %   Ascending Node (in degrees) omega - Argument of Periapsis
                %   (in degrees) nu    - True Anomaly (in degrees)
                %
                % Returns:
                %   x0_equinoctial - The state vector in Equinoctial elements
                %   [p, f, g, h, k, L]
                
                % Compute the semilatus rectum
                p = a * (1 - e^2);
                
                % Compute the Equinoctial elements f and g (related to
                % eccentricity)
                f = e * cos(omega + RAAN);
                g = e * sin(omega + RAAN);
                
                % Compute the Equinoctial elements h and k (related to
                % inclination)
                h = tan(i / 2) * cos(RAAN);
                k = tan(i / 2) * sin(RAAN);
                
                % Compute the true longitude (L) in radians for Modified
                % Equinoctial Elements
                L = omega + RAAN + nu;
                
                % Construct the Equinoctial state vector
                x0_equinoctial = [p; f; g; h; k; L];
            end
        



        %% ====== Reference Frame Converter Functions ======

            %==================================================================
            %  ECI → HCI with user-supplied Earth state history
            %==================================================================
            function states_hci = eci2hci_knownEarthICs(obj, states_eci, rE_Sun_km, vE_Sun_kms)
                %ECI2HCI_FROMEARTH  Convert Earth-centred inertial states to Sun-centred
                %                   inertial using *explicit* Earth ephemerides.
                %
                %   states_hci = obj.eci2hci_fromEarth(states_eci, rE_Sun_km, vE_Sun_kms)
                %
                % INPUTS
                %   states_eci  : [N×6]  spacecraft states w.r.t. Earth  [x y z vx vy vz] (km | km/s)
                %   rE_Sun_km   : [N×3] *or* 1×3 Earth position  w.r.t. Sun (HCI) (km)
                %   vE_Sun_kms  : [N×3] *or* 1×3 Earth velocity  w.r.t. Sun (HCI) (km/s)
                %
                % OUTPUT
                %   states_hci  : [N×6]  spacecraft states w.r.t. Sun   (km | km/s)
                %
                % FORMULAE
                %   r_SC/S = r_E/S + r_SC/E
                %   v_SC/S = v_E/S + v_SC/E
                %
                % NOTES
                %   • Passing a single 1×3 vector for rE_Sun_km or vE_Sun_kms applies that
                %     state to every row (broadcast).
                %   • No circular-orbit approximation is used—Earth ephemerides come from
                %     the caller.
                %==================================================================

                % --------- sanity checks -------------------------------------
                N  = size(states_eci,1);
                assert(size(states_eci,2)==6, 'states_eci must be N×6');

                if isrow(rE_Sun_km),  rE_Sun_km  = repmat(rE_Sun_km ,N,1); end
                if isrow(vE_Sun_kms), vE_Sun_kms = repmat(vE_Sun_kms,N,1); end

                assert(all(size(rE_Sun_km)==[N 3]),  'rE_Sun_km must be N×3 or 1×3');
                assert(all(size(vE_Sun_kms)==[N 3]), 'vE_Sun_kms must be N×3 or 1×3');

                % --------- vectorized addition -------------------------------
                rSC_eci = states_eci(:,1:3);
                vSC_eci = states_eci(:,4:6);

                rSC_hci = rE_Sun_km + rSC_eci;
                vSC_hci = vE_Sun_kms + vSC_eci;

                states_hci = [rSC_hci , vSC_hci];
            end


            %==================================================================
            %  ECI → HCI with user-supplied Earth state history
            %==================================================================
            

            function states_hci = eci2hci(obj, states_eci, t)
                %ECI2HCI  Convert Earth‐centered inertial to Sun‐centered inertial
                %
                % states_hci = obj.eci2hci(states_eci, t)
                %
                % Converts a history of spacecraft states from an Earth‐Centered
                % Inertial (ECI) frame into a Sun‐Centered Inertial (HCI) frame,
                % assuming Earth's orbit is circular.
                %
                % Inputs:
                %   states_eci : [N×6] matrix of ECI states [x, y, z, vx, vy, vz]
                %                in km / km·s⁻¹, relative to Earth.
                %   t          : [N×1] or [1×N] vector of times since epoch [s].
                %
                % Output:
                %   states_hci : [N×6] matrix of HCI states [x, y, z, vx, vy, vz]
                %                in km / km·s⁻¹, relative to Sun.
                %
                % Frame relationship (circular Earth orbit):
                %   θ(t)    = n⊙ · t,      n⊙ = 2π / Tyear
                %   r_E/S  = a⊕ · [cosθ, sinθ, 0]
                %   v_E/S  = n⊙·a⊕ · [−sinθ, cosθ, 0]
                %   r_sc/S = r_E/S + r_sc/E
                %   v_sc/S = v_E/S + v_sc/E

                % ensure t is column, matching rows
                t = t(:);
                N = size(states_eci,1);
                assert(numel(t)==N, 'Time vector length must equal rows of states_eci');

                % pull Earth's orbital parameters from CelestialBody
                a_earth   = obj.bodyEarth.orbit.a_km;       % semi-major axis [km]
                Tyear     = obj.bodyEarth.orbit.period_sec; % sidereal year [s]
                n_sun     = 2*pi / Tyear;                   % mean motion [rad/s]

                % compute Earth→Sun rotation
                theta = n_sun * t;              % [N×1]
                c     = cos(theta);             % [N×1]
                s     = sin(theta);             % [N×1]

                % Earth's position & velocity in Sun‐Centered frame
                r_ES = a_earth * [c,    s,    zeros(N,1)];        % [N×3]
                v_ES = n_sun*a_earth * [-s,   c,    zeros(N,1)];  % [N×3]

                % split input ECI
                rSC_E = states_eci(:,1:3);
                vSC_E = states_eci(:,4:6);

                % convert to HCI
                rSC_hci = r_ES + rSC_E;
                vSC_hci = v_ES + vSC_E;

                states_hci = [rSC_hci, vSC_hci];
            end


            %==================================================================
            %  HCI → ECI with user-supplied Earth state history
            %==================================================================

            function states_eci = hci2eci_knownEarthICs(obj, states_hci, rE_Sun_km, vE_Sun_kms)
                %HCI2ECI_FROMEARTH  Convert Sun-centred inertial states to Earth-centred
                %                   inertial using *explicit* Earth ephemerides.
                %
                %  states_eci = obj.hci2eci_fromEarth(states_hci, rE_Sun_km, vE_Sun_kms)
                %
                % INPUTS
                %   states_hci  : [N×6]  spacecraft states w.r.t. the Sun  [x y z vx vy vz] (km | km/s)
                %   rE_Sun_km   : [N×3] (or 1×3) Earth position w.r.t. Sun          (km)
                %   vE_Sun_kms  : [N×3] (or 1×3) Earth velocity w.r.t. Sun          (km/s)
                %
                % OUTPUT
                %   states_eci  : [N×6]  spacecraft states w.r.t. the Earth (ECI)   (km | km/s)
                %
                % FORMULAE
                %   r_SC/E = r_SC/S − r_E/S
                %   v_SC/E = v_SC/S − v_E/S
                %
                % NOTES
                %   • If you pass a single 1×3 vector for rE_Sun_km or vE_Sun_kms it is
                %     automatically expanded (assumes the same Earth state for every row).
                %   • No assumptions are made about Earth's orbit; the caller controls the
                %     fidelity via the supplied ephemerides.
                %==================================================================

                % --------- basic size checks ---------------------------------
                N  = size(states_hci,1);
                assert(size(states_hci,2)==6, 'states_hci must be N×6');

                if isrow(rE_Sun_km),  rE_Sun_km  = repmat(rE_Sun_km ,N,1); end
                if isrow(vE_Sun_kms), vE_Sun_kms = repmat(vE_Sun_kms,N,1); end

                assert(all(size(rE_Sun_km)==[N 3]),  'rE_Sun_km must be N×3 or 1×3');
                assert(all(size(vE_Sun_kms)==[N 3]), 'vE_Sun_kms must be N×3 or 1×3');

                % --------- vectorized subtraction ----------------------------
                rSC_hci = states_hci(:,1:3);
                vSC_hci = states_hci(:,4:6);

                rSC_eci = rSC_hci - rE_Sun_km;
                vSC_eci = vSC_hci - vE_Sun_kms;

                states_eci = [rSC_eci , vSC_eci];
            end


            function states_eci = hci2eci(obj, states_hci, t)
                %HCI2ECI  Convert Sun‐centered inertial back to Earth‐centered inertial
                %
                % states_eci = obj.hci2eci(states_hci, t)
                %
                % Inverse of eci2hci: maps HCI states back into ECI by subtracting
                % Earth's orbital motion, again assuming a circular orbit.
                %
                % Inputs:
                %   states_hci : [N×6] matrix [x, y, z, vx, vy, vz] in km / km·s⁻¹
                %                relative to Sun.
                %   t          : [N×1] or [1×N] vector of times since epoch [s].
                %
                % Output:
                %   states_eci : [N×6] matrix [x, y, z, vx, vy, vz] in km / km·s⁻¹
                %                relative to Earth.
                %
                % Inverse relationships:
                %   r_sc/E = r_sc/S − r_E/S
                %   v_sc/E = v_sc/S − v_E/S

                t = t(:);
                N = size(states_hci,1);
                assert(numel(t)==N, 'Time vector length must equal rows of states_hci');

                % Earth's orbit parameters
                a_earth = obj.bodyEarth.orbit.a_km;
                Tyear   = obj.bodyEarth.orbit.period_sec;
                n_sun   = 2*pi / Tyear;

                theta = n_sun * t;    c = cos(theta);   s = sin(theta);
                r_ES = a_earth * [c,    s,    zeros(N,1)];
                v_ES = n_sun*a_earth * [-s,   c,    zeros(N,1)];

                rSC_hci = states_hci(:,1:3);
                vSC_hci = states_hci(:,4:6);

                rSC_eci = rSC_hci - r_ES;
                vSC_eci = vSC_hci - v_ES;

                states_eci = [rSC_eci, vSC_eci];
            end

          
            %% ====== Direction Cosine Matrix from Euler Angles ======

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


%% ====== Conversions And Other Functions ======

            function [ang2] = sine_law_angle (obj, ang1,side1,side2)
                ang2 = [asin((side2*sin(ang1))/side1),180-asin((side2*sin(ang1))/side1)];
            end 

            function [side3] = cos_law_side(obj, ang1_2,side1,side2)
                side3 = sqrt(side1^2 + side2^2 - 2*side1*side2*cos(ang1_2)); 
            end 

            function [ang_1_2] = cos_law_angle_s1s2(side1,side2,side3)
                ang_1_2 = acos((side1^2+side2^2-side3^2)/(2*side1*side2)); 
            end 
            
            function t = secToHours(obj,n)
                % Convert seconds to hours (supports vector input)
                t = n ./ 3600;
            end

            function t = secToDays(obj,n)
                % Convert seconds to days (supports vector input)
                t = n ./ 86400;
            end

            function t = secToWeeks(obj,n)
                % Convert seconds to weeks (supports vector input)
                t = n ./ (86400 * 7);
            end

            function t = secToMonths(obj,n)
                % Convert seconds to months (approx., supports vector input)
                t = n ./ (86400 * 30);
            end

            function t = secToYears(obj,n)
                % Convert seconds to years (approx., supports vector input)
                t = n ./ (86400 * 365);
            end

            function km = au_to_km(obj,distance)
                % Convert AU to km (supports vector input)
                km = distance .* 149597870.7;
            end

            function au = km_to_au(obj,distance)
                % Convert km to AU (supports vector input)
                au = distance ./ 149597870.7;
            end


   end 
end 
