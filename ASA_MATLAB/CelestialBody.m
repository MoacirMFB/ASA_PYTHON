classdef CelestialBody
    % CELESTIALBODY  Physical, gravitational, and orbital constants for a body.
    %
    %   Encapsulates canonical parameters (GM, radius, J2, spin rate, SOI) and
    %   major‐orbit elements, with convenient accessors in km, m, and AU.
    %
    % FEATURES
    %   • Preloaded bodies: 'Earth', 'Moon', 'Sun'
    %   • Gravitational parameter μ (km^3/s^2) with .km, .m, .AU getters
    %   • Mean radius with .km, .m, .AU getters
    %   • J2 and rotational rate ω (rad/s)
    %   • Sphere of influence (SOI) with .km, .m, .AU, .Re (radii) getters
    %   • Orbital elements struct (aphelion, perihelion, a, e, i, Ω, M0,
    %     sidereal period, mean orbital speed) plus convenience conversions
    %
    % DEPENDENT PROPERTIES
    %   name     – Display name (e.g., 'Earth')
    %   mu       – struct with fields .km, .m, .AU
    %   radius   – struct with fields .km, .m, .AU
    %   J2       – scalar (unitless)
    %   omega    – scalar (rad/s)
    %   soi      – struct with fields .km, .m, .AU, .Re
    %   orbit    – struct of orbit elements + derived fields
    %
    % UNITS
    %   • Length: km (stored), m and AU via getters
    %   • Time:   s
    %   • Angle:  degrees for i, Ω, M0 (stored), radians for ω
    %
    % EXAMPLES
    %   E = CelestialBody('Earth');
    %   E.mu.km         % → GM in km^3/s^2
    %   E.radius.AU     % → mean radius in AU
    %   E.soi.Re        % → SOI in Earth radii
    %   E.orbit.a_km    % → semi‐major axis [km]
    %
    % NOTES
    %   - AU constant is 1 AU = 149,597,870.7 km (internal reference).
    %   - Orbital elements are representative (J2000‐style where noted) and
    %     intended for analysis/visualization, not precise ephemerides.
    %   - SOI values are approximate and scenario‐dependent; use with care.
    %
    % AUTHOR
    %   Moacir Fonseca Becker
    %   Purdue University
    %
    % LAST MODIFIED
    %   07/02/2025


    properties (Constant, Access = private)
        AU_KM = 149597870.7;  % 1 AU in km
    end

    properties (Access = private)
        mu_km          % gravitational parameter in [km^3/s^2]
        rad_km         % mean radius in [km]
        J2_            % 2nd zonal harmonic coefficient (private storage)
        omega_         % Rotational angular velocity [rad/s] (private storage)
        name_
        soi_km         % sphere-of-influence radius [km]

        %--- Orbital elements (specific) ---
        aphelion_km    % Aphelion distance [km]
        perihelion_km  % Perihelion distance [km]
        a_km           % Semi-major axis [km]
        e_             % Eccentricity (unitless)
        i_deg          % Inclination [deg]
        RAAN_deg       % Longitude of ascending node [deg]
        M0_deg         % Mean anomaly at epoch [deg]
        period_days    % Orbital period (sidereal) [days]
        speed_km_s     % Average orbital speed [km/s]
       


    end

    properties (Dependent)
        name
        mu             % struct with fields .km, .m, .AU
        radius         % struct with fields .km, .m, .AU
        J2             % public-facing
        omega          % public-facing
        orbit          % struct of orbital elements
        soi            % struct with fields .km, .m, .AU, .Re
    end

    methods
        %% Constructor
        function obj = CelestialBody(bodyName)
            switch lower(bodyName)
                case 'earth'
                    obj.name_           = 'Earth';
                    obj.mu_km           = 398600.4418;
                    obj.rad_km          = 6378.1363;                    
                    obj.soi_km          = 145 * obj.rad_km;
                    obj.J2_             = 1.08262668e-3;
                    obj.omega_          = 7.2921159e-5;
                    % Earth orbital elements (J2000 ecliptic)
                    obj.aphelion_km     = 152097597;
                    obj.perihelion_km   = 147098450;
                    obj.a_km            = 149598023;
                    obj.e_              = 0.0167086;
                    obj.i_deg           = 0.00005;
                    obj.RAAN_deg        = -11.26064;
                    obj.M0_deg          = 358.617;
                    obj.period_days     = 365.256363004;
                    obj.speed_km_s      = 29.7827;

                case 'moon'
                    obj.name_   = 'Moon';
                    obj.mu_km   = 4902.800066;        % [km^3/s^2]
                    obj.rad_km  = 1737.4;             % mean radius
                    obj.J2_     = 2.03263e-4;         % oblateness
                    obj.omega_  = 2.6617e-6;          % rot. rate  [rad /s]

                    % Moon orbital elements about the Earth (J2000 ecliptic)
                    obj.aphelion_km   = 405503;       % apogee distance
                    obj.perihelion_km = 363300;       % perigee distance
                    obj.a_km          = 384400;       % semimajor axis
                    obj.e_            = 0.0549;       % eccentricity
                    obj.i_deg         = 5.145;        % incl. to ecliptic
                    obj.RAAN_deg      = 125.08;       % Ω  @ J2000
                    obj.M0_deg        = 115.3654;     % mean anomaly @ J2000
                    obj.period_days   = 27.321661;    % sidereal period
                    obj.speed_km_s    = 1.022;        % mean orbital speed


                case 'sun'
                    obj.name_   = 'Sun';
                    obj.mu_km   = 1.32712440041279419e11;
                    obj.rad_km  = 696340;
                    obj.J2_     = NaN;
                    obj.omega_  = 2.86533e-6;

                otherwise
                    error('Unsupported celestial body: %s', bodyName)
            end
        end

        %% name getter
        function v = get.name(obj)
            v = obj.name_;
        end

        %% mu getter
        function mu = get.mu(obj)
            mu.km = obj.mu_km;
            mu.m  = obj.mu_km * 1e9;                   % 1 km^3 = 1e9 m^3
            mu.AU = obj.mu_km / (obj.AU_KM^3);
        end

        %% radius getter
        function r = get.radius(obj)
            r.km = obj.rad_km;
            r.m  = obj.rad_km * 1e3;
            r.AU = obj.rad_km / obj.AU_KM;
        end

        %% J2 getter
        function v = get.J2(obj)
            v = obj.J2_;
        end

        %% omega getter
        function w = get.omega(obj)
            w = obj.omega_;
        end

        %% soi getter 
        function s = get.soi(obj)
            s.km = obj.soi_km;                         % km
            s.m  = obj.soi_km * 1e3;                   % m
            s.AU = obj.soi_km / obj.AU_KM;             % AU
            s.Re = obj.soi_km / obj.rad_km;            % Earth radii
        end

        %% orbit elements getter
        function o = get.orbit(obj)
            o.aphelion_km   = obj.aphelion_km;
            o.perihelion_km = obj.perihelion_km;
            o.a_km          = obj.a_km;
            o.e             = obj.e_;
            o.i_deg         = obj.i_deg;
            o.RAAN_deg      = obj.RAAN_deg;
            o.M0_deg        = obj.M0_deg;
            o.period_days   = obj.period_days;
            o.speed_km_s    = obj.speed_km_s;
            % convenience conversions:
            o.aphelion_AU   = obj.aphelion_km / obj.AU_KM;
            o.perihelion_AU = obj.perihelion_km / obj.AU_KM;
            o.a_AU          = obj.a_km / obj.AU_KM;
            o.period_sec    = obj.period_days * 86400;
        end
    end
end
