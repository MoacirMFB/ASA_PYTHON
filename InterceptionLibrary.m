classdef InterceptionLibrary
    % INTERCEPTIONLIBRARY  Toolbox-style class for MORPHO asteroid-interception studies.
    %
    %   Provides methods to model kinetic impact missions against
    %   potentially hazardous asteroids. Includes:
    %     • Lambert solver wrappers (multi-rev, ΔV-minimizing search)
    %     • Performance grid sweeps over departure/arrival MBIs and SC masses
    %     • Post-intercept trajectory modeling with momentum transfer (β-law)
    %     • Convenience functions for propagation, plotting, and analysis
    %
    % AUTHOR
    %   Moacir Fonseca Becker
    %   Purdue University
    %
    % LAST MODIFIED
    %   10/17/2025
    %
    % NOTES
    %   - Units follow [km, km/s, s] unless otherwise specified.
    %   - Built to support MORPHO (Modular Orbital system for Rapid response
    %     to Potentially Hazardous Objects).
    %   - Requires KeplerianOrbitalMechanicsLibrary for 2BP dynamics.
    %

    %% --------------------------------------------------------------------
    properties (Access = private)
        orb   % KeplerianOrbitalMechanicsLibrary handle (for 2BP dynamics)
    end


    %% --- Constants of this class ---
    properties (Constant, Access = private)
        MU_SUN_KM = 1.32712440041279419e11; % km^3/s^2
        MU_SUN_AU = 3.96401599317504e-14;   % AU^3/s^2
        R_EARTH_KM = 6378.1363;             % km
    end


    methods

        %% --- Constructor and other classes used ---
        function obj = InterceptionLibrary(orbInstance)
            % Constructor: cache a KeplerianOrbitalMechanicsLibrary handle.
            if nargin >= 1 && isa(orbInstance,'KeplerianOrbitalMechanicsLibrary')
                obj.orb = orbInstance;
            else
                obj.orb = KeplerianOrbitalMechanicsLibrary();
            end

        end

        %% --- Environment Preparation ---
        % --- > Propagate Earth Compute MOID/CA, Force Intersection and Compute States MBI ---

        function env = makeEnv(obj, varargin)
            % MAKEENV  Bundles common constants, grids, and ODE options in one struct.
            %   env = obj.makeEnv('years',4,'step_min',10,'odeOpt',odeset(...))
            %
            % Fields in env:
            %   .orb            KeplerianOrbitalMechanicsLibrary
            %   .muSun_km       Sun μ [km^3/s^2]
            %   .muSun_AU       Sun μ [AU^3/s^2]
            %   .AU2km          [km/AU]
            %   .R_E_km         Earth radius [km]
            %   .odeOpt         odeset struct
            %   .tspan          global time grid [s]
            %   .years, .step_min
            %
            % (Filled later)
            %   .t_Earth, .X_Earth_hist

            p = inputParser;
            p.addParameter('years', 4, @(x)isnumeric(x)&&isscalar(x)&&x>0);
            p.addParameter('step_min', 10, @(x)isnumeric(x)&&isscalar(x)&&x>0);
            p.addParameter('odeOpt', odeset('RelTol',1e-13,'AbsTol',1e-13));
            p.parse(varargin{:});
            years    = p.Results.years;
            step_min = p.Results.step_min;
            odeOpt   = p.Results.odeOpt;

            env.orb      = obj.orb;
            env.muSun_km = obj.MU_SUN_KM;
            env.muSun_AU = obj.MU_SUN_AU;
            env.AU2km    = 149597870.7;
            env.R_E_km   = obj.R_EARTH_KM;

            env.odeOpt   = odeOpt;
            env.years    = years;
            env.step_min = step_min;

            tf_s   = years * 365.25 * 86400;
            dt_s   = step_min * 60;
            env.tspan = 0:dt_s:tf_s;
        end

        function [tE, XE] = propagateEarth(obj, env)
            % PROPAGATEEARTH  Propagate Earth's Sun-centered two-body trajectory.
            %
            %   [tE, XE] = propagateEarth(env)
            %
            % INPUT
            %   env   struct with fields:
            %           .tspan   1xN time grid [s]
            %           .AU2km   AU→km scale
            %           .odeOpt  ODE options
            %
            % OUTPUT
            %   tE    [Nx1] time stamps [s]
            %   XE    [Nx6] Earth states [r(km), v(km/s)]
            Earth = CelestialBody("Earth");
            oe = Earth.orbit;  % COE fields in mixed units
            X_coe_earth = [oe.a_AU, oe.e, deg2rad(oe.i_deg), deg2rad(oe.RAAN_deg), deg2rad(102.94719), deg2rad(oe.M0_deg)];
            X_AU = obj.orb.coe_to_cartesian(X_coe_earth, obj.MU_SUN_AU);

            r0_km   = X_AU(1:3) * env.AU2km;
            v0_km_s = X_AU(4:6) * env.AU2km;
            X0      = [r0_km(:); v0_km_s(:)];
            [tE, XE] = ode45(@(t,X) obj.orb.dynamics_2BP_cartesian(t,X,obj.MU_SUN_KM), env.tspan, X0, env.odeOpt);
        end



        function ast = astCatalog(~, names)
            % ASTCATALOG  Minimal name→COE catalog (returns a *cell array* of structs).
            %   ast = astCatalog(names)
            % Inputs
            %   names : char | string | cellstr | string[]
            %           e.g. {'2004 VD17','Apophis'}
            % Output
            %   ast   : cell{K,1}, each ast{k} is a struct:
            %             .name (string) , .coe (1x6 [a_AU e i RAAN ω M], a in AU, angles in rad)

            if ischar(names) || (isstring(names) && isscalar(names)), names = {char(names)}; end
            names = string(names(:));

            db = containers.Map('KeyType','char','ValueType','any');
            db('2007 DX40')   = [1.538448416729292, 0.5382979734185152, deg2rad([0.4519685, 329.8163426, 273.6828503, 0])];
            db('2004 VD17')   = [1.5080000, 0.5887000, deg2rad([4.22,    223.98,   90.99,    0])];
            db('2007 FT3')    = [1.1246634, 0.3057574, deg2rad([26.72735, 9.79108, 277.56582, 0])];
            db('1979 XB')     = [2.2199764, 0.7073090, deg2rad([24.57497, 84.74573, 76.73211, 0])];
            db('1950 DA')     = [1.6986794, 0.5075054, deg2rad([12.15458,356.54705,224.82640,0])];
            db('Bennu')       = [1.1259897, 0.2037311, deg2rad([6.03290, 1.97241, 66.39438, 0])];
            db('Apophis')     = [0.9225521, 0.1912907, deg2rad([3.33974,203.91537,126.68323,0])];
            db('2011 AG5')    = [1.4241515, 0.3882701, deg2rad([3.69422,135.59205,54.05017,0])];
            db('2022 AE1')    = [1.4708823, 0.5461987, deg2rad([6.29686,102.18992,268.32032,0])];
            db('2000 SG344')  = [0.9774614, 0.0669332, deg2rad([0.11213,191.95995,275.30264,0])];
            db('2023 DW')     = [0.8200000, 0.3970000, deg2rad([5.81, 326.10, 40.40, 0])];

            ast = cell(numel(names),1);
            for k = 1:numel(names)
                key = char(names(k));
                if ~isKey(db, key)
                    error('astCatalog:UnknownName','Asteroid "%s" not in catalog.', key);
                end
                ast{k} = struct('name', names(k), 'coe', db(key));
            end
        end

        function ast = propagateAsteroids(obj, ast, env)
            % PROPAGATEASTEROIDS  Fill .t_hist, .X_hist for each asteroid (cell array in, cell out).
            %   ast = obj.propagateAsteroids(ast, env)
            % ast{k} fields added: .t_hist [Nx1], .X_hist [Nx6] (km, km/s)

            for k = 1:numel(ast)
                X_AU   = obj.orb.coe_to_cartesian(ast{k}.coe, obj.MU_SUN_AU);
                r0_km  = X_AU(1:3) * env.AU2km;
                v0_km_s= X_AU(4:6) * env.AU2km;
                X0     = [r0_km(:); v0_km_s(:)];
                [t, X] = ode45(@(t,X) obj.orb.dynamics_2BP_cartesian(t,X,obj.MU_SUN_KM), env.tspan, X0, env.odeOpt);
                ast{k}.t_hist = t;
                ast{k}.X_hist = X;
            end
        end


        function bodies = makeBodiesForPlot(~, ast, env)
            % MAKEBODIESFORPLOT  Package Earth + asteroid trajectories for plotting.
            %
            %   bodies = makeBodiesForPlot(ast, env)
            %
            % INPUTS
            %   ast   cell array of asteroid structs with .name, .X_hist, .t_hist
            %   env   struct with Earth fields .X_Earth_hist, .t_Earth
            %
            % OUTPUT
            %   bodies  cell array {N+1,1} of structs ready for PlotterLibrary:
            %              bodies{1} = Earth, bodies{2..N+1} = asteroids
            bodies = cell(numel(ast)+1,1);
            bodies{1} = struct('name',"Earth", 'X_hist', env.X_Earth_hist, 't_hist', env.t_Earth);
            for k = 1:numel(ast)
                nm = string(ast{k}.name);
                if ~isscalar(nm), nm = strjoin(nm, ""); end   % collapse any pieces
                bodies{k+1} = struct('name', nm, 'X_hist', ast{k}.X_hist, 't_hist', ast{k}.t_hist);
            end
        end


        function [ast, IC_atMBI_Earth_Ast] = prepareMBI(obj, ast, env, monthsBack, forceImpact)
            % prepareMBI  Compute nominal MOID/CA, optionally force impact, then back-prop to MBI.
            %   ast is a *cell* array of asteroid structs; output ast (cell) is updated in place.
            % Outputs:
            %   ast{k}.MOID_pre_km, ast{k}.CA_pre_km (0 if forced), optional name suffix "_forced_"
            %   States_MBI_Earth_Ast{k}(i).stateEarth / .stateAst at monthsBack(i)

            IC_atMBI_Earth_Ast = cell(1, numel(ast));

            for k = 1:numel(ast)
                [~, stateMOID_nom] = obj.get_CA_MOID(env.X_Earth_hist, ast{k}.X_hist, env.t_Earth, ast{k}.t_hist);
                ast{k}.MOID_pre_km = stateMOID_nom.d_km;
                ast{k}.CA_pre_km   = stateMOID_nom.d_km;

                if forceImpact
                    stateMOID_forced = stateMOID_nom;
                    stateMOID_forced.stateAst(1:3) = stateMOID_forced.stateEarth(1:3);
                    ast{k}.MOID_pre_km = 0;
                    ast{k}.CA_pre_km   = 0;
                    ast{k}.name = string(ast{k}.name);
                    ast{k}.name = ast{k}.name + "-forced";
                    stateAtMOID = stateMOID_forced;
                else
                    stateAtMOID = stateMOID_nom;
                end

                IC_atMBI_Earth_Ast{k} = obj.getStatesAtMBI(stateAtMOID, monthsBack);
            end
        end



        %  --- TYPE 1 ANALYSIS: Single Transfer Performance  ---
        function [perf, XE_post, XA_post, XA_nom] = ...
                computeSingleCasePerformance(obj, bodies, cfgPostInt, best, depMBI, arrMBI, asteroids, kAst, ...
                odeOpt, dt_min)
            % SIMPLE version:
            % - bodies = {Earth, Ast(pre), Interceptor} with histories only up to the interception (TOF)
            % - No post body provided; we create the post-impact asteroid IC here
            % - Propagate Earth, Ast(post), and Ast(nominal) on the same long tspan for CA/MOID

            % --------------- constants ---------------
            SAFETY = 2;                       % post window safety factor vs asteroid period
            dt_s   = dt_min * 60;

            % --------------- times (sec) -------------
            TLI_sec =  depMBI           * 30 * 86400;     % launch → Earth impact
            TOF_sec = (depMBI - arrMBI) * 30 * 86400;     % launch → intercept

            tImpact =  arrMBI           * 30 * 86400;   % intercept → (nominal) Earth-impact


            % --------------- basic perf at intercept --
            perf.Vinf = best.dV;

            % XA_nom_int = bodies{2}.X_hist(end,:);
            XA_nom_int = cfgPostInt.Xast_nom_int;
            Xsc_int = bodies{3}.X_hist(end,:);

            rA_nom_int = XA_nom_int(1:3).';   rA_nom_int = rA_nom_int(:);
            rSC_int  = Xsc_int(end,1:3);  rSC_int   = rSC_int(:);

            % --------------- sanity check: SC meets Ast at intercept ---------------

            if norm(rA_nom_int - rSC_int) > 1e-3
                warning('SC and nominal asteroid positions differ at intercept (||Δr|| = %.3g km).', ...
                    norm(rA_nom_int - rSC_int));
            end


            % --------------- Get asteroid state post intercept from bodies -----------------
            IC_Apost = bodies{4}.X_hist(1,:);

            % --------------- Earth and asteroid states at intercept -----------------
            IC_Epost      = bodies{1}.X_hist(end,:).';         % Earth state @ intercept
            vAst_nom_int  = bodies{2}.X_hist(end,4:6).';       % nominal asteroid vel @ intercept [km/s]

            % Relative velocity spacecraft vs asteroid at intercept (for reporting)
            perf.DVint = norm(vAst_nom_int - best.VF);
            perf.theta = real(acosd( dot(best.VF, vAst_nom_int) / (norm(best.VF)*norm(vAst_nom_int)) ));
            perf.R1    = 1 - TOF_sec / TLI_sec;

            % --------------- choose post window (long enough for CA/MOID) -----------
            T_need  = arrMBI * 30 * 86400;                   % intercept → Earth-impact (nominal)
            T_Earth = 365.25 * 86400;                        % 1 yr
            a_AU    = asteroids{kAst}.coe(1);
            T_ast   = 2*pi*sqrt(a_AU^3 / obj.MU_SUN_AU);      % asteroid period [s]
            T_end   = SAFETY * max([T_need, SAFETY*T_ast, T_Earth]);     % cover impact + give some safety room

            tspan_post_perf = 0 : dt_s : T_end;

            % --------------- propagate on common post grid --------------------------

            % Earth from IC_Epost
            [~, XE_post] = ode45(@(t,X) obj.orb.dynamics_2BP_cartesian(t,X,obj.MU_SUN_KM), ...
                tspan_post_perf, IC_Epost, odeOpt);

            % Asteroid (post intercept)
            [~, XA_post] = ode45(@(t,X) obj.orb.dynamics_2BP_cartesian(t,X,obj.MU_SUN_KM), ...
                tspan_post_perf, IC_Apost, odeOpt);

            % Asteroid (nominal/no kick) for comparison
            IC_A_nom = bodies{2}.X_hist(end,:).';
            [~, XA_nom]  = ode45(@(t,X) obj.orb.dynamics_2BP_cartesian(t,X,obj.MU_SUN_KM), ...
                tspan_post_perf, IC_A_nom, odeOpt);

            % --------------- CA & MOID (post) ---------------------------------------
            [CA_post, MOID_post] = obj.get_CA_MOID(XE_post, XA_post, tspan_post_perf, tspan_post_perf);

            baseMOID = asteroids{kAst}.MOID_pre_km;
            baseCA   = asteroids{kAst}.CA_pre_km;

            perf.DeltaMOID_RE = (MOID_post.d_km - baseMOID) / obj.R_EARTH_KM;
            perf.DeltaCA_RE   = (CA_post  .d_km - baseCA  ) / obj.R_EARTH_KM;
            perf.MOID_post    = MOID_post;
            perf.CA_post      = CA_post;
        end




        %% --- TYPE 2 ANALYSIS : Sweep Departure and Arrival Times vs Spacecraft Mass ---
        function R = computeMonthsSweep(obj, cfg, meta)
            % computeMonthsSweep  Sweep dep/arr MBI pairs and SC mass; compute ΔV, θ, R1, TOF,
            %                  and post-intercept ΔMOID/ΔCA (R_E). Optionally saves per-asteroid .mat.
            %
            %   R = obj.computeMonthsSweep(cfg, meta)
            %
            % cfg (required fields)
            %   monthsBack : [1×nM]  MBI grid (ascending, e.g., 3:12)
            %   m_sc_vec   : [1×nMass] spacecraft masses [kg]
            %   maxRev     : scalar   Lambert max revs
            %   vInfMax    : scalar   ΔV cap for screening [km/s] (not used here but saved in filenames)
            %   beta       : scalar   ejecta momentum enhancement
            %   -- asteroid mass specification (choose one) --
            %   M_ast      : scalar   asteroid mass [kg], constant for this sweep
            %   OR
            %   D_ast + rhoAst : scalar diameter [m] and density [kg/m^3] to derive M_ast=(4/3)π(R^3)ρ
            %
            % cfg (optional fields)
            %   dt_min     : scalar   post-grid step [min]   (default 120)
            %   SAFETY     : scalar   post-window factor × T_ast (default 1.5)
            %   odeOpt     : struct   odeset (default tight)
            %   save       : logical  save per-asteroid .mat (default false)
            %
            % meta
            %   asteroids  : {1×nAst} cell with fields .name, .coe(1)=a_AU, .MOID_pre_km, .CA_pre_km
            %   StatesMBI  : {1×nAst} each a struct array indexed by monthsBack, fields .stateEarth/.stateAst
            %
            % Output R(1×nAst)
            %   .DVlaunch [nM×nM], .DVint [nM×nM], .Mbest [nM×nM], .R1 [nM×nM], .TOFmonths [nM×nM], .Theta [nM×nM]
            %   .DeltaMOID_RE [nMass×nM×nM], .DeltaCA_RE [nMass×nM×nM]
            %   .filenames (cell) present if cfg.save=true

            % ---- defaults
            if ~isfield(cfg,'dt_min')  || isempty(cfg.dt_min),  cfg.dt_min  = 120; end
            if ~isfield(cfg,'SAFETY')  || isempty(cfg.SAFETY),  cfg.SAFETY  = 1.5; end
            if ~isfield(cfg,'save')    || isempty(cfg.save),    cfg.save    = false; end
            if ~isfield(cfg,'odeOpt')  || isempty(cfg.odeOpt),  cfg.odeOpt  = odeset('RelTol',1e-13,'AbsTol',1e-13); end

            % ---- asteroid mass (constant for the whole sweep)
            if isfield(cfg,'M_ast') && ~isempty(cfg.M_ast)
                M_ast = cfg.M_ast;
            elseif isfield(cfg,'D_ast') && isfield(cfg,'rhoAst') && ~isempty(cfg.D_ast) && ~isempty(cfg.rhoAst)
                R_ast = cfg.D_ast/2;
                M_ast = (4/3)*pi*R_ast^3 * cfg.rhoAst;
            else
                error('computePerfGrid:AstMass','Provide cfg.M_ast OR cfg.D_ast + cfg.rhoAst.');
            end

            monthsBack = cfg.monthsBack(:)';  % row
            m_sc_vec   = cfg.m_sc_vec(:)';    % row

            nAst   = numel(meta.asteroids);
            nM     = numel(monthsBack);
            nMass  = numel(m_sc_vec);

            R = repmat(struct( ...
                'DVlaunch',[],'DVint',[],'Mbest',[],'R1',[],'TOFmonths',[],'Theta',[], ...
                'DeltaMOID_RE',[],'DeltaCA_RE',[],'filenames',[]), 1, nAst);

            % ---- broadcast constants (parfor safe)
            dt_sec = cfg.dt_min * 60;
            mu_km  = obj.MU_SUN_KM;
            mu_AU  = obj.MU_SUN_AU;
            RE_km  = obj.R_EARTH_KM;
            beta   = cfg.beta;

            for kAst = 1:nAst
                ast    = meta.asteroids{kAst};
                States = meta.StatesMBI{kAst};

                % 2-D (pair-level) outputs
                DVlaunch = nan(nM);  DVint = nan(nM);  Mbest = nan(nM);
                R1       = nan(nM);  TOFmo = nan(nM);  Theta = nan(nM);

                % 3-D (mass-dependent) cubes
                DeltaMOID_RE = nan(nMass, nM, nM);
                DeltaCA_RE   = nan(nMass, nM, nM);

                % build (dep,arr) index list with iDep>jArr
                pairI = []; pairJ = [];
                for iDep = 2:nM
                    jv = 1:(iDep-1);
                    pairI = [pairI, repmat(iDep,1,numel(jv))]; %#ok<AGROW>
                    pairJ = [pairJ, jv];                      %#ok<AGROW>
                end
                nPairs = numel(pairI);

                % temp holders for parfor
                dvLaunch_pair = nan(1,nPairs);
                dvInt_pair    = nan(1,nPairs);
                mBest_pair    = nan(1,nPairs);
                theta_pair    = nan(1,nPairs);
                r1_pair       = nan(1,nPairs);
                tofmo_pair    = nan(1,nPairs);
                moid_cell     = cell(1,nPairs);  % each: [nMass×1]
                ca_cell       = cell(1,nPairs);  % each: [nMass×1]

                % asteroid period (for post-window)
                a_AU  = ast.coe(1);
                T_ast = 2*pi*sqrt(a_AU^3 / mu_AU);    % [s]

                parfor p = 1:nPairs
                    iDep = pairI(p);
                    jArr = pairJ(p);

                    XE_launch  = States(iDep).stateEarth;
                    XE_atInt   = States(jArr).stateEarth;
                    XAst_atInt = States(jArr).stateAst;

                    rSC0 = XE_launch(1:3).'; vSC0 = XE_launch(4:6).';
                    rAst = XAst_atInt(1:3).';  vAst = XAst_atInt(4:6).';

                    % Lambert (pair)
                    TOF_mo = monthsBack(iDep) - monthsBack(jArr);
                    best   = obj.lambert_multiRev(rSC0, vSC0, rAst, TOF_mo*30, cfg.maxRev, mu_km);
                    if ~isfinite(best.dV), continue; end

                    % pair-level metrics
                    dvLaunch_pair(p) = best.dV;
                    dvInt_pair(p)    = norm(vAst - best.VF);
                    mBest_pair(p)    = best.m;
                    theta_pair(p)    = real(acosd( dot(best.VF, vAst) / (norm(best.VF)*norm(vAst)) ));
                    TLI_s            = monthsBack(iDep)*30*86400;
                    r1_pair(p)       = 1 - TOF_mo*30*86400 / TLI_s;
                    tofmo_pair(p)    = TOF_mo;

                    % post-intercept window (SAFETY × T_ast)
                    tRemain = monthsBack(jArr)*30*86400;
                    tFwd    = max(tRemain, cfg.SAFETY*T_ast);
                    tspan   = 0:dt_sec:tFwd; if tspan(end)<tFwd, tspan=[tspan tFwd]; end

                    % propagate Earth (baseline) & nominal asteroid
                    [Tnom, XE_post] = ode45(@(t,X) obj.orb.dynamics_2BP_cartesian(t,X,mu_km), tspan, XE_atInt(:), cfg.odeOpt);
                    [~,    XA_nom ] = ode45(@(t,X) obj.orb.dynamics_2BP_cartesian(t,X,mu_km), tspan, XAst_atInt(:), cfg.odeOpt);

                    % baseline CA/MOID (pre-kick)
                    [CA_pre, MOID_pre] = obj.get_CA_MOID(XE_post, XA_nom, Tnom, Tnom);

                    % nominal relative velocity at intercept
                    U_nom_mps = (best.VF - vAst)*1e3;
                    Uhat      = U_nom_mps / max(norm(U_nom_mps), eps);

                    moid_local = nan(nMass,1);
                    ca_local   = nan(nMass,1);

                    for kM = 1:nMass
                        m_sc = m_sc_vec(kM);

                        % ΔV vector (km/s): (m/M)*(U + (β−1)(U·Û)Û)/1e3  with constant M_ast
                        baseKV = (U_nom_mps + (beta-1)*dot(Uhat,U_nom_mps)*Uhat)/1e3; % km/s
                        dV_vec = (m_sc / M_ast) * baseKV;

                        vAst_post   = vAst + dV_vec;
                        XAst_postIC = [rAst; vAst_post];

                        [Tpost, XA_post] = ode45(@(t,X) obj.orb.dynamics_2BP_cartesian(t,X,mu_km), tspan, XAst_postIC(:), cfg.odeOpt);
                        [CA_post, MOID_post] = obj.get_CA_MOID(XE_post, XA_post, Tnom, Tpost);

                        % deltas vs pre, normalized to Earth radii
                        moid_local(kM) = (MOID_post.d_km - MOID_pre.d_km) / RE_km;
                        ca_local  (kM) = (CA_post  .d_km - CA_pre .d_km) / RE_km;
                    end

                    moid_cell{p} = moid_local;
                    ca_cell  {p} = ca_local;
                end % parfor

                % stitch pair results back
                for p = 1:nPairs
                    iDep = pairI(p); jArr = pairJ(p);
                    DVlaunch(iDep,jArr) = dvLaunch_pair(p);
                    DVint   (iDep,jArr) = dvInt_pair(p);
                    Mbest   (iDep,jArr) = mBest_pair(p);
                    Theta   (iDep,jArr) = theta_pair(p);
                    R1      (iDep,jArr) = r1_pair(p);
                    TOFmo   (iDep,jArr) = tofmo_pair(p);

                    if ~isempty(moid_cell{p}), DeltaMOID_RE(:,iDep,jArr) = moid_cell{p}; end
                    if ~isempty(ca_cell  {p}), DeltaCA_RE  (:,iDep,jArr) = ca_cell{p};   end
                end

                % package result per asteroid
                R(kAst).DVlaunch     = DVlaunch;
                R(kAst).DVint        = DVint;
                R(kAst).Mbest        = Mbest;
                R(kAst).R1           = R1;
                R(kAst).TOFmonths    = TOFmo;
                R(kAst).Theta        = Theta;
                R(kAst).DeltaMOID_RE = DeltaMOID_RE;
                R(kAst).DeltaCA_RE   = DeltaCA_RE;

                % optional save
                if cfg.save
                    outDir = "./perf";
                    if ~exist(outDir, 'dir')
                        mkdir(outDir);
                    end
                    nameTag  = regexprep(ast.name,'\s+','_');
                    mb_range = sprintf('%dto%d', monthsBack(1), monthsBack(end));
                    m_min_t  = m_sc_vec(1)/1e3;
                    m_max_t  = m_sc_vec(end)/1e3;
                    massTag  = sprintf('m%.0fto%.0ft', m_min_t, m_max_t);
                    dvTag    = sprintf('dv%.1f', cfg.vInfMax);
                    filename = fullfile(outDir,sprintf('perf_%s_mb_%s_%s_%s.mat', nameTag, mb_range, massTag, dvTag));
                    S = struct( ...
                        'DVlaunch',DVlaunch,'DVint',DVint,'Mbest',Mbest,'R1',R1,'TOFmonths',TOFmo,'Theta',Theta, ...
                        'DeltaMOID_RE',DeltaMOID_RE,'DeltaCA_RE',DeltaCA_RE,'m_sc_vec',m_sc_vec,'monthsBack',monthsBack, ...
                        'M_ast',M_ast);
                    save(filename,'-struct','S');
                    R(kAst).filenames = {filename};
                end
            end
        end



        %%  --- TYPE 3 ANALYSIS: Asteroid Diameter vs Spacecraft Mass --- Single Transfer
        function [DeltaCA_RE, DeltaMOID_RE] = sweepDiamVsMass( ...
                obj, D_ast_vec, m_sc_vec, rhoAst, beta, ...
                XE_int, XAst_int, best, tf_MBI, a_AU, odeOpt)

            % sweepDiamVsMass ΔCA and ΔMOID (in Earth radii) over asteroid diameter × spacecraft mass.
            %
            %   [DeltaCA_RE, DeltaMOID_RE] = obj.sweepDiamVsMass( ...
            %       D_ast_vec, m_sc_vec, rhoAst, beta, XE_int, XAst_int, best, tf_MBI, a_AU, odeOpt)
            %
            % Inputs
            %   D_ast_vec [nD×1]   asteroid diameters (m)
            %   m_sc_vec  [nM×1]   spacecraft masses (kg)
            %   rhoAst              asteroid bulk density (kg/m^3)
            %   beta                ejecta momentum enhancement (–)
            %   XE_int    [1×6]     Earth state (km, km/s) at intercept epoch
            %   XAst_int  [1×6]     Asteroid(pre) state (km, km/s) at intercept epoch
            %   best                Lambert result (uses .VF for nominal relative-velocity direction)
            %   tf_MBI              months-before-impact at intercept (for "time-to-impact")
            %   a_AU                asteroid semi-major axis (AU)  ← used for period & SAFETY sizing
            %   odeOpt              odeset; default high accuracy
            %
            % Outputs
            %   DeltaCA_RE   [nD×nM]  (CA_post  − CA_pre ) / R_E
            %   DeltaMOID_RE [nD×nM]  (MOID_post − MOID_pre) / R_E
            %
            % Notes
            %   • The post-intercept window is sized as: max( tf_MBI*30*86400 , SAFETY * T_ast )
            %     with SAFETY = 1.5 and T_ast = 2π√(a_AU^3/μ_sun_AU). Both CA and MOID are
            %     recomputed after full re-propagation on this window.

            % ---------- constants ----------
            ER_km   = obj.R_EARTH_KM;     % Earth radius [km]
            mu_km   = obj.MU_SUN_KM;      % Sun μ [km^3/s^2]
            mu_AU   = obj.MU_SUN_AU;      % Sun μ [AU^3/s^2]
            SAFETY  = 1.5;                % additional time factor for post intercept propagation

            % ---------- common post-intercept window (uses SAFETY * T_ast) ----------
            tRemain = tf_MBI * 30 * 86400;              % "time to impact" once intercepted [s]
            T_ast   = 2*pi*sqrt(a_AU^3 / mu_AU);        % asteroid period [s]
            tFwd    = max(tRemain, SAFETY * T_ast);     % enforce safety margin
            dt_min  = 30;                               % grid step [min] (keep as in main)
            dt_sec  = dt_min * 60;
            tspan   = 0:dt_sec:tFwd;
            if tspan(end) < tFwd, tspan(end+1) = tFwd; end

            % ---------- propagate Earth & nominal asteroid on that grid ----------
            [~, XE_post] = ode45(@(t,X) obj.orb.dynamics_2BP_cartesian(t,X,mu_km), tspan, XE_int(:),  odeOpt);
            [~, XA_nom ] = ode45(@(t,X) obj.orb.dynamics_2BP_cartesian(t,X,mu_km), tspan, XAst_int(:), odeOpt);

            % ---------- CA/MOID BEFORE kick (baseline, same for all masses/diameters) ----------
            [CA_pre, MOID_pre] = obj.get_CA_MOID(XE_post, XA_nom, tspan, tspan);

            % ---------- precompute nominal relative velocity at intercept ----------
            vAst_int0  = XAst_int(4:6).';
            rAst_int0  = XAst_int(1:3).';
            U_nom_mps  = (best.VF(:) - vAst_int0(:)) * 1e3;      % m/s
            Uhat       = U_nom_mps / max(norm(U_nom_mps), eps);  % unit direction

            % ---------- sweep grids ----------
            nD = numel(D_ast_vec);
            nM = numel(m_sc_vec);
            DeltaCA_RE    = nan(nD, nM);
            DeltaMOID_RE  = nan(nD, nM);

            for id = 1:nD
                D_ast = D_ast_vec(id);
                R_ast = D_ast/2;
                M_ast = (4/3)*pi*R_ast^3 * rhoAst;                 % kg

                % ΔV vector scales linearly with m_sc / M_ast:
                % dV_vec[km/s] = (m/M) * ( U + (β−1)(U·Uhat)Uhat ) / 1e3
                U      = U_nom_mps;
                Udot   = dot(Uhat, U);
                baseKV = (U + (beta-1)*Udot*Uhat) / 1e3;           % km/s (vector)
                dVvec_by_mass = (m_sc_vec(:) / M_ast) .* baseKV.'; % [nM×3]

                for im = 1:nM
                    % post-kick initial state (@ intercept epoch)
                    vAst_post   = vAst_int0(:).' + dVvec_by_mass(im,:);  % km/s
                    XAst_postIC = [rAst_int0(:).'  vAst_post];

                    % propagate post-kick asteroid on the *same* grid (recompute orbit!)
                    [~, XA_post] = ode45(@(t,X) obj.orb.dynamics_2BP_cartesian(t,X,mu_km), tspan, XAst_postIC(:), odeOpt);

                    % CA/MOID AFTER kick
                    [CA_post, MOID_post] = obj.get_CA_MOID(XE_post, XA_post, tspan, tspan);

                    % deltas in Earth radii (vs pre)
                    DeltaCA_RE(id,im)   = (CA_post  .d_km - CA_pre .d_km) / ER_km;
                    DeltaMOID_RE(id,im) = (MOID_post.d_km - MOID_pre.d_km) / ER_km;
                end
            end
        end



        function plotDeltaCAMOIDContours(obj, DeltaCA_RE, DeltaMOID_RE, D_ast_vec, m_sc_vec, astName, t0_MBI, tf_MBI)
            % PLOTDELTACAMOIDCONTOURS  Plot ΔCA/R_E and ΔMOID/R_E in *separate* figures.
            %
            %   obj.plotDeltaCAMOIDContours(DeltaCA_RE, DeltaMOID_RE, D_ast_vec, m_sc_vec, astName, t0_MBI, tf_MBI)
            %
            % Inputs
            %   DeltaCA_RE    [nD×nM]  (CA_post  − CA_pre ) / R_E
            %   DeltaMOID_RE  [nD×nM]  (MOID_post − MOID_pre) / R_E
            %   D_ast_vec     [nD×1]   asteroid diameters [m]
            %   m_sc_vec      [nM×1]   spacecraft dry masses [kg]
            %   astName       string   asteroid name for titles
            %   t0_MBI        scalar   departure lead time (months-before-impact)
            %   tf_MBI        scalar   interception lead time (months-before-impact)

            % ---------- Figure 1: ΔCA / R_E ----------
            f1 = figure('Name','ΔCA / R_E – mass vs asteroid size', ...
                'Color','w','Position',[80 80 950 720]);
            localPlotOne(DeltaCA_RE, D_ast_vec, m_sc_vec, ...
                sprintf('%s – %d/%d MBI — ΔCA / R_E', string(astName), t0_MBI, tf_MBI));

            % ---------- Figure 2: ΔMOID / R_E ----------
            f2 = figure('Name','ΔMOID / R_E – mass vs asteroid size', ...
                'Color','w','Position',[1060 80 950 720]);
            localPlotOne(DeltaMOID_RE, D_ast_vec, m_sc_vec, ...
                sprintf('%s – %d/%d MBI — ΔMOID / R_E', string(astName), t0_MBI, tf_MBI));

            %==================== nested helper ====================
            function localPlotOne(Z, Dvec, Mvec, ttl)
                % Build fine interpolation grid
                [Xm, Ym] = meshgrid(Mvec/1e3, Dvec);                 % tons vs meters
                D_fine   = linspace(min(Dvec), max(Dvec), 200);
                M_fine   = linspace(min(Mvec)/1e3, max(Mvec)/1e3, 250);
                [Xf, Yf] = meshgrid(M_fine, D_fine);
                Zf       = interp2(Xm, Ym, Z, Xf, Yf, 'makima');

                % Color scale (log) — guard against nonpositive/degenerate data
                Zpos = Zf(Zf > 0);
                if isempty(Zpos), Zpos = 1; end
                zmin = max(min(Zpos(:)), 1e-6);
                zmax = max(Zf(:));
                if ~isfinite(zmax) || zmax <= zmin, zmax = zmin*10; end

                % Filled background
                contourf(Xf, Yf, Zf, logspace(log10(zmin), log10(zmax), 80), 'LineColor','none');
                set(gca, 'ColorScale','log', 'YDir','normal');
                colormap(turbo(256)); caxis([zmin zmax]);
                xlabel('Spacecraft dry mass  [tons]');
                ylabel('Asteroid diameter  [m]');
                title(ttl);
                set(gca,'FontSize',16,'Layer','top'); hold on;

                % Fixed grey contours
                fixedLevels = [2 5 10 20 40 60 80 100 120];
                [CSfixed, hAll] = contour(Xf, Yf, Zf, fixedLevels, ...
                    'LineColor',[0.2 0.2 0.2],'LineWidth',0.7);
                if isprop(hAll,'ShowText')
                    hAll.ShowText     = 'on';
                    hAll.LabelSpacing = 220;
                    hAll.LabelFormat  = '%.0f';
                else
                    clabel(CSfixed, hAll, 'FontSize',10,'Color','k','LabelSpacing',220, ...
                        'FontWeight','bold','Rotation',0,'Interpreter','none');
                end

                % Special dashed contours: Earth–Moon distance and 5 R_E
                EM_re = 384400 / obj.R_EARTH_KM;
                [CEM, hEM] = contour(Xf, Yf, Zf, [EM_re EM_re], ...
                    'LineStyle','--','LineWidth',2,'LineColor','w');
                [C5, h5]  = contour(Xf, Yf, Zf, [5 5], ...
                    'LineStyle','--','LineWidth',2,'LineColor','w');

                if isprop(hEM,'ShowText')
                    hEM.ShowText     = 'on'; hEM.LabelSpacing = 220; hEM.LabelFormat = '%.1f';
                else
                    clabel(CEM, hEM, 'FontSize',10,'Color','w','LabelSpacing',220,'FontWeight','bold');
                end
                if isprop(h5,'ShowText')
                    h5.ShowText      = 'on'; h5.LabelSpacing  = 220; h5.LabelFormat  = '%.0f';
                else
                    clabel(C5, h5, 'FontSize',10,'Color','w','LabelSpacing',220,'FontWeight','bold');
                end

                ylim([min(Dvec) max(Dvec)]);
                hold off;
            end
        end





        %% --- Lambert's Multi-Revolution Solver (ΔV-minimizing search) ---
        function best = lambert_multiRev(obj, ri, v0, rf, TOF_days, maxRev, mu)
            %LAMBERT_MULTIREV  Multi-revolution Lambert solver with ΔV minimization.
            %
            %   best = obj.lambert_multiRev(ri, v0, rf, TOF_days, maxRev, mu)
            %
            %   Class method of InterceptionLibrary. Provides a thin wrapper around
            %   the Izzo-based LAMBERT solver, performing a sweep across revolution
            %   counts and transfer-time branches to identify the minimum-ΔV solution.
            %
            %   INPUTS
            %     ri        3x1 initial position vector at departure [km]
            %     v0        3x1 initial velocity at departure        [km/s]
            %     rf        3x1 target position at arrival            [km]
            %     TOF_days  Time-of-flight [days]. Both short-arc (+) and long-arc (−)
            %               transfer branches are tested automatically.
            %     maxRev    Maximum number of revolutions to consider (integer ≥ 0).
            %     mu        Central body gravitational parameter [ km^3 / s^2 ].
            %
            %   OUTPUT (struct BEST)
            %     .VI       3x1 Lambert departure velocity [km/s]
            %     .VF       3x1 Lambert arrival velocity   [km/s]
            %     .dV       Scalar ΔV magnitude at departure [km/s], minimized over
            %               revolutions and arc choices
            %     .DVv      3x1 ΔV vector = VI − v0 [km/s]
            %     .m        Revolution count (±M). Positive = short-way, negative = long-way.
            %     .tfsgn    +1 for short-arc, −1 for long-arc solution branch.
            %
            %   NOTES
            %   • Iterates over [0…maxRev] revolutions, both long/short arcs, to find
            %     the best feasible solution (lowest ΔV).
            %   • Returns Inf in .dV if no feasible solution is found.
            %
            %   EXAMPLE
            %     lib  = InterceptionLibrary();
            %     best = lib.lambert_multiRev(rDep, vDep, rArr, 180, 3, Sun.mu.km);
            %
            %   See also LAMBERT (Izzo's algorithm).

            % force column vectors
            ri = ri(:);
            v0 = v0(:);
            rf = rf(:);

            best.dV = Inf;

            for M = 0:maxRev
                for tfsgn = [+1, -1]
                    tof = tfsgn * TOF_days;
                    for branch = [+1, -1]
                        m = branch * M;
                        % call lambert with row vectors
                        [VIr, VFr, ~, flag] = lambert(ri.', rf.', tof, m, mu);
                        if flag ~= 1
                            continue;
                        end
                        % convert back to columns
                        VI = VIr(:);
                        VF = VFr(:);
                        % compute ΔV
                        DVv = VI - v0;
                        dVmag = norm(DVv);
                        if dVmag < best.dV
                            best.VI    = VI;
                            best.VF    = VF;
                            best.dV    = dVmag;
                            best.DVv   = DVv;
                            best.m     = m;
                            best.tfsgn = tfsgn;
                        end
                    end
                end
            end
        end


        function varargout = lambert(obj,varargin)
            % Pass‑through to the existing lambert solver.
            [varargout{1:nargout}] = lambert(varargin{:});
        end





        function [CA, MOID] = get_CA_MOID(obj, EarthStates, AstStates, timeE, timeA)
            % get_CA_MOID  Compute time-synced closest approach and global MOID
            %
            % Syntax:
            %   [CA, MOID] = get_CA_MOID(EarthStates, AstStates, timeE, timeA)
            %
            % Brief:
            %   Finds the minimum separation between Earth and asteroid trajectories
            %   at matching epochs (CA) and the overall minimum-distance pair (MOID).
            %
            % Inputs:
            %   EarthStates  [N×6 double] – Earth states [r v] in km and km/s
            %   AstStates    [M×6 double] – Asteroid states [r v] in km and km/s
            %   timeE        [N×1 double] – Epochs for EarthStates (s)
            %   timeA        [M×1 double] – Epochs for AstStates (s)
            %
            % Outputs:
            %   CA   struct with fields
            %     .d_km       – closest approach distance (km)
            %     .idxEarth   – index in EarthStates/timeE
            %     .idxAst     – index in AstStates/timeA
            %     .timeEarth  – epoch of closest approach (s)
            %     .timeAst    – epoch of closest approach (s)
            %     .stateEarth – Earth state at CA [1×6]
            %     .stateAst   – Asteroid state at CA [1×6]
            %
            %   MOID struct with same fields, but referring to the global minimum-distance
            %   pair (time-independent).


            % -------------------------------------------------------------------------
            % 0) user-tuneable search radius for MOID speed-up of computation
            THRESH_KM = 2e6;            % 2 million km  (more less 5.25 × lunar distance)

            % -------------------------------------------------------------------------
            % 1) time-synced closest approach  (vectorized) --------------------
            N  = min(size(EarthStates,1), size(AstStates,1));

            rE = EarthStates(1:N,1:3);
            rA = AstStates (1:N,1:3);

            [d_ca, k_ca] = min(vecnorm(rE - rA, 2, 2) );

            CA = struct( ...
                'd_km'      , d_ca              , ...
                'idxEarth'  , k_ca              , ...
                'idxAst'    , k_ca              , ...
                'timeEarth' , timeE(k_ca)       , ...
                'timeAst'   , timeA(k_ca)       , ...
                'stateEarth', EarthStates(k_ca,:), ...
                'stateAst'  , AstStates (k_ca,:) );

            % -------------------------------------------------------------------------
            % 2) global MOID  – KD-tree with radius pre-filter for speed  -------------
            rE_all = EarthStates(:,1:3);
            rA_all = AstStates  (:,1:3);

            tree  = KDTreeSearcher(rA_all,'Distance','euclidean');
            idxIn = rangesearch(tree, rE_all, THRESH_KM);   % cell array of candidate hits

            % flatten hits into two parallel index vectors
            iE = [];  iA = [];
            for k = 1:numel(idxIn)
                if ~isempty(idxIn{k})
                    iE = [iE ; k*ones(numel(idxIn{k}),1)];
                    iA = [iA ; idxIn{k}(:)];
                end
            end

            if isempty(iE)
                % no point fell inside thresh_km -> fall back to plain knnsearch
                [idxNearest, dNearest] = knnsearch(tree, rE_all);
                [d_moid, iE_moid]  = min(dNearest);
                iA_moid            = idxNearest(iE_moid);
            else
                % evaluate only the small set of candidates
                dCand   = vecnorm( rE_all(iE,:) - rA_all(iA,:), 2, 2 );
                [d_moid, jMin] = min(dCand);
                iE_moid = iE(jMin);
                iA_moid = iA(jMin);
            end

            MOID = struct( ...
                'd_km'      , d_moid                   , ...
                'idxEarth'  , iE_moid                 , ...
                'idxAst'    , iA_moid                 , ...
                'timeEarth' , timeE(iE_moid)          , ...
                'timeAst'   , timeA(iA_moid)          , ...
                'stateEarth', EarthStates(iE_moid,:)  , ...
                'stateAst'  , AstStates (iA_moid,:)   );
        end



        %% --- Setup Functions ---
        function ICs_atMBI = getStatesAtMBI(obj, statesAtMOID, monthsBack, odeOpt)
            % getStatesAtMBI  Backward‐propagate Earth & asteroid from MOID
            %
            % Syntax:
            %   [T, IC] = obj.getStatesAtMBI(statesAtMOID, months, odeOpt)
            %
            % Brief:
            %   Propagates Earth and a single asteroid backward in time from their
            %   MOID closest‐approach conditions for specified look-back months,
            %   using a two-body integrator.
            %
            % Inputs:
            %   statesAtMOID – struct with fields:
            %       .stateEarth [1×6] – Earth [r v] at CA (km, km/s)
            %       .stateAst   [1×6] – Asteroid [r v] at CA (km, km/s)
            %   months        [1×K] – look-back times in months (1 mo≈30 d)
            %   odeOpt        struct – (opt.) ODE45 options (RelTol, AbsTol)
            %
            % Outputs:
            %   IC struct(K×1) with fields:
            %      .month, .stateEarth [1×6], .stateAst [1×6]


            if nargin<4 || isempty(odeOpt)
                odeOpt = odeset('RelTol',1e-12,'AbsTol',1e-12);
            end

            monthsBack  = monthsBack(:).';           % make months a row vector
            nM          = numel(monthsBack);         % number of months

            rE = zeros(nM,3);  vE = zeros(nM,3);
            rA = zeros(nM,3);  vA = zeros(nM,3);
            ICs_atMBI(nM) = struct('month',[],'stateEarth',[],'stateAst',[]);

            for k = 1:nM   % for each month in the monthsBack vector
                dt = monthsBack(k) * 30 * 86400;    % time in seconds before the MOID point

                % ------- Earth backward propagation -------
                [~,XE] = ode45(@(t,X) obj.orb.dynamics_2BP_cartesian(t,X,obj.MU_SUN_KM), [0 -dt], statesAtMOID.stateEarth(:), odeOpt);
                rE(k,:) = XE(end,1:3);    vE(k,:) = XE(end,4:6);

                % ------- Asteroid backward propagation ----
                [~,XA] = ode45(@(t,X) obj.orb.dynamics_2BP_cartesian(t,X,obj.MU_SUN_KM), [0 -dt], statesAtMOID.stateAst(:),  odeOpt);
                rA(k,:) = XA(end,1:3);    vA(k,:) = XA(end,4:6);

                % ------- struct entry for each month ---------------------
                ICs_atMBI(k).month      = monthsBack(k);
                ICs_atMBI(k).stateEarth = [rE(k,:)  vE(k,:)];
                ICs_atMBI(k).stateAst   = [rA(k,:)  vA(k,:)];
            end

        end


        %%  --- Dynamics Functions ---

        function bodies = propagate2BPHistories(obj, bodies, muCentral, odeOpts)
            % propagate2BPHistories  Fill in t_hist/X_hist for a list of 2BP bodies.
            %
            %   bodies = propagate2BPHistories(bodies, muCentral, orb, odeOpts)
            %   loops over each element of the cell‐array "bodies", reads
            %     bodies{k}.IC     – 1×6 initial state [r v] (km, km/s)
            %     bodies{k}.tspan  – scalar tf or vector of epochs [s]
            %   and computes
            %     bodies{k}.t_hist – time history [s]
            %     bodies{k}.X_hist – state history [N×6]
            %
            % Inputs:
            %   bodies     cell array of structs with fields .IC and .tspan
            %   muCentral  GM of central body (km^3/s^2)
            %   orb         instance of KeplerianOrbitalMechanicsLibrary
            %   odeOpts     odeset options for ODE45
            %
            % Example:
            %   bodies = propagate2BPHistories(bodies, Sun.mu.km, orb, optionsODE);

            for k = 1:numel(bodies)
                ts = bodies{k}.tspan;
                % ensure time‐span is a 2‐element or vector, starting at zero
                if isscalar(ts)
                    ts = [0 ts];
                elseif ts(1)~=0
                    ts = [0 ts(:)'];
                end
                % propagate
                [tH, XH] = ode45(@(t,X) obj.orb.dynamics_2BP_cartesian(t, X, muCentral), ts, bodies{k}.IC(:), odeOpts);
                bodies{k}.t_hist = tH;
                bodies{k}.X_hist = XH;
            end
        end


        function [tHist,Xhist] = propagate2BP(obj,IC,tspan,muCentral,odeOpts)
            % Convenience 2‑BP propagator using obj.orb.
            if nargin<5||isempty(odeOpts), odeOpts=odeset('RelTol',1e-13,'AbsTol',1e-13); end
            if isscalar(tspan), tspan=[0 tspan]; end
            [tHist,Xhist]=ode45(@(t,X)obj.orb.dynamics_2BP_cartesian(t,X,muCentral),tspan,IC(:),odeOpts);
        end

        function deltaV = computeAsteroidDeltaV(obj,m,M,U,Ehat,beta)
            % Robust against m<=0, M<=0, and undefined ratios.
            if ~(m>0) || ~(M>0) || ~isfinite(m/M)
                deltaV = zeros(3,1);
                return
            end
            U = U(:); Ehat = Ehat(:);
            UdotE = dot(Ehat,U);
            deltaV = (m/M) * ( U + (beta-1)*UdotE*Ehat );
        end




        % --- TYPE 1 ANALYSIS HELPERS ---


        function bodies = addPostInterceptAsteroid_simple(obj, bodies, cfg)
            % REQUIRED: cfg.tail_sec (s), cfg.step_min (min), cfg.bestVF, cfg.m_sc, cfg.M_ast, cfg.beta
            % cfg.odeOpt

            % Use the nominal asteroid state at the intercept from bodies{2} (last sample of the leg)
            XA_nom_int = bodies{2}.X_hist(end,:);
            rA_nom_int = XA_nom_int(1:3).'; vA_nom_int = XA_nom_int(4:6).';
            VF = cfg.bestVF(:);

            % Impactor SC state at interception epoch
            Xsc_int = bodies{3}.X_hist(end,:);        
            rSC_int  = Xsc_int(end,1:3);  rSC_int   = rSC_int(:);

            if norm(rA_nom_int - rSC_int) > 1e-3
                warning('SC and nominal asteroid positions differ at intercept (||Δr|| = %.3g km).', ...
                    norm(rA_nom_int - rSC_int));
            end

            % ΔV on asteroid (km/s) – works for m_sc = 0 too (then dV = 0)
            U_mps  = (VF - vA_nom_int) * 1e3;
            Uhat   = U_mps ./ max(norm(U_mps), eps);
            dV_kms = obj.computeAsteroidDeltaV(cfg.m_sc, cfg.M_ast, U_mps, Uhat, cfg.beta) / 1e3;

            X0_post = [rA_nom_int; vA_nom_int + dV_kms];    

            % ---- build local time vector and FORCE the exact endpoint
            step_s   = cfg.step_min * 60;
            tail_sec = max(0, cfg.tail_sec);
            t_rel    = 0:step_s:tail_sec;
            if t_rel(end) < tail_sec
                t_rel(end+1) = tail_sec;   % <-- critical: include exact cadence end
            end

            % integrate on requested epochs (ode45 returns states exactly at t_rel)
            [~, X_rel] = ode45(@(t,X) obj.orb.dynamics_2BP_cartesian(t,X,obj.MU_SUN_KM), ...
                t_rel, X0_post, cfg.odeOpt);

            % name
            astName = "Asteroid";
            if numel(bodies) >= 2 && isfield(bodies{2},'name') && ~isempty(bodies{2}.name)
                astName = string(bodies{2}.name);
            end

            postBody.name   = sprintf('%s (post-intercept)', astName);
            postBody.IC     = X0_post.';     % IC at t=0 (impact)
            postBody.tspan  = t_rel.';       % local time since impact
            postBody.t_hist = t_rel.';
            postBody.X_hist = X_rel;
            bodies{end+1}   = postBody;
        end




        % function bodies = addPostInterceptAsteroid_simple(obj, bodies, cfg)
        %     % ADDPOSTINTERCEPTASTEROID_SIMPLE
        %     % Simple post-impact asteroid whose time axis starts at t=0 (at impact)
        %     % and runs forward for a requested duration. No NaNs, no global alignment.
        %     %
        %     % REQUIRED cfg fields
        %     %   Xast_nom_int  [1x6]  nominal asteroid state at intercept [km, km/s]
        %     %   bestVF        [3x1]  interceptor velocity at intercept [km/s]
        %     %   m_sc, M_ast   [kg]
        %     %   beta          [–]
        %     %   tail_sec      [s]    how long to propagate after the impact
        %     %   step_min      [min]
        %     % OPTIONAL
        %     %   odeOpt        odeset; default RelTol=AbsTol=1e-13
        %
        %
        %
        %     % ---- unpack inputs, asteroid  and sc position at interception
        %
        %     % note there are 2 opions to pick ast nom pos at int
        %
        %     % XA_nom_int = bodies{2}.X_hist(end,:);
        %     XA_nom_int = cfg.Xast_nom_int;
        %
        %     Xsc_int = bodies{3}.X_hist(end,:);
        %
        %     rA_nom_int = XA_nom_int(1:3).';   rA_nom_int = rA_nom_int(:);
        %     vA_nom_int = XA_nom_int(4:6).';   vA_nom_int = vA_nom_int(:);
        %
        %     rSC_int  = Xsc_int(end,1:3);  rSC_int   = rSC_int(:);
        %
        %     if norm(rA_nom_int - rSC_int) > 1e-3
        %         warning('SC and nominal asteroid positions differ at intercept (||Δr|| = %.3g km).', ...
        %             norm(rA_nom_int - rSC_int));
        %     end
        %
        %     VF = cfg.bestVF(:);
        %
        %     % ---- ΔV on asteroid (km/s)
        %     U_mps  = (VF - vA_nom_int) * 1e3;
        %     Uhat   = U_mps ./ max(norm(U_mps), eps);
        %     dV_kms = obj.computeAsteroidDeltaV(cfg.m_sc, cfg.M_ast, U_mps, Uhat, cfg.beta) / 1e3;
        %
        %     % ---- post-impact IC (at t=0 of the post body)
        %     X0_post = [rA_nom_int; vA_nom_int + dV_kms];
        %
        %     % ---- build local time from impact → impact + tail_sec
        %     step_s   = cfg.step_min * 60;
        %
        %     tail_sec = max(0, getfielddef(cfg,'tail_sec',0));   % <— use tail_sec
        %
        %     t_rel = 0:step_s:tail_sec;
        %
        %     % ---- integrate local post-interception asteroid for the time in tail
        %     [T_rel, X_rel] = ode45(@(t,X) obj.orb.dynamics_2BP_cartesian(t,X,obj.MU_SUN_KM), ...
        %         t_rel, X0_post, cfg.odeOpt);
        %
        %     % ---- name
        %     astName = "Asteroid";
        %     if numel(bodies) >= 2 && isfield(bodies{2},'name') && ~isempty(bodies{2}.name)
        %         astName = string(bodies{2}.name);
        %     end
        %
        %     % ---- append new, independent-timeline body (starts at t=0 = impact)
        %     postBody.name   = sprintf('%s (post-intercept)', astName);
        %     postBody.IC     = X0_post.';   % row
        %     postBody.tspan  = t_rel.';     % column = local time
        %     postBody.t_hist = t_rel.';     % "
        %     postBody.X_hist = X_rel;
        %     bodies{end+1}   = postBody;
        %
        %     % tiny local helper (add at bottom of file if you don't already have one)
        %     function val = getfielddef(S, field, default)
        %         if isfield(S, field) && ~isempty(S.(field)), val = S.(field); else, val = default; end
        %     end
        %
        % end
        %
        %


        function TOF = computeTOF(obj, t0_MBI, tf_MBI)
            % COMPUTETOF  Convert a pair of MBI values to time-of-flight fields.
            %
            %   TOF = obj.computeTOF(t0_MBI, tf_MBI)
            %
            % Outputs a struct with:
            %   .months = t0_MBI - tf_MBI
            %   .days   = months * 30
            %   .sec    = days   * 86400
            TOF.months = t0_MBI - tf_MBI;
            TOF.days   = TOF.months * 30;
            TOF.sec    = TOF.days   * 86400;
        end



        function scaleUp = pickScaleUp(obj, cfg, a_AU, t0_MBI, tf_MBI)
            % PICKSCALEUP  Decide how much to extend the propagation window.
            %
            %   s = obj.pickScaleUp(cfg, a_AU, t0_MBI, tf_MBI)
            %
            % Behavior:
            %   • If cfg.propTilEarthImpact, s = t0_MBI / (t0_MBI - tf_MBI)
            %   • If cfg.propForOneMoreRev, s = (TOF.sec + T_ast) / TOF.sec
            %       where T_ast is the asteroid period from a_AU and obj.MU_SUN_AU
            %   • Else, s = 1
            %
            % Notes:
            %   • Uses obj.computeTOF to get TOF.sec
            %   • Safe divisions guard against zero denominators with EPS

            % --- compute TOF struct (months, days, sec)
            TOF = obj.computeTOF(t0_MBI, tf_MBI);

            if isfield(cfg,'propTilEarthImpact') && cfg.propTilEarthImpact
                scaleUp = t0_MBI / max(t0_MBI - tf_MBI, eps);

            elseif isfield(cfg,'propForOneMoreRev') && cfg.propForOneMoreRev
                T_ast = 2*pi*sqrt(a_AU^3 / obj.MU_SUN_AU);
                scaleUp     = (TOF.sec + T_ast) / max(TOF.sec, eps);

            else
                scaleUp = 1;
            end
        end

        function tspan = makeTimeGrid(obj, tf_sec, step_min, scaleUp)
            % MAKETIMEGRID  Build a fixed-step time vector up to tf_sec*scaleUp.
            %   outputs a tspan in units of seconds
            %   tspan = obj.makeTimeGrid(tf_sec, step_min, scaleUp)
            step_s = step_min * 60;
            tEnd   = tf_sec * scaleUp;
            tspan  = 0:step_s:tEnd;
            if tspan(end) < tEnd, tspan(end+1) = tEnd; end
        end

        function bodies = assembleBodies(obj, Xearth_t0, Xast_t0, VI, tspan_tf, astName)
            % ASSEMBLEBODIES  Create {Earth, Asteroid, Interceptor} structs for plotting or propagation.
            %
            %   bodies = obj.assembleBodies(Xearth_t0, Xast_t0, VI, tspan_tf, astName)
            if nargin < 6 || isempty(astName), astName = "Asteroid"; end

            Xint_t0 = [Xearth_t0(1:3), VI(:)'];  % interceptor's initial state, Earth's pos and VI from Lambert

            % we put toegether in a cell array of structs, the Earth, Ast
            % and Iterceptor ICs at DEPARTURE TIME of the Interceptor
            bodies = {
                struct('name',"Earth",         'IC',Xearth_t0, 'tspan',tspan_tf)
                struct('name',string(astName), 'IC',Xast_t0,   'tspan',tspan_tf)
                struct('name',"Interceptor",   'IC',Xint_t0,   'tspan',tspan_tf)
                };
        end

        function s = makeMetricsTitle(obj, perf, cfg, M_ast_ton, t0_MBI, tf_MBI)
            % MAKEMETRICSTITLE  Compose a multiline title with key run metrics.
            %
            %   s = obj.makeMetricsTitle(perf, cfg, M_ast_ton, t0_MBI, tf_MBI)
            s = sprintf(['\nAst Mass = %.1f tons \nSC Mass = %.1f tons \n' ...
                'Interception at %.1f MBI \nLaunch at %.1f MBI \n' ...
                '|V_{inf}| = %.2f km/s \n|ΔV_{inter}| = %.2f km/s \n' ...
                'θ_{int} = %.1f°\n\\beta = %.1f'], ...
                M_ast_ton, cfg.m_sc/1e3, tf_MBI, t0_MBI, ...
                perf.Vinf, perf.DVint, perf.theta, cfg.beta);
        end

        function printPerfDelta(obj, perf, ast)
            % PRINTPERFDELTA  Console summary of CA/MOID deltas vs. pre-impact baselines.
            %
            %   obj.printPerfDelta(perf, ast)
            fprintf('\n===== ΔMOID / ΔCA (post) =====\n');
            fprintf('MOID  pre  : %10.3f km\n', ast.MOID_pre_km);
            fprintf('MOID  post : %10.3f km\n', perf.MOID_post.d_km);
            fprintf('CA    pre  : %10.3f km\n', ast.CA_pre_km);
            fprintf('CA    post : %10.3f km\n', perf.CA_post.d_km);
            fprintf('ΔMOID (R_E): %10.4f R_E\n', perf.DeltaMOID_RE);
            fprintf('ΔCA   (R_E): %10.4f R_E\n', perf.DeltaCA_RE);
            fprintf('==============================\n\n');
        end

        function markCAonCurrentAxes(obj, perf, astName)
            % MARKCAONCURRENTAXES  Add Earth/asteroid markers & labels at post-intercept CA.
            %
            %   obj.markCAonCurrentAxes(perf, astName)

            ax = gca; hold(ax,'on');

            % --- ensure astName is a scalar string/char ---
            if isstring(astName)
                if ~isscalar(astName)
                    astName = strjoin(astName, "");   % collapse to one string
                end
                astName = char(astName);              % convert to char
            elseif iscell(astName)
                astName = strjoin(string(astName), "");
                astName = char(astName);
            elseif isnumeric(astName)
                astName = char(string(astName));
            end

            % --- plot Earth marker ---
            plot3(ax, perf.CA_post.stateEarth(1), perf.CA_post.stateEarth(2), perf.CA_post.stateEarth(3), ...
                'bo','MarkerFaceColor','b','DisplayName','Earth @ CA');
            text(perf.CA_post.stateEarth(1), perf.CA_post.stateEarth(2), perf.CA_post.stateEarth(3), ...
                '  Earth @ CA','Color','b','FontSize',12);

            % --- plot Asteroid marker ---
            plot3(ax, perf.CA_post.stateAst(1), perf.CA_post.stateAst(2), perf.CA_post.stateAst(3), ...
                'ro','MarkerFaceColor','r','DisplayName',[astName ' @ CA']);
            text(perf.CA_post.stateAst(1), perf.CA_post.stateAst(2), perf.CA_post.stateAst(3), ...
                [' ' astName ' (post-intercept) @ CA'],'Color','r','FontSize',12);

            legend(ax,'show','Location','best','FontSize',12);
        end


        function applyOverlaySphere(obj, which, xyz)
            % APPLYOVERLAYSPHERE  Draw Earth SOI or Earth–Moon distance sphere at xyz.
            %
            %   obj.applyOverlaySphere(which, xyz)
            %
            % Notes:
            %   Looks up Earth SOI via CelestialBody('Earth'); EM distance fixed at 384,400 km.
            ax = gca; hold(ax,'on'); [ux,uy,uz] = sphere(50);
            earth = CelestialBody('Earth');
            switch upper(string(which))
                case "SOI"
                    r = earth.soi.km; nm = 'Earth SOI';
                otherwise
                    r = 384400;       nm = 'Earth–Moon distance';
            end
            surf(ux*r + xyz(1), uy*r + xyz(2), uz*r + xyz(3), ...
                'FaceColor',[.2 .6 1], 'FaceAlpha',.25, 'EdgeColor','none', 'DisplayName',nm);
        end


        function plotEarthCentered(obj, bodies, perf, XE_post, XA_post)
            % PLOTEARTHCENTERED  Earth-centered visualization of nominal/post-impact asteroid.
            %
            %   obj.plotEarthCentered(bodies, perf, XE_post, XA_post)
            %
            % Robust to duplicate Earth time stamps and NaN-padded segments. Also plots the
            % post-impact asteroid position at the nominal Earth-impact epoch (end of timeline).

            astName = bodies{2}.name;
            % --- normalize astName to scalar char (defensive) ---
            if isstring(astName)
                astName = char(strjoin(astName,""));
            elseif iscell(astName)
                astName = char(strjoin(string(astName),""));
            else
                astName = char(string(astName));
            end

            % ---------------- Earth-centered histories ----------------
            % Earth timeline (may contain duplicates at the intercept due to padding)
            tE  = bodies{1}.t_hist(:);
            XE  = bodies{1}.X_hist(:,1:3);

            % De-duplicate Earth sample points (critical for interp1)
            [tEuniq, ia] = unique(tE,'stable');
            XEuniq       = XE(ia,:);

            % Nominal asteroid (same pre-intercept timeline as Earth, plus NaN tail)
            tA     = bodies{2}.t_hist(:);
            % Interpolate Earth @ tA safely
            EX_atA = interp1(tEuniq, XEuniq, tA, 'linear', 'extrap');
            XA_rel = bodies{2}.X_hist(:,1:3) - EX_atA;

            % Post-intercept asteroid (NaN padded pre-hit; valid after intercept)
            tAp     = bodies{4}.t_hist(:);
            EX_atAp = interp1(tEuniq, XEuniq, tAp, 'linear', 'extrap');
            XAp_rel = bodies{4}.X_hist(:,1:3) - EX_atAp;

            % ---------------- Plot Earth-centered ----------------
            bodiesRel = {
                struct('name','Earth',                  'X_hist',zeros(size(XA_rel,1),6),'t_hist',tA)
                struct('name',[astName ' (nominal)'],   'X_hist',[XA_rel  zeros(size(XA_rel))],  't_hist',tA)
                struct('name',[astName ' (intercept)'], 'X_hist',[XAp_rel zeros(size(XAp_rel))], 't_hist',tAp)
                };

            plt = PlotterLibrary();
            plt.plot_2BP_trajectories(bodiesRel, "Earth", CelestialBody("Earth").mu.km, false, ...
                'StatesProvided', true, 'ShowStartEnd', false, 'Title', "");

            hold on; axis equal; grid on;

            % Earth–Moon nominal distance sphere for scale
            [sx,sy,sz] = sphere(60);
            surf(sx*384400, sy*384400, sz*384400, ...
                'FaceColor',[.2 .6 1], 'FaceAlpha',.25, 'EdgeColor','none', ...
                'DisplayName','Earth–Moon distance');

            % ---------------- Mark CA (post) ----------------
            cols = lines(numel(bodiesRel));
            rCA  = perf.CA_post.stateAst(1:3) - perf.CA_post.stateEarth(1:3);
            plot3(rCA(1), rCA(2), rCA(3), 'o', 'MarkerFaceColor', cols(3,:), ...
                'MarkerEdgeColor','k', 'MarkerSize', 8, 'DisplayName', 'Post-intercept CA');

            % ---------------- Path from impact → CA (if provided) ----------------
            if nargin >= 5 && ~isempty(XE_post) && ~isempty(XA_post)
                XExt_rel = XA_post(:,1:3) - XE_post(:,1:3);
                % try to use CA index if present; otherwise fall back to min distance
                idxImpact = 1;
                if isfield(perf,'CA_post') && isfield(perf.CA_post,'idxAst') && ...
                        ~isempty(perf.CA_post.idxAst) && perf.CA_post.idxAst <= size(XExt_rel,1)
                    idxCA = perf.CA_post.idxAst;
                else
                    [~,idxCA] = min(sum(XExt_rel.^2,2));
                end
                plot3(XExt_rel(idxImpact:idxCA,1), XExt_rel(idxImpact:idxCA,2), XExt_rel(idxImpact:idxCA,3), ...
                    'Color', cols(3,:), 'LineWidth', 2, 'HandleVisibility', 'off');
            end

            % ---------------- Mark post-asteroid @ nominal Earth-impact epoch ----------------
            % Assumption: your unified timeline ends at the original nominal Earth-impact time.
            t_nomImpact = bodies{1}.t_hist(end);           % end of unified timeline
            % Locate this epoch on the post-impact asteroid timeline (tAp)
            i_nom = find(abs(tAp - t_nomImpact) <= max(eps(t_nomImpact),1e-6), 1, 'last');
            if isempty(i_nom), i_nom = numel(tAp); end     % fallback to last available sample
            if all(isfinite(XAp_rel(i_nom,:)))
                rNom = XAp_rel(i_nom,:);
                plot3(rNom(1), rNom(2), rNom(3), 's', ...
                    'MarkerFaceColor',[0.9 0.3 0.1], 'MarkerEdgeColor','k', ...
                    'MarkerSize', 8, 'DisplayName','Post-ast @ nominal impact time');
            end

            xlim([-1.5e6 1.5e6]); ylim([-1.5e6 1.5e6]);
            legend('Location','best'); drawnow;
        end


    end



end
