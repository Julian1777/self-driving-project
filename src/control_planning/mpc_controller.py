import numpy as np
import casadi as ca
import time

# we use forward euler method to predict the next state its not perfect but works well for small T like 0.01 seconds
# Two weight matrices becasue Q penalizes deviation from refrence trajectory
# and R penalizes control effort like aggresive acceleration and rough steering


class MPCController:
    def __init__ (self, T=0.1, N=10, verbose=False):
        self.T = T # Time we look ahead 0.1 sec per step = 10 steps × 0.1 = 1 second lookahead
        self.N = N # Number of optimization steps
        self.verbose = verbose

        print("[MPC] Initializing MPC Controller")

        # define symbolic variables
        # State vector: [x, y, vx, vy, theta, vtheta]
        # - x, y: position in world frame (meters)
        # - vx: longitudinal velocity (m/s)
        # - vy: lateral velocity (m/s)
        # - theta: heading angle (radians)
        # - vtheta: heading rate / angular velocity (rad/s)


        self.x = ca.SX.sym('x')
        self.y = ca.SX.sym('y')
        self.vx = ca.SX.sym('vx')
        self.vy = ca.SX.sym('vy')
        self.theta = ca.SX.sym('theta')
        self.vtheta = ca.SX.sym('vtheta')

        self.states = ca.vertcat(self.x, self.y, self.vx, self.vy, self.theta, self.vtheta)
        self.n_states = self.states.numel() # 6 elements in the state vector

        # Control vector: [ax, delta]
        # - ax: longitudinal acceleration (m/s²), mapped to throttle/brake
        # - delta: steering angle (radians)

        self.ax = ca.SX.sym('ax') # acceleration
        self.delta = ca.SX.sym('delta') # steering angle
        self.controls = ca.vertcat(self.ax, self.delta) # vector with 2 control inputs
        self.n_controls = self.controls.numel() # 2 control inputs

        # define the vehicle dynamics using simplified bicycle model will expand later

        self.caf = 1.0 # front cornering stiffness
        self.car = 1.0 # rear cornering stiffness
        self.m = 1500.0 # mass of the vehicle in kg
        self.lf = 1.0 # distance from center to front axle in meters
        self.lr = 1.0 # distance from center to rear axle in meters
        self.Iz = 1500.0 # yaw moment of inertia in kg*m²
        self.g = 9.81 # gravity on earth

        # rotation matrix from body frame to world frame

        cos_theta = ca.cos(self.theta)
        sin_theta = ca.sin(self.theta)

        # world frame vel = rot * body frame vel
        x_dot = self.vx * cos_theta - self.vy * sin_theta
        y_dot = self.vx * sin_theta + self.vy * cos_theta

        # longitudal acceleration
        vx_dot = self.ax

        # lateral acceleration from tires
        # lateral accel depends on steering angle and velocity
        vy_dot = (2 * self.caf / self.m) * self.delta + (-(2 * self.caf + 2 * self.car) / (self.m * (self.vx + 1e-4))) * self.vy

        # heading rate angular vel
        theta_dot = self.vtheta

        # angular accel from tire torques
        vtheta_dot = (2 * self.lf * self.caf / self.Iz) * self.delta + (-(2 * self.lf * self.caf + 2 * self.lr * self.car) / (self.Iz * (self.vx + 1e-4))) * self.vy


        # stack derivatives into vector right hand side of the dynamics equations
        self.rhs = ca.vertcat(x_dot, y_dot, vx_dot, vy_dot, theta_dot, vtheta_dot)

        # create casadi function f(state, control) that computes the state derivatives
        self.f_dynamics = ca.Function('f', [self.states, self.controls], [self.rhs])

        # define decision variables for optimization
        # X is the future trajectory (position vel etc)
        # 6 states N+1 timestamps
        self.X = ca.SX.sym('X', self.n_states, self.N + 1) # states over the horizon (N+1 steps)

        # U are the future control inputs
        # 2 controls N timestamps since we apply control at each step
        self.U = ca.SX.sym('U', self.n_controls, self.N) # controls over the horizon (N steps)

        # P parameter vector containing the intial states, refrence traj, and refrence controls
        self.P = ca.SX.sym('P', self.n_states + self.N * (self.n_states + self.n_controls)) # initial state + reference trajectory (x,y,vx) for N steps



        # define cost function for optimization this is what the optimizer will try and minimize
        # weight matrices taken from origial repo
        Q = np.zeros((self.n_states, self.n_states))
        Q[0, 0] = 10.0   # x-position tracking (low importance, we mostly care about y)
        Q[1, 1] = 50.0   # y-position tracking (stay in lane)
        Q[2, 2] = 100.0  # vx tracking (maintain desired speed)
        Q[3, 3] = 1.0    # vy tracking (lateral velocity less critical)
        Q[4, 4] = 1.0    # theta tracking (heading less critical)
        Q[5, 5] = 1.0    # vtheta tracking
        
        R = np.zeros((self.n_controls, self.n_controls))
        R[0, 0] = 1.0    # ax cost (avoid aggressive acceleration)
        R[1, 1] = 5.0    # delta cost (avoid jerky steering)

        obj = 0 # initialize cost

        # CODE FROM ORIGINAL REPO

        # For each prediction step k = 0, 1, ..., N-1
        for k in range(self.N):
            # Extract state and control at step k
            X_k = self.X[:, k]
            U_k = self.U[:, k]
            
            # Extract reference state and control from parameter vector P
            # Reference state starts at index: 6 + k*(6+2) = 6 + 8k
            X_ref_k = self.P[6 + 8*k : 6 + 8*k + 6]
            U_ref_k = self.P[6 + 8*k + 6 : 6 + 8*k + 8]
            
            # State tracking cost: (X_k - X_ref_k)ᵀ Q (X_k - X_ref_k)
            state_error = X_k - X_ref_k
            obj += state_error.T @ Q @ state_error
            
            # Control effort cost: (U_k - U_ref_k)ᵀ R (U_k - U_ref_k)
            # (Usually U_ref = [0, 0], meaning we prefer smooth control)
            control_error = U_k - U_ref_k
            obj += control_error.T @ R @ control_error
        
        # define constraints
        constraints = []
        
        # Initial state constraint: X_0 = initial state from P
        # This forces the first state to match current measurement
        constraints.append(self.X[:, 0] - self.P[0:6])
        
        # Dynamics constraints: X_{k+1} = X_k + T * f(X_k, U_k)
        # This enforces the vehicle physics at each step
        for k in range(self.N):
            X_k = self.X[:, k]
            U_k = self.U[:, k]
            X_next = self.X[:, k + 1]
            
            # Euler integration: x(t+T) = x(t) + T * dx/dt
            state_derivative = self.f_dynamics(X_k, U_k)
            X_next_predicted = X_k + self.T * state_derivative
            
            constraints.append(X_next - X_next_predicted)
        
        # Stack all constraints into a single vector
        # These will be enforced to equal zero
        g = ca.vertcat(*constraints)
        
        # These are inequality constraints implemented as variable bounds
        
        # Lower and upper bounds for optimization variables [X, U]
        lbx = np.zeros((self.n_states * (self.N + 1) + self.n_controls * self.N, 1))
        ubx = np.zeros((self.n_states * (self.N + 1) + self.n_controls * self.N, 1))
        
        # State bounds for all timesteps
        for k in range(self.N + 1):
            idx_x = k * self.n_states + 0
            idx_y = k * self.n_states + 1
            idx_vx = k * self.n_states + 2
            idx_vy = k * self.n_states + 3
            idx_theta = k * self.n_states + 4
            idx_vtheta = k * self.n_states + 5
            
            lbx[idx_x, 0] = -np.inf
            ubx[idx_x, 0] = np.inf
            
            lbx[idx_y, 0] = -10.0  # Don't go too far from road
            ubx[idx_y, 0] = 10.0
            
            lbx[idx_vx, 0] = 0.1   # Forward motion only
            ubx[idx_vx, 0] = 15.0  # Max speed 15 m/s  around 54 km/h
            
            lbx[idx_vy, 0] = -5.0  # Some lateral sliding allowed
            ubx[idx_vy, 0] = 5.0
            
            lbx[idx_theta, 0] = -np.pi  # Any heading allowed
            ubx[idx_theta, 0] = np.pi
            
            lbx[idx_vtheta, 0] = -2.0   # Angular velocity limits
            ubx[idx_vtheta, 0] = 2.0
        
        # Control bounds for all timesteps
        for k in range(self.N):
            idx_ax = self.n_states * (self.N + 1) + k * self.n_controls + 0
            idx_delta = self.n_states * (self.N + 1) + k * self.n_controls + 1
            
            lbx[idx_ax, 0] = -5.0    # Max deceleration: -5 m/s²
            ubx[idx_ax, 0] = 3.0     # Max acceleration: 3 m/s²
            
            lbx[idx_delta, 0] = -np.pi / 6  # Max steering: ±30°
            ubx[idx_delta, 0] = np.pi / 6
        
        # Constraint bounds (all equality constraints = 0)
        lbg = np.zeros((g.numel(), 1))
        ubg = np.zeros((g.numel(), 1))
        
        # build and combine nlp Nonlinear Programming problem solver
        # We're using IPOPT solver (interior point optimizer)
        
        nlp_prob = {
            'f': obj,              # Objective minimize cost
            'x': ca.vertcat(ca.reshape(self.X, -1, 1),
                           ca.reshape(self.U, -1, 1)),
            'g': g,                # Constraints must equal zero
            'p': self.P            # Parameters initial state + references
        }
        
        # Solver options
        opts = {
            'ipopt.max_iter': 1000,          # Max iterations
            'ipopt.acceptable_tol': 1e-8,
            'ipopt.acceptable_obj_change_tol': 1e-6,
            'ipopt.print_level': 0,          # Mute IPOPT iterations output
            'print_time': 0                  # Mute compiler timing prints
        }
        
        print("[MPC] Compiling IPOPT solver")
        self.solver = ca.nlpsol('solver', 'ipopt', nlp_prob, opts)
        print("[MPC] Solver compiled successfully")
        
        # This dictionary is reused every frame to avoid memory allocation
        
        self.args = {
            'lbx': lbx,
            'ubx': ubx,
            'lbg': lbg,
            'ubg': ubg,
            'p': np.zeros((self.P.numel(), 1))
        }
        
        # Initialize decision variables with reasonable values
        self.X0 = np.zeros((self.n_states, self.N + 1))
        self.U0 = np.zeros((self.n_controls, self.N))
        
        print("[MPC] Controller ready")




    
    def compute_control(self, current_state, waypoints, obstacles, target_speed = 10.0):
        """
        current_state: [x, y, vx, vy, theta, vtheta] 
        waypoints: list of reference (x, y, vx) tuples for next N steps
        obstacles: list of {"x": obs_x, "y": obs_y, "length": L, "width": W}

        prepares the parametor for vector
        calls the solver
        extracts and returns final control actions

        -1 throttle is full brake and +1 is full throttle
        steering is in radians, negative is left and positive is right
        
        Returns: [throttle, steering_angle]
        """

        P = np.zeros((self.P.numel(), 1))
        
        P[0:6, 0] = current_state.flatten()

        # refrence traj
        # if waypoints are shorter than N repeat the last one

        waypoints_extended = list(waypoints)
        while len(waypoints_extended) < self.N:
            waypoints_extended.append(waypoints_extended[-1] if waypoints else (0, 0, target_speed))

        for k in range(self.N):
            x_ref, y_ref, vx_ref = waypoints_extended[k]
            
            # Reference state: [x_ref, y_ref, vx_ref, 0, 0, 0]
            # (We don't care about vy, theta, vtheta just position and speed)
            P[6 + 8*k + 0, 0] = x_ref
            P[6 + 8*k + 1, 0] = y_ref
            P[6 + 8*k + 2, 0] = vx_ref
            P[6 + 8*k + 3, 0] = 0.0      # vy_ref = 0
            P[6 + 8*k + 4, 0] = 0.0      # theta_ref = 0
            P[6 + 8*k + 5, 0] = 0.0      # vtheta_ref = 0
            
            # Reference control (typically zero for smooth motion)
            P[6 + 8*k + 6, 0] = 0.0      # ax_ref = 0
            P[6 + 8*k + 7, 0] = 0.0      # delta_ref = 0
        
        # add obstacle control later using the 3d detections from the lidar point cloud
        # (Constraints would be added to self.solver dynamically)
        
        # Initial guess use previous solution or zeros
        x0 = np.vstack((
            ca.reshape(self.X0, -1, 1),
            ca.reshape(self.U0, -1, 1)
        ))
        
        # Call IPOPT solver
        sol = self.solver(x0=x0, lbx=self.args['lbx'], ubx=self.args['ubx'],
                         lbg=self.args['lbg'], ubg=self.args['ubg'], p=P)
        
        # Unpack: [X_0, X_1, ..., X_N, U_0, U_1, ..., U_N-1]
        x_opt = np.array(sol['x']).flatten()
        
        # Extract U (controls)
        U_opt = x_opt[self.n_states * (self.N + 1):]
        U_opt = U_opt.reshape((self.N, self.n_controls))
        
        # Extract X (trajectory)
        X_opt = x_opt[0:self.n_states * (self.N + 1)]
        X_opt = X_opt.reshape((self.N + 1, self.n_states))
        
        # Save for warm-starting next solve
        self.X0 = X_opt.T
        self.U0 = U_opt
        
        ax_mpc = U_opt[0, 0]      # Acceleration (m/s²)
        delta_mpc = U_opt[0, 1]   # Steering angle (radians)
        
        # Convert acceleration to throttle/brake
        # Positive ax → throttle
        # Negative ax → brake
        if ax_mpc >= 0:
            throttle = np.clip(ax_mpc / 3.0, 0.0, 1.0)  # Max accel = 3 m/s²
            brake = 0.0
        else:
            throttle = 0.0
            brake = np.clip(-ax_mpc / 5.0, 0.0, 1.0)    # Max decel = -5 m/s²
        
        # Convert MPC mathematical steering to game steering
        # In mathematical model (delta_mpc > 0) means CCW rotation = Left Turn
        # In driving simulators, steering < 0 is Left Turn, steering > 0 is Right Turn
        # So we MUST invert delta_mpc when passing to the simulator!
        steering = np.clip(-delta_mpc / (np.pi / 6), -1.0, 1.0)  # ±30 max
        
        if self.verbose:
            print(f"[MPC] ax={ax_mpc:.2f}, delta={delta_mpc:.2f} " 
                  f"→ throttle={throttle:.2f}, steering={steering:.2f}")
        
        return throttle, steering
