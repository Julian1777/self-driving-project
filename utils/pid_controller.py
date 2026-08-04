import numpy as np


class PIDController:
    def __init__(self, Kp=0.015, Ki=0.0, Kd=0.025, integral_limit=1.0, Kf=0.0, debug=False):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.Kf = Kf
        self.integral_limit = integral_limit
        self.previous_error = 0.0
        self.integral = 0.0
        self.debug = debug
        
    def update(self, error, dt):
        if dt <= 0:
            dt = 0.01

        if np.sign(error) != np.sign(self.previous_error) and abs(self.previous_error) > 1e-6:
            self.integral = 0.0

        p_term = self.Kp * error

        self.integral += error * dt
        self.integral = np.clip(self.integral, -self.integral_limit, self.integral_limit)
        i_term = self.Ki * self.integral

        d_term = self.Kd * ((error - self.previous_error) / dt)

        f_term = self.Kf * error

        self.previous_error = error

        output = p_term + i_term + d_term + f_term
        
        if self.debug:
            print(f"[PID] error={error:.4f}, dt={dt:.4f}, P={p_term:.4f}, I={i_term:.4f}, D={d_term:.4f}, F={f_term:.4f}, out={output:.4f}")
        
        return output
    
    def reset(self):
        self.previous_error = 0.0
        self.integral = 0.0
