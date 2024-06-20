import numpy as np
import matplotlib.pyplot as plt

# Kalman-Bucy Filter Implementation
class KalmanBucyFilter:
    def __init__(self, A, B, H, Q, R):
        self.A = A
        self.B = B
        self.H = H
        self.Q = Q
        self.R = R
        self.P = np.eye(A.shape[0])
        self.x = np.zeros((A.shape[0], 1))
    
    def predict(self, u, dt):
        self.x = self.x + (self.A @ self.x + self.B @ u) * dt
        self.P = self.P + (self.A @ self.P + self.P @ self.A.T + self.Q) * dt
    
    def update(self, z):
        K = self.P @ self.H.T @ np.linalg.inv(self.H @ self.P @ self.H.T + self.R)
        self.x = self.x + K @ (z - self.H @ self.x)
        self.P = self.P - K @ self.H @ self.P
    
    def step(self, u, z, dt):
        self.predict(u, dt)
        self.update(z)
        return self.x

# Stratonovich Filter Implementation
class StratonovichFilter:
    def __init__(self, f, h, Q, R):
        self.f = f
        self.h = h
        self.Q = Q
        self.R = R
        self.P = np.eye(len(Q))
        self.x = np.zeros((len(Q), 1))
    
    def predict(self, u, dt):
        self.x = self.x + self.f(self.x, u) * dt
        F = np.eye(len(self.x)) + dt * self._jacobian_f(self.x, u)
        self.P = F @ self.P @ F.T + self.Q * dt
    
    def update(self, z):
        H = self._jacobian_h(self.x)
        K = self.P @ H.T @ np.linalg.inv(H @ self.P @ H.T + self.R)
        self.x = self.x + K @ (z - self.h(self.x))
        self.P = self.P - K @ H @ self.P
    
    def step(self, u, z, dt):
        self.predict(u, dt)
        self.update(z)
        return self.x
    
    def _jacobian_f(self, x, u):
        return np.array([[0, 1], [-1, np.cos(x[1, 0])]])
    
    def _jacobian_h(self, x):
        return np.array([[2*x[0, 0], 0]])

# Define the system parameters
A = np.array([[0, 1], [-1, -0.5]])
B = np.array([[0], [1]])
H_kf = np.array([[1, 0]])
Q = np.array([[0.1, 0], [0, 0.1]])
R = np.array([[0.1]])

def f(x, u):
    return np.array([[x[1, 0]], [-x[0, 0] + np.sin(x[1, 0])]])

def h(x):
    return np.array([[x[0, 0]**2]])

H_sf = np.array([[1, 0]])  # Observation matrix for Kalman-Bucy filter

# Initialize filters
kf = KalmanBucyFilter(A, B, H_kf, Q, R)
sf = StratonovichFilter(f, h, Q, R)

# Simulate a scenario
u = np.array([[0]])
dt = 0.1
states_kf = []
states_sf = []
true_states = []
measurements = []
time = np.arange(0, 10, dt)

for t in time:
    true_state = np.array([[np.sin(t)], [np.cos(t)]])
    z_kf = H_kf @ true_state + np.random.normal(0, np.sqrt(R[0, 0]), (1, 1))
    z_sf = h(true_state) + np.random.normal(0, np.sqrt(R[0, 0]), (1, 1))
    
    kf_state = kf.step(u, z_kf, dt)
    sf_state = sf.step(u, z_sf, dt)
    
    states_kf.append(kf_state)
    states_sf.append(sf_state)
    true_states.append(true_state)
    measurements.append(z_kf)

# Convert lists to numpy arrays for easier plotting
states_kf = np.hstack(states_kf)
states_sf = np.hstack(states_sf)
true_states = np.hstack(true_states)
measurements = np.hstack(measurements)

# Plot the results
plt.figure(figsize=(15, 10))

plt.subplot(2, 1, 1)
plt.plot(time, true_states[0, :], label='True State 1', color='black')
plt.plot(time, states_kf[0, :], label='Kalman-Bucy Estimate 1', color='blue')
plt.plot(time, states_sf[0, :], label='Stratonovich Estimate 1', color='red')
plt.scatter(time, measurements[0, :], label='Measurements', color='green', s=10)
plt.xlabel('Time')
plt.ylabel('State 1')
plt.legend()
plt.title('Comparison of Kalman-Bucy and Stratonovich Filters (State 1)')

plt.subplot(2, 1, 2)
plt.plot(time, true_states[1, :], label='True State 2', color='black')
plt.plot(time, states_kf[1, :], label='Kalman-Bucy Estimate 2', color='blue')
plt.plot(time, states_sf[1, :], label='Stratonovich Estimate 2', color='red')
plt.xlabel('Time')
plt.ylabel('State 2')
plt.legend()
plt.title('Comparison of Kalman-Bucy and Stratonovich Filters (State 2)')

plt.tight_layout()
plt.show()

