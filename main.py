import numpy as np
import matplotlib.pyplot as plt

# Constants
G = 6.67430e-11  # Gravitational constant in Nm^2/kg^2
m_earth = 5.972e24  # Mass of the Earth in kg
m_satellite = 1000  # Mass of the satellite in kg
r_initial = 6.771e6  # Initial distance from the Earth's center in meters

# Initial conditions
position = np.array([r_initial, 0])  # Initial position (x, y)
velocity = np.array([0, np.sqrt(G * m_earth / r_initial)])  # Initial velocity for circular orbit

# Simulation parameters
dt = 1  # Time step in seconds, reduced for accuracy
total_time = 5400 * 2  # Increasing total simulation time to cover more than one full orbit
steps = int(total_time / dt)

# Function to calculate gravitational acceleration
def acceleration_simple(position):
    r = np.linalg.norm(position)
    return -G * m_earth * position / r**3

# Lists to store positions for plotting
x_positions = []
y_positions = []

# Runge-Kutta 4th order method for numerical integration
for _ in range(steps):
    k1_v = acceleration_simple(position) * dt
    k1_p = velocity * dt
    
    k2_v = acceleration_simple(position + 0.5 * k1_p) * dt
    k2_p = (velocity + 0.5 * k1_v) * dt
    
    k3_v = acceleration_simple(position + 0.5 * k2_p) * dt
    k3_p = (velocity + 0.5 * k2_v) * dt
    
    k4_v = acceleration_simple(position + k3_p) * dt
    k4_p = (velocity + k3_v) * dt
    
    velocity += (k1_v + 2*k2_v + 2*k3_v + k4_v) / 6
    position += (k1_p + 2*k2_p + 2*k3_p + k4_p) / 6
    
    x_positions.append(position[0])
    y_positions.append(position[1])

# Plotting the orbit
plt.figure(figsize=(8, 8))
plt.plot(x_positions, y_positions, label='Satellite Orbit')
plt.plot(0, 0, 'ro', label='Earth')  # Earth's position
plt.xlabel('X Position (m)')
plt.ylabel('Y Position (m)')
plt.title('Adjusted Satellite Orbit Simulation')
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.show()
