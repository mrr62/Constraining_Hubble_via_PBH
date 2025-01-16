"""
Created on 11/17/24

@author: madeleine
"""

import numpy as np
import matplotlib.pyplot as plt

#set bounds on redshift z
z = np.linspace(0,1, 1500)

#define E(z)
def E(z, Omega):
    Ez = np.sqrt(Omega * (1 + z)**3 + (1-Omega))
    return Ez

# Define Omega and H0 values:
Omega_values = [0.23, 0.27, 0.31] #.27 +/- .04
H0_values = [67.5, 68, 72] # CMB, GW, and SN values respectively, might add mroe in between if needed:)

#add in H0 points
points = {
    "H0_CMB": 67.5,
    "H0_GW": 68,
    "H0_SN": 72
}

#plot
plt.figure(figsize=(10, 6))

for H0 in H0_values:
    for Omega in Omega_values:
        H_z = H0 * E(z, Omega)
        label = f"H0 = {H0}, Omega = {Omega}"
        plt.plot(z, H_z, label=label)

# Add labeled points for H(z=0)
for label, value in points.items():
    plt.plot(0, value, 'o', label=label, markersize=8)  # Markersize adjusts point size

# Add labels, legend, and title
plt.xlabel("Redshift (z)")
plt.ylabel("H(z) [km/s/Mpc]")
plt.title("H(z) for Various Omega M and H0 Values")
plt.legend()

# Show the plot
plt.show()