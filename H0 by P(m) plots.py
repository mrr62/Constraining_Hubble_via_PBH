"""
Created on 11/8/24

@author: madeleine
"""

import scipy.integrate as integrate
import numpy as np
import matplotlib.pyplot as plt

#define constants
omega_m = 0.27 #matter density parameter
"""H0 time units don't matter, they will cancel out from t(z)/t(0) in merger rates, so I used s^-1 since we originally have km/s/Mpc"""
H0_SN = 72 # 1/(3.085677581 * 10**19) #Supernova H0 in s^-1
H0_CMB = 67.5 # 1/(3.085677581 * 10**19) #Cosmic Microwave Background H0 in s^-1
H0_GW = 68 # 1/(3.085677581 * 10**19)#Gravitational Wave H0 in s^-1
sigma = 0.005 #variance of density perturbations of the rest of dark matter at radiation-matter equality
z = 0.2
real_merger_lower = 17.9
real_merger_upper = 44

#define cosmic time:
def E_z(z, omega_m): #this defines E(z) fct found in cosmic time integral
    return np.sqrt(omega_m * (1+z)**3 + (1-omega_m))

def cosmic_time(z, omega_m): #defines cosmic time t(z) as a fct of redshift z
    integrand = lambda z1: 1 / (E_z(z1, omega_m) * (1+z1)) #defines function inside integral
    integral, error =  integrate.quad(integrand, z, np.inf) #integrate function above
    return integral #don't divide by H0 since it was multiplied out


#define merger rate function
def H0_values(m1, m2, z, f_pbh, P_m1, P_m2, sigma, real_merger):
    t_z = cosmic_time(z, omega_m)
    t_0 = cosmic_time(0, omega_m)
    cosmic_time_term = (t_z / t_0)**(-34/37) #cosmic time term (where H0 shows up)
    abundance_term = (f_pbh**2) * ((0.7 * f_pbh**2) + (sigma**2))**(-21/74) #abundance term
    mass_fct_term =min(P_m1 / m1, P_m2 / m2) * ((P_m1 / m1) + (P_m2 / m2))
    mass_term = (m1 * m2)**(1/37) * (m1 + m2)**(36/37) #mass term
    real_merger_term = 1/real_merger
    # debug intermediate terms
    print(f"cosmic_time_term: {cosmic_time_term:.2e}")
    print(f"abundance_term: {abundance_term:.2e}")
    print(f"mass_fct_term: {mass_fct_term:.2e}")
    print(f"mass_term: {mass_term:.2e}")
    # Print progressive multiplication
    result = 2.8 * 10 ** 6
    print(f"\nProgressive multiplication:")
    print(f"Initial (2.8 * 10^6): {result:.2e}")
    result *= real_merger_term
    print(f"After real_merger_term: {result:.2e}")
    result *= cosmic_time_term
    print(f"After cosmic_time_term: {result:.2e}")

    result *= abundance_term
    print(f"After abundance_term: {result:.2e}")

    result *= mass_fct_term
    print(f"After mass_fct_term: {result:.2e}")

    result *= mass_term
    print(f"Final result: {result:.2e}")
    print(f"m2 = {m2_val}, H0_results = {result}")
    return (0.8 * 10**6 * real_merger_term * cosmic_time_term * abundance_term * mass_fct_term * mass_term)**(-37/34)

#define PBH abundance
def f_pbh(m):
    if m > (6 * 10**(-8)) and m <= (10**(-3)):
        return 1
    elif m > (10**(-3)) and m < (0.4):
        return 0.05
    elif m >= (0.4) and m < (1):
        return 0.1
    else:
        if m > 0:  # this is arbitrary bound, just needed to make sure all my data would be included
            return 0.25  # this is a theorized value that seems to be independent of mass

#define variables
real_merger = [real_merger_lower, real_merger_upper]
#m2 = [10**(-17), 5.02765 * 10**(-14), 500]
m2 = [1, 5, 100]
m1 = np.logspace(1,100, 1000) #mass spans range of magnitudes, therefore use log scale:)
P_m1 = 1 #monochromatic mass distribution
P_m2 = 1

#plotting:)
plt.figure(figsize=(12, 8))

colors = ['blue', 'green', 'red']
labels = ['Asteroid range (m2=10^-17)',
          'Sublunar range (m2=5.03*10^-11)',
          'Intermediate range (m2=500)']

# Define mass range boundaries
mass_ranges = [
    (5.02765e-18, 5.02765e-17),  # Asteroid range
    (5.02765e-14, 5.02765e-8),   # Sublunar range
    (10, 1e3)                    # Intermediate range
]

# Loop over each m2 and calculate H0
for idx, m2_val in enumerate(m2):
    H0_results = []
    for m1_val in m1:
        f_pbh_val = f_pbh(m1_val)
        # Calculate H0 using the function
        H0 = H0_values(m1_val, m2_val, z, f_pbh_val, P_m1, P_m2, sigma, real_merger_upper)
        H0_results.append(H0)

    # Plot H0 as a function of m1
    plt.plot(m1, H0_results, label=labels[idx], color=colors[idx])

# Add vertical lines and shaded regions for mass ranges
shading_colors = ['lightblue', 'lightgreen', 'lightcoral']
for idx, (start, end) in enumerate(mass_ranges):
    # Vertical lines
    plt.axvline(start, color=shading_colors[idx], linestyle='--', alpha=0.7)
    plt.axvline(end, color=shading_colors[idx], linestyle='--', alpha=0.7)
    # Shaded region
    plt.axvspan(start, end, color=shading_colors[idx], alpha=0.2, label=f'{labels[idx]} Range')

# Add the horizontal lines for H0_SN, H0_CMB, and H0_GW
plt.axhline(H0_SN, color='black', linestyle='--', label='H0 (Supernova)')
plt.axhline(H0_CMB, color='purple', linestyle='--', label='H0 (CMB)')
plt.axhline(H0_GW, color='orange', linestyle='--', label='H0 (Gravitational Waves)')

# Formatting the plot
plt.xscale('log')  # Log scale for mass
plt.yscale('log') #log scale for H0, needed due to python being unable to compute at the higher mass needed to get a valid H0 range :(
plt.xlabel(r'Mass $m_1$ ($M_\odot$)', fontsize=14)
plt.ylabel(r'$H_0$ (kmMpc$^{-1}$s$^{-1}$)', fontsize=14)
#limit H0 axis
plt.title(r'$H_0$ as a Function of $m_1$ for Different $m_2$', fontsize=16)
plt.legend(fontsize=12)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()

# Show the plot
plt.show()