"""
Created on 11/8/24

@author: madeleine
"""

import numpy as np
from scipy.special import gamma
import matplotlib.pyplot as plt

#define mass functions
def lognormal_mass_function(m):
    """
    Lognormal mass function P(m, σc, Mc)
    Parameters:
        m: Mass (scalar or array)
        sigma_c: Width of the mass spectrum
        M_c: Peak mass
    Returns:
        P(m): Lognormal mass function
    """
    sigma_c = 0.4 #width of peak
    M_c = 100 #center value
    #lognormal = 0 if m<0, but clearly we have no negative masses in our range
    result = (1 / (np.sqrt(2 * np.pi) * sigma_c * m)) * np.exp(-(np.log(m / M_c)**2) / (2 * sigma_c**2))
    return result

def power_law_mass_function(m):
    """
    Power-law mass function P(m, Mmin)
    Parameters:
        m: Mass (scalar or array)
        M_min: Minimum mass cut-off
    Returns:
        P(m): Power-law mass function
    """
    M_min = (5.02765 * 10**(-18))
    result = np.where(
        m >= M_min,
        (1 / (2 * np.sqrt(M_min))) * m ** (-3 / 2),
        0 ) # Θ(m - Mmin) = 0 when m < Mmin, = 1 when m >= M_min
    return result

def critical_collapse_mass_function(m):
    """
    Critical collapse mass function P(m, α, Mf)
    Parameters:
        m: Mass, m1 and m2
        alpha: Universal exponent
        M_f: Mass scale
    Returns:
        P(m): Critical collapse mass function
    """
    alpha = 0.35 # universal exponent related to critical collapse of radiation
    M_f = 3.884 * 10**(-18)
    # P(m) undefined for m <= 0, bit our range is m>=0
    result = (alpha**2 / (M_f**(1 + alpha) * gamma(1 / alpha))) * m**alpha * np.exp(-(m / M_f)**alpha)
    return result

#plotting:)
m = np.linspace(10**(-25),10**(-15), 10000) #include mass from lowest (asteroid range, 10^16g) to highest (intermediate range, 10^3 solar masses)
plt.figure(figsize=(12, 8))

lognormal_values = lognormal_mass_function(m)
power_law_values = power_law_mass_function(m)
critical_collapse_values = critical_collapse_mass_function(m)

# Plotting
plt.figure(figsize=(12, 8))
plt.plot(m, lognormal_values, label="Lognormal", lw=2)
plt.plot(m, power_law_values, label="Power-law", lw=2)
plt.plot(m, critical_collapse_values, label="Critical Collapse", lw=2)

# Configure the plot
plt.yscale("log")
plt.xlabel("Mass m in Solar Masses", fontsize=14)
plt.ylabel("Probability Density P(m)", fontsize=14)
plt.title("Mass Functions", fontsize=16)
plt.legend(fontsize=12)
plt.grid(which="both", linestyle="--", linewidth=0.5)
plt.tight_layout()

# Show the plot
plt.show()