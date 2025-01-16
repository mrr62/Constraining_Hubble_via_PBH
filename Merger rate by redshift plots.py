"""
Created on 11/8/24

@author: madeleine
"""
import scipy.integrate as integrate
import numpy as np
from scipy.special import gamma
import matplotlib.pyplot as plt

#define constants
omega_m = 0.27 #matter density parameter
"""H0 time units don't matter, they will cancel out from t(z)/t(0) in merger rates, so I used s^-1 since we originally have km/s/Mpc"""
H0_SN = 72 * 1/(3.085677581 * 10**19) #Supernova H0 in s^-1
H0_CMB = 67.5 * 1/(3.085677581 * 10**19) #Cosmic Microwave Background H0 in s^-1
H0_GW = 68 * 1/(3.085677581 * 10**19)#Gravitational Wave H0 in s^-1
sigma = 0.005 #variance of density perturbations of the rest of dark matter at radiation-matter equality


#define cosmic time:
def E_z(z1, omega_m): #this defines E(z) fct found in cosmic time integral
    return np.sqrt(omega_m * (1+z1)**3 + (1-omega_m))

def cosmic_time(z, H0, omega_m): #defines cosmic time t(z) as a fct of redshift z
    integrand = lambda z1: 1 / (E_z(z1, omega_m) * (1+z1)) #defines function inside integral
    integral, error =  integrate.quad(integrand, z, np.inf) #integrate function above
    print(f"Cosmic time integral for z={z}, H0={H0}: {integral}")
    return integral / H0


#define merger rate function
def merger_rate(m1, m2, z, f_pbh, P_m1, P_m2, sigma):
    t_z = cosmic_time(z, H0_value, omega_m)
    t_0 = cosmic_time(0, H0_value, omega_m)
    normalization = 10**(-6)
    # Add debug prints
    print(f"\nDebug merger_rate calculation:")
    print(f"m1: {m1:.2e}, m2: {m2:.2e}, z: {z:.2f}")
    print(f"f_pbh: {f_pbh:.2e}, P_m1: {P_m1:.2e}, P_m2: {P_m2:.2e}")
    print(f"t_z: {t_z:.2e}, t_0: {t_0:.2e}")
    cosmic_time_term = (t_z / t_0)**(-34/37) #cosmic time term (where H0 shows up)
    abundance_term = (f_pbh**2) * ((0.7 * f_pbh**2) + (sigma**2))**(-21/74) #abundance term
    mass_fct_term =min(P_m1 / m1, P_m2 / m2) * ((P_m1 / m1) + (P_m2 / m2)) # mass fct and min function term
    mass_term = (m1 * m2)**(1/37) * (m1 + m2)**(36/37)
    #debug intermediate terms
    print(f"cosmic_time_term: {cosmic_time_term:.2e}")
    print(f"abundance_term: {abundance_term:.2e}")
    print(f"mass_fct_term: {mass_fct_term:.2e}")
    print(f"mass_term: {mass_term:.2e}")
    # Print progressive multiplication
    result = 2.8 * 10 ** 6
    print(f"\nProgressive multiplication:")
    print(f"Initial (2.8 * 10^6): {result:.2e}")

    result *= cosmic_time_term
    print(f"After cosmic_time_term: {result:.2e}")

    result *= abundance_term
    print(f"After abundance_term: {result:.2e}")

    result *= mass_fct_term
    print(f"After mass_fct_term: {result:.2e}")

    result *= mass_term
    print(f"Final result: {result:.2e}")
    return .8 * 10**6 * cosmic_time_term * abundance_term * mass_fct_term * mass_term
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
    sigma_c = 0.4 #value given on pg 10 [] <-- ADD SOURCE NUMBER
    M_c = 100 #given in source
    if m<=0:  # lognormal undefined for m <= 0
        return 0
    result = (1 / (np.sqrt(2 * np.pi) * sigma_c * m)) * np.exp(-(np.log(m / M_c)**2) / (2 * sigma_c**2))
    print(f"Lognormal mass function for m={m:.2e}: {result:.2e}")
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
    if m < M_min:  # Θ(m - Mmin) = 0 when m < Mmin, = 1 when m >= M_min
        return 0
    result = (1 / (2 * np.sqrt(M_min))) * m**(-3 / 2)
    print(f"Power law mass function for m={m:.2e}: {result:.2e}")
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
    if m <= 0:  # P(m) undefined for m <= 0
        return 0
    result = (alpha**2 / (M_f**(1 + alpha) * gamma(1 / alpha))) * m**alpha * np.exp(-(m / M_f)**alpha)
    print(f"Critical collapse mass function for m={m:.2e}: {result:.2e}")
    return result

#monochromatic mass function considers all PBH to be approximately the same mass, therefore when normalized it will just be 1 :)
def monochromatic_mass_function(m):
    return 1

#define abundance percentages f_pbh
def f_pbh1(m):
    if m > (6 * 10**(-8)) and m <= (10**(-3)):
        return 1
    elif m > (10**(-3)) and m < (0.4):
        return 0.05
    elif m >= (0.4) and m < (1):
        return 0.1
    else:
        if m > 0:  # this is arbitrary bound, just needed to make sure all my data would be included
            return 0.25  # this is a theorized value that seems to be independent of mass

def f_pbh_constant(m):
    if m > 0: #this is arbitrary bound, just needed to make sure all my data would be included
        return 0.25 # this is a theorized value that seems to be independent of mass


#define variables H0,PBH masses m1, m2, mass functs, f_pbh, and z:
H0 = [H0_CMB, H0_GW, H0_SN]
#define masses as dictionary (easier to plot)
# Define masses as a dictionary to ensure all combinations without repeating
masses = {
    "m11": (10**(-17), 10**(-17)),
    "m12": (10**(-17), 5.02765 * 10**(-11)),
    "m13": (10**(-17), 500),
    "m21": (5.02765 * 10**(-11),10**(-17)),
    "m22": (5.02765 * 10**(-11), 5.02765 * 10**(-11)),
    "m23": (5.02765 * 10**(-11), 500),
    "m31": (500, 10**(-17)),
    "m32": (500, 5.02765 * 10**(-11)),
    "m33": (500, 500),
     }
"""3 possible PBH mass windows are 
    1) 10^16 - 10^17 grams (the "asteroid range")
    2) 10^20 - 10^26 grams (the "sublunar range")
    3) 10 - 10^3 solar masses (the "intermediate range")
    (this is, to my understanding, at matter-radiation equality, 
    the size at formation (within the 1st second of the big bang 
    due to quantum fluctutations collapsing) is much smaller and 
    broader, with mass depending on when within the second they are
    created. However, the above values are constrained to those which 
    could contribute meaningfully to CDM, and would not fully 
    dissipate due to hawking radiation by matter-radiation equality) """
#put mass functions into dictionary to loop through easier
mass_functions = {
    "Lognormal": lognormal_mass_function,
    "Power Law": power_law_mass_function,
    "Critical Collapse": critical_collapse_mass_function,
    "Monochromatic": monochromatic_mass_function
}
for mass_key, (m1, m2) in masses.items():
    #f_pbh = [f_pbh1(m1), f_pbh2(m1), f_pbh3(m1), f_pbh4(m1), f_pbh5(m1)]
    f_pbh = f_pbh1(m1)
z_range = np.linspace(0, 1, 100)

# plotting:)
plt.figure(figsize=(12, 8))

# Specify the Hubble constant value you want to use
H0_value = H0_SN  # Choose from H0_SN, H0_CMB, or H0_GW

# Compute Rl and Ru arrays outside the loop
Rl = 10.5494 * (1 + z_range)**2.9  # Lower bound merger rate by redshift z
Ru = 25.9315 * (1 + z_range)**2.9  # Upper bound merger rate by redshift z

# Loop over all mass combinations
for mass_key, (m1, m2) in masses.items():
    f_pbh = f_pbh1(m1)
    print(f"\nTesting {mass_key} mass range:")
    print(f"m1: {m1:.2e}, m2: {m2:.2e}, f_pbh: {f_pbh:.2e}")

    for mass_func_name, mass_func in mass_functions.items():
        # Initialize storage for merger rates
        merger_rates = []
        print(f"\nTesting {mass_func_name} mass function:")

        for z in z_range:
            # Calculate P(m1) and P(m2) using the current mass function
            P_m1 = mass_func(m1)
            P_m2 = mass_func(m2)

            # Ensure we have valid mass function results
            if P_m1 is not None and P_m2 is not None and P_m1 > 0 and P_m2 > 0:
                # Calculate the merger rate
                merger_rate_value = merger_rate(m1, m2, z, f_pbh, P_m1, P_m2, sigma)
                merger_rates.append(merger_rate_value)

        # Only plot if we have merger rates
        if merger_rates:
            plt.semilogy(  # Change to semilogy for logarithmic y-axis
                z_range[:len(merger_rates)],
                merger_rates,
                label=f"{mass_key}, {mass_func_name}",
                alpha=0.7
            )
# Plot the lower and upper bounds
plt.semilogy(z_range, Rl, 'k--', label="Lower Bound R(z)")
plt.semilogy(z_range, Ru, 'k--', label="Upper Bound R(z)")

# Customize the graph
plt.title(f"Merger Rates for All Configurations, H0 = {H0_value:.1e}", fontsize=16)
plt.xlabel("Redshift (z)", fontsize=14)
plt.ylabel("Merger Rate R(z) [log scale]", fontsize=14)
plt.grid(True, which="both", ls="-", alpha=0.2)  # Add grid for both major and minor ticks
plt.legend(fontsize=8, ncol=2, loc="upper left", bbox_to_anchor=(1.05, 1))
plt.tight_layout()

# Show the plot
plt.show()