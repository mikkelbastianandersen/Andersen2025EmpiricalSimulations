

import numpy as np
import matplotlib.pyplot as plt
import random
from utility import ExponentialUtility, CobbDouglasUtility


random.seed(42)  # For reproducibility
### Fixed parameters

S_0 = np.array([100, 100])  # Initial prices of the assets
G_0 = np.array([100, 100])  # Initial ESG impact
mu = np.array([0.07, 0.03])  # Expected returns
sigma = np.array([[0.25, 0.0], [0.0, 0.2]])  # Volatility matrix
alpha = np.array([0.01, 0.05])  # Expected ESG impact
beta = np.array([[0.06, 0.0], [0.0, 0.1]])  # Volatility matrix for ESG impact
rho_fixed = np.array([[0.0, 0.0], [0.0, 0.0]])  # Correlation matrix
r = 0.02

X_t = 1  # Initial wealth
Y_t = 1 # Initial ESG impact

# Main simulation function
if __name__ == "__main__":
    ### First explore the exponential utility function
    # Fixed values
    b_fixed = 1
    a_fixed = 1

    # Range of a and b
    a_vals = np.linspace(0.5, 6, 100)
    b_vals = np.linspace(0.5, 6, 100)
    rho_vals = np.linspace(-0.95, 0.95, 100)

    pi1_rho = []
    pi2_rho = []

    # Vary a, fix b
    ## Store results
    pi1_a = []
    pi2_a = []
    for a in a_vals:
        exp_utility = ExponentialUtility(a, b_fixed)
        pi_star = exp_utility.optimal_portfolio(X_t, Y_t, mu, sigma, alpha, beta, rho_fixed, r)
        pi1_a.append(pi_star[0])
        pi2_a.append(pi_star[1])


    # Vary b, fix a
    ## Store results
    pi1_b = []
    pi2_b = []
    for b in b_vals:
        exp_utility = ExponentialUtility(a_fixed, b)
        pi_star = exp_utility.optimal_portfolio(X_t, Y_t, mu, sigma, alpha, beta, rho_fixed, r)
        pi1_b.append(pi_star[0])
        pi2_b.append(pi_star[1])
    
    # Vary rho, fix a and b
    ## Store results
    pi1_rho = []
    pi2_rho = []
    for rho in rho_vals:
        var_rho = np.array(
            [
                [rho, 0],
                [0, rho],
            ]
        )
        exp_utility = ExponentialUtility(a_fixed, b_fixed)
        pi_star = exp_utility.optimal_portfolio(X_t, Y_t, mu, sigma, alpha, beta, var_rho, r)
        pi1_rho.append(pi_star[0])
        pi2_rho.append(pi_star[1])


    # Plotting
    # Plot vs a
    plt.figure(figsize=(6, 5))
    plt.plot(a_vals, pi1_a, label='Asset 1')
    plt.plot(a_vals, pi2_a, label='Asset 2')
    plt.title(f'Optimal Allocation vs a (b = {b_fixed}) - Exponential Utility')
    plt.xlabel('a')
    plt.ylabel('Allocation')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


    # Plot vs b
    plt.figure(figsize=(6, 5))
    plt.plot(b_vals, pi1_b, label='Asset 1')
    plt.plot(b_vals, pi2_b, label='Asset 2')
    plt.title(f'Optimal Allocation vs b (a = {b_fixed}) - Exponential Utility')
    plt.xlabel('b')
    plt.ylabel('Allocation')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # === Second figure: rho plot ===
    plt.figure(figsize=(6, 5))
    plt.plot(rho_vals, pi1_rho, label='Asset 1')
    plt.plot(rho_vals, pi2_rho, label='Asset 2')
    plt.title(f'Optimal Allocation vs ρ (a = {a_fixed}, b = {b_fixed}) - Exponential Utility')
    plt.xlabel('Diagonal value in ρ')
    plt.ylabel('Allocation')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    ### Now explore the Cobb-Douglas utility function

    # Fixed values
    a_fixed = 0.5

    # Range of a and rho
    a_vals = np.linspace(0.05, 0.95, 100)
    rho_vals = np.linspace(-0.95, 0.95, 100)

    # Vary a, fix b
    ## Store results
    pi1_a = []
    pi2_a = []
    for a in a_vals:
        cd_utility = CobbDouglasUtility(a)
        pi_star = cd_utility.optimal_portfolio(X_t, Y_t, mu, sigma, alpha, beta, rho_fixed, r)
        pi1_a.append(pi_star[0])
        pi2_a.append(pi_star[1])
    
    # Vary rho, fix a and b
    ## Store results
    pi1_rho = []
    pi2_rho = []
    for rho in rho_vals:
        var_rho = np.array(
            [
                [rho, 0],
                [0, rho],
            ]
        )
        cd_utility = CobbDouglasUtility(a_fixed)
        pi_star = cd_utility.optimal_portfolio(X_t, Y_t, mu, sigma, alpha, beta, var_rho, r)
        pi1_rho.append(pi_star[0])
        pi2_rho.append(pi_star[1])


    # Plotting
    # Plot vs a
    plt.figure(figsize=(6, 5))
    plt.plot(a_vals, pi1_a, label='Asset 1')
    plt.plot(a_vals, pi2_a, label='Asset 2')
    plt.title(f'Optimal Allocation vs a (Cobb-Douglas)')
    plt.xlabel('a')
    plt.ylabel('Allocation')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # === Second figure: rho plot ===
    plt.figure(figsize=(6, 5))
    plt.plot(rho_vals, pi1_rho, label='Asset 1')
    plt.plot(rho_vals, pi2_rho, label='Asset 2')
    plt.title(f'Optimal Allocation vs ρ (Cobb-Douglas)')
    plt.xlabel('Diagonal value in ρ')
    plt.ylabel('Allocation')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()