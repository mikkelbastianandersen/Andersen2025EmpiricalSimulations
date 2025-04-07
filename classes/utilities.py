# utility.py
import numpy as np

def exponential_utility(wealth, esg_impact, risk_aversion):
    return -np.exp(-risk_aversion * (wealth + esg_impact))
