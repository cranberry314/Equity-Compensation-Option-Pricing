# -*- coding: utf-8 -*-
"""
Created 3.15.2023
Options Pricing

@author: andrew finn
MIT Open Source License
"""
import pandas as pd
import numpy as np
from scipy.stats import norm
import math

# Import data and format
# file path
file_path = 'C:/Option_Pricing'

# file name
file_name = 'Option_Data_20221231.txt'

# read file
options_df = pd.read_csv(file_path + "/" + file_name, sep='\t')

# test on 1 company
# options_df = options_df.loc[options_df['$ticker'] == 'SWKS']

'''
Parameters:
S: current price of the underlying asset, 'PRCCM - Price - Close - Monthly - USD'
K: strike price of the option, 'OPTPRCWA - Options Exercisable - Weighted Avg Price - INDL - NA - USD'
T: time to expiration in years, 'OPTLIFE - Life of Options - Assumption (# yrs) - INDL - NA - USD'
r: risk-free interest rate, 'OPTRFR - Risk Free Rate - Assumption (%) - INDL - NA - USD'/100
sigma: volatility of the underlying asset, 'OPTVOL - Volatility - Assumption (%) - INDL - NA - USD'/100
'''


########################
# Black Scholes
# Option Pricing
########################
def black_scholes(S, K, T, r, sigma):
    lnS_K = np.log((S/K))
    r_sigma2 = r + (np.square(sigma)/2)
    d_1 = (lnS_K + (r_sigma2 * T)) / (sigma * np.sqrt(T))
    d_2 = d_1 - (sigma * np.sqrt(T))
    
    # Option value
    bsm_option_value = (S * norm.cdf(d_1)) - ( K * np.exp(-r * T) * norm.cdf(d_2))
    return bsm_option_value


options_df['bsm_option_value'] = options_df.apply(lambda x: black_scholes(x['PRCCM - Price - Close - Monthly - USD'],
                                                                                 x['OPTPRCWA - Options Exercisable - Weighted Avg Price - INDL - NA - USD'],
                                                                                 x['OPTLIFE - Life of Options - Assumption (# yrs) - INDL - NA - USD'],
                                                                                 x['OPTRFR - Risk Free Rate - Assumption (%) - INDL - NA - USD']/100,
                                                                                 x['OPTVOL - Volatility - Assumption (%) - INDL - NA - USD']/100), axis=1)

########################
# Binomial
# Option Pricing
########################
def binomial_pricing(S, K, T, r, sigma):
    steps = 200
    dt = T/steps
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    p = (np.exp(r*dt) - d) / (u -d)
    C=0
    for k in reversed(range(steps+1)):
        p_star = math.comb(steps, k)*p**k *(1-p)**(steps-k)
        ST = S * u**k * d ** (steps-k)
        C += max(ST-K,0)*p_star
        binomial_value = np.exp(-r*T)*C
    return binomial_value
    
options_df['binomial_option_value'] = options_df.apply(lambda x: binomial_pricing(x['PRCCM - Price - Close - Monthly - USD'],
                                                                                 x['OPTPRCWA - Options Exercisable - Weighted Avg Price - INDL - NA - USD'],
                                                                                 x['OPTLIFE - Life of Options - Assumption (# yrs) - INDL - NA - USD'],
                                                                                 x['OPTRFR - Risk Free Rate - Assumption (%) - INDL - NA - USD']/100,
                                                                                 x['OPTVOL - Volatility - Assumption (%) - INDL - NA - USD']/100), axis=1)


########################
# Monte Carlo
# Option Pricing
########################
def monte_carlo_pricing(S, K, T, r, sigma):
    steps = 200 # time steps
    N = 1000 # number of trials    
    dt = T/steps
    ST = np.log(S) +  np.cumsum(((r - sigma**2/2)*dt + sigma*np.sqrt(dt) * np.random.normal(size=(steps,N))),axis=0)
    payoffs = np.maximum(np.exp(ST)[-1]-K, 0)
    
    # Option value
    option_price = np.mean(payoffs)*np.exp(-r*T)
    return option_price

options_df['mcmc_option_value'] = options_df.apply(lambda x: monte_carlo_pricing(x['PRCCM - Price - Close - Monthly - USD'],
                                                                                 x['OPTPRCWA - Options Exercisable - Weighted Avg Price - INDL - NA - USD'],
                                                                                 x['OPTLIFE - Life of Options - Assumption (# yrs) - INDL - NA - USD'],
                                                                                 x['OPTRFR - Risk Free Rate - Assumption (%) - INDL - NA - USD']/100,
                                                                                 x['OPTVOL - Volatility - Assumption (%) - INDL - NA - USD']/100), axis=1)


THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
