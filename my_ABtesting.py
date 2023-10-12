# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 05:02:08 2023

@author: domingosdeeularia
"""

import numpy as np
import pandas as pd

# %%
""" DESIGNING THE EXPERIMENT """
    
    '''
    Defining the principal parameters:
        - mde is the Minimum Detectable Effect, that is the lower difference 
         between the two versions we are intersted in
        - bcr is the Baseline Conversion Rate or the current results with the
          control version (we assumed it is 0.12).
        - alpha is the significance level, the risk we accept to reject the 
          Null hypothesis (there is no difference between the version A and B)
          when it is true. A good value for alpha is 5%.
        - power defines the statistical power of our test and measures the ability
          of rejecting the Null Hypothesis when it is false. A recommended
          value for power is 80%.          
    '''
mde, bcr, alpha, power = 0.04, 0.12, 0.05, 0.8


     '''
     Defining the principal parameters:
         - H0: (there is no difference between the two variants, p_B-p_A)
         d = 0.
         - Ha: (there is a difference between the two variants
           control version) d ≠ 0.          
     '''

    '''
    The prior parameters allows us to define the ideal size for both, control
    and test groups. Using Evan's Miller Sample Size Calculator, we have 1078 
    as the sample size per variant.
    
    As we do not have real data, we have created a function to generate ideal
    data for our test.    
    '''

results_A, results_B = np.random.binomial(size = 1078,
                                          n = 1, p = 0.603), np.random.binomial(size = 1078,
                                                                                n = 1, p = 0.687)
N = 2*len(results_A)


""" IMPLEMENTING THE EXPERIMENT"""
    
def create(results_A, results_B, N):
    
    n_A, n_B, N_total = len(results_A), len(results_A), 2*len(results_A)
    
    df_results = pd.concat([pd.DataFrame({'group': 'control',
                                          'result': results_A}),
                            pd.DataFrame({'group': 'treatment',
                                          'result': results_B})]).sample(frac = 1)
    return df_results

df_results = create(results_A, results_B, N)

table = df_results.pivot_table(values='result', index = 'group',
                               aggfunc = np.sum)

table['total'] = df_results.pivot_table(values = 'result', index = 'group',
                                        aggfunc = lambda size: len(size))

table['conversion_rate'] = df_results.pivot_table(values='result',
                                                  index='group').round(3)

""" ANALYSIS OF THE RESULTS """


x_A = table['result'][0]
p_A = table['conversion_rate'][0]
x_B = table['result'][1]
p_B = table['conversion_rate'][1]
n_A = table['total'][0]
n_B = table['total'][1]

def pooled_probability(n_A, n_B, x_A, x_B):
    
    return (x_A + x_B) / (n_A + n_B)

p_pool = pooled_probability(n_A, n_B, x_A, x_B)

def pooled_SE(p_pool, n_A, n_B):
    return np.sqrt(p_pool*(1-p_pool)*((1/n_A)+(1/n_B)))

SE = pooled_SE(p_pool, n_A, n_B)

d_hat = p_B - p_A

lower_bound = d_hat-1.96*SE
upper_bound = d_hat+1.96*SE

if 0 < lower_bound <=mde:
    print('Launch the B version!')
else:
    print('Check the results wether it is to not launch or make additional research.')
    
    '''
    - The table presented a difference of 0.076 in terms of conversion rate 
    between the varant B and variant A. The least desired change was 0.04.
    Despite this result, we still needed to evaluate the significance level, 
    that is the zscore for alpha equal to 5% (+/- 1.96, approximating this
    distribution to a normal distribution).
    - Lastly we checked if the Minimum Detectable Effect was in the interval of
    practical and statistical significance. And made the decision to launch 
    the version B.      
    '''

