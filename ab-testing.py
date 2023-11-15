# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 05:50:44 2023

@author: domingosdeeulariadumba
"""
# %%

# %% 
""" IMPORTING LIBRARIES """
from scipy.stats import norm, binom
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# %%

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
        - power defines the statistical power of our test and measures the 
        ability of rejecting the Null Hypothesis when it is false. A 
        recommended value for power is 80%.          
    '''
mde, bcr, alpha, power = 0.04, 0.12, 0.05, 0.8


     '''
     Defining the principal parameters:
         - H0: (there is no difference between the two variants, p_B-p_A)
         d = 0.
         - Ha: (there is a difference between the two variants) d ≠ 0.          
     '''

# %% 

# %% 
""" IMPLEMENTING THE EXPERIMENT"""
    
'''
    The prior parameters allow us to define the ideal size for both, control
    and test groups. Using Evan Miller Sample Size Calculator, we have 1078 
    as the sample size per variant.
    
    As we do not have real data, we have created a function to generate ideal
    data for our test.    
    '''

results_A, results_B = np.random.binomial(size = 1078,
                                          n = 1, p = 0.603), np.random.binomial(size = 1078,
                                                                                n = 1, p = 0.701)
    '''
    Calculating the sample size of the experiment.
    '''

N = 2*len(results_A)

    '''
    Function to store the results in a dataframe.
    '''
def create(results_A, results_B, N):
    
    n_A, n_B, N_total = len(results_A), len(results_A), 2*len(results_A)
    
    df_results = pd.concat([pd.DataFrame({'group': 'control',
                                          'result': results_A}),
                            pd.DataFrame({'group': 'treatment',
                                          'result': results_B})]).sample(frac = 1)
    return df_results

    '''
    Calling the function above to store the different results of the experiment
    '''
df_results = create(results_A, results_B, N)

table = df_results.pivot_table(values='result', index = 'group',
                               aggfunc = np.sum)

table['sample size'] = df_results.pivot_table(values = 'result', index = 'group',
                                        aggfunc = lambda size: len(size))

table['conversion_rate'] = df_results.pivot_table(values='result',
                                                  index='group').round(3)
# %% 

# %% 
""" ANALYSIS OF THE RESULTS """


def get_results(table_results):
    
    # Extracting results from the experiment dataframe
    x_A = table['result'][0]
    p_A = table['conversion_rate'][0]
    x_B = table['result'][1]
    p_B = table['conversion_rate'][1]
    n_A = table['sample size'][0]
    n_B = table['sample size'][1]
    N_total = n_A + n_B
    
    # Pooled Probability
    p = (x_A + x_B) / N_total
    q = 1-p
    
    # Pooled Standard Error
    SE = np.sqrt(p*q*((1/n_A)+(1/n_B)))
    
    # Difference between test and control groups
    d_hat = p_B-p_A
    
    return SE, d_hat

    '''
    From the function above, we'll take the pooled standard error, and the 
    CRT difference between the two versions. Additionally, we'll find the 
    confidence intervals and make a decision regarding this experiment.   
    '''
SE, d_hat = get_results(table)

null_dist, alt_dist = norm(0, 1), norm(d_hat, SE)
Z_alpha = null_dist.ppf(1-alpha/2)

lower_bound, upper_bound = d_hat-Z_alpha*SE, d_hat+Z_alpha*SE

ci_upper = Z_alpha*SE

    '''
    Condition to launch or not the B version.
    '''
if ci_upper <= lower_bound >= mde:
    print('Launch the B version!')
else:
    print('Please, heck the results \
          whether it is to not launch or make additional research.')
    
    '''
    - The table presented a difference of 0.084 in terms of conversion rate 
    between the varant B and variant A. The least desired change was 0.04.
    Despite this result, we still needed to evaluate the significance level, 
    that is the zscore for alpha equal to 5% (+/- 1.96, approximating this
    distribution to a normal distribution).
    - Lastly we checked if the minimum lift (lower_bound) of the experiment was 
    greater or equal than the Minimum Detectable Effect, to assure the margin 
    of the difference was also practically significant. The lower bound is 0.45
    whereas the mde defined was 0.040. Therefore, it is recommended to launch 
    the version B.
    To make this more clear, below we plot all the parameters considered for
    the decision of this experiment.     
    '''
x = np.linspace(-12 * SE, 12 * SE, 1000)
y = norm(0, SE).pdf(x)
x_1 = np.linspace(d_hat - 12 * SE, d_hat + 12 * SE, 1000)
y_1 = norm(d_hat, SE).pdf(x_1)
diff = np.linspace(-Z_alpha*SE, Z_alpha*SE)
y_diff = norm(0, SE).pdf(diff)

fig, ax = plt.subplots(figsize=(12, 6))
plt.axvline(lower_bound, color='g', linestyle='-.', label = 'Minimum Lift')
plt.axvline(mde, -mde, color='k', linestyle='-.', label = 'MDE')
ax.fill_between(x_1, 0, alt_dist.pdf(x_1), color='g', alpha=0.25,
                        where=(x_1 > lower_bound))
ax.fill_between(diff, y_diff, null_dist.pdf(diff), color='r', alpha=0.25)
ax.plot(x, y, label = 'Null Hypothesis', color = 'r')
ax.plot(x_1, y_1, label='Alternative Hypothesis', color = 'g')
ax.plot(diff, y_diff, label = 'Rejection Area', color = 'thistle')
ax.legend()
plt.title('Experiment Results', fontweight = 'bold', fontsize = 14)
# %%      
                  ________  ________   _______   ______ 
                 /_  __/ / / / ____/  / ____/ | / / __ \
                  / / / /_/ / __/    / __/ /  |/ / / / /
                 / / / __  / /___   / /___/ /|  / /_/ / 
                /_/ /_/ /_/_____/  /_____/_/ |_/_____/  

# %%                                     