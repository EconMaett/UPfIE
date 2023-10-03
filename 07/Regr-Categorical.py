import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

CPS1985 = pd.read_csv('data/CPS1985.csv')
# rename variable to make outputs more compact:
CPS1985['oc'] = CPS1985['occupation']

# table of categories and frequencies for two categorical variables:
freq_gender = pd.crosstab(CPS1985['gender'], columns='count')
print(f'freq_gender: \n{freq_gender}\n')

freq_occupation = pd.crosstab(CPS1985['oc'], columns='count')
print(f'freq_occupation: \n{freq_occupation}\n')

# directly using categorical variables in regression formula:
reg = smf.ols(formula='np.log(wage) ~ education +'
                      'experience + C(gender) + C(oc)', data=CPS1985)
results = reg.fit()

# print regression table:
table = pd.DataFrame({'b': round(results.params, 4),
                      'se': round(results.bse, 4),
                      't': round(results.tvalues, 4),
                      'pval': round(results.pvalues, 4)})
print(f'table: \n{table}\n')

# rerun regression with different reference category:
reg_newref = smf.ols(formula='np.log(wage) ~ education + experience + '
                             'C(gender, Treatment("male")) + '
                             'C(oc, Treatment("technical"))', data=CPS1985)
results_newref = reg_newref.fit()

# print results:
table_newref = pd.DataFrame({'b': round(results_newref.params, 4),
                             'se': round(results_newref.bse, 4),
                             't': round(results_newref.tvalues, 4),
                             'pval': round(results_newref.pvalues, 4)})
print(f'table_newref: \n{table_newref}\n')
