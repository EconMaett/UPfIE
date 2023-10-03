import wooldridge as woo
import pandas as pd
import statsmodels.formula.api as smf
import linearmodels as plm

jtrain = woo.dataWoo('jtrain')
jtrain['entity'] = jtrain['fcode']
jtrain = jtrain.set_index(['fcode', 'year'])

# Manual computation of deviations of entity means:
jtrain['lscrap_w'] = jtrain['lscrap'] - jtrain.groupby('fcode').mean()['lscrap']
jtrain['d88_w'] = jtrain['d88'] - jtrain.groupby('fcode').mean()['d88']
jtrain['d89_w'] = jtrain['d89'] - jtrain.groupby('fcode').mean()['d89']
jtrain['grant_w'] = jtrain['grant'] - jtrain.groupby('fcode').mean()['grant']
jtrain['grant_1_w'] = jtrain['grant_1'] - jtrain.groupby('fcode').mean()['grant_1']

# manual FE model estimation:
results_man = smf.ols(formula='lscrap_w ~ 0 + d88_w + d89_w + grant_w + grant_1_w', data=jtrain).fit()
table_man = pd.DataFrame({'b': round(results_man.params, 4),
                          'se': round(results_man.bse, 4),
                          't': round(results_man.tvalues, 4),
                          'pval': round(results_man.pvalues, 4)})
print(f'table_man: \n{table_man}\n')

# automatic FE model estimation:
reg_aut = plm.PanelOLS.from_formula(formula='lscrap ~ d88 + d89 + grant + grant_1 + EntityEffects', data=jtrain)
results_aut = reg_aut.fit()
table_aut = pd.DataFrame({'b': round(results_aut.params, 4),
                          'se': round(results_aut.std_errors, 4),
                          't': round(results_aut.tstats, 4),
                          'pval': round(results_aut.pvalues, 4)})
print(f'table_aut: \n{table_aut}\n')
