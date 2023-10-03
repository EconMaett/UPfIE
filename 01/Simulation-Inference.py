import numpy as np
import scipy.stats as stats

# set the random seed:
np.random.seed(123456)

# set sample size and MC simulations:
r = 10000
n = 100

# initialize arrays to later store results:
CIlower = np.empty(r)
CIupper = np.empty(r)
pvalue1 = np.empty(r)
pvalue2 = np.empty(r)

# repeat r times:
for j in range(r):
    # draw a sample:
    sample = stats.norm.rvs(10, 2, size=n)
    sample_mean = np.mean(sample)
    sample_sd = np.std(sample, ddof=1)

    # test the (correct) null hypothesis mu=10:
    testres1 = stats.ttest_1samp(sample, popmean=10)
    pvalue1[j] = testres1.pvalue
    cv = stats.t.ppf(0.975, df=n - 1)
    CIlower[j] = sample_mean - cv * sample_sd / np.sqrt(n)
    CIupper[j] = sample_mean + cv * sample_sd / np.sqrt(n)

    # test the (incorrect) null hypothesis mu=9.5 & store the p value:
    testres2 = stats.ttest_1samp(sample, popmean=9.5)
    pvalue2[j] = testres2.pvalue

# test results as logical value:
reject1 = pvalue1 <= 0.05
count1_true = np.count_nonzero(reject1)  # counts true
count1_false = r - count1_true
print(f'count1_true: {count1_true}\n')
print(f'count1_false: {count1_false}\n')

reject2 = pvalue2 <= 0.05
count2_true = np.count_nonzero(reject2)
count2_false = r - count2_true
print(f'count2_true: {count2_true}\n')
print(f'count2_false: {count2_false}\n')
