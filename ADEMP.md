## ADEMP document

### Aims
The goal of the simulations is to investigate the power of tests controlling the FDR agains other Bonferroni-type procedures controlling the FWER in multiple hypothesis testing.

The authors state that, given the FDR is strictly less than FWER when there is at least one non-true null hypothesis, the power of a method controlling FDR is supposed to be higher thant that of a method controlling FWER.  

Simulations are used to investigate this property and get a sense of the magnitude of the gain in power relative to the number of hypothesis tested (and the number of true nulls).

### Data Generating Mechanism 
Simulations are designed to test families of hypothesis of the mean of $m$ independent normally distributed random variables with mean zero and variance one.

The experiments are performend by varying the number of hypotheses across multiple levels $m = 4, 8, 16, 32, 64$ and the proportion of true nulls $3m/4, m/2, m/4, 0$. 
Moreover we use 4 non-zero mean levels: $L/4, L/2, 3L/4, L$ with two choices $L=5, L=10$ . The number of hypothesis in each level is decided according to three different schemes
a) equal (E): each of the fours group gets the same number of non-nulls (e.g. if 16 non nulls, 4 to each)
b) decreasing (D): the number of non-nulls decreases as the signal increases (i.e. more $L/4$ than $L$)
c) increasing (I): the number of non-nulls increases as the signal increases (i.e. more $L$ than $L/4$)

Variation of factors is handles in a factorial manner by testing all combinations of these specifications with 20,000 repetitions each.
Within each of these the non-zero mean levels remain fixed.

To incorporate dependence we draw use the same random draws across different methods inducing positive correlation that reduces the variance of comparisons.

### Estimands and other targets
Our target is the power of the test:
$$
\mathbb{P}(\text{Reject }H_0| H_1 \text{ is true})
$$
In this case the authors decide to use a z-test which is constructed by computing:
$$
z = \frac{\hat{\theta}-\theta_0}{SE(\hat{\theta})}
$$
which under the null hypothesis has standard Gaussian distribution $\mathcal{N}(0, 1)$.


### Methods
The FDR controlling methods is compared with:
- Bonferroni method: reject $H_i$ if $p_i \le \alpha/m$
- Hochberg (1988): let $k$ be the largest $i$ for which $P_{(i)} \le 1/(m+1-i) \alpha$. Then reject all $H_{(i)}$ $i=1, \dots, k$

### Performance measures
Performance measure is the percentage of wrongly rejected null hypothesis. 
Together with this estimate we report also the monte carlo standard errors.