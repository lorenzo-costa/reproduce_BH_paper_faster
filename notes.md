Changes made:
- before: pandas concat of each out df, replaced with list append and pandas concat at the end
- before: 2* (1-cdf(zscores)) replaced with erf function (C optimised) -> from 89.6 seconds to 56.95 on 2k sims
- numba optimised functions for metric computation -> from 56.95 to 27.68 
(note numba slower at the beginning because of compilation then faster)
interestingly seems to be slower when parallelised, wonder if it's because of small data.
nvm it is slower when using imap but faster when using map