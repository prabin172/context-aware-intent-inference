# Sub3 original-scope audit

This audit isolates the current cleaned outputs to subject sub3 only.

The purpose is to check whether the original accepted-paper trend still appears under a narrow sub3-style comparison.

## Summary

On sub3, Bayesian Fusion remains the best overall method:

- BF accuracy: 0.9111
- E2E accuracy: 0.8989
- GR accuracy: 0.8576

BF also has the highest macro-F1:

- BF macro-F1: 0.7668
- E2E macro-F1: 0.7558
- GR macro-F1: 0.5043

For early prediction, BF has the largest mean sustained lead time:

- BF: 544.6 ms
- GR: 487.9 ms
- E2E: 414.8 ms

This supports the interpretation that the accepted-paper story is consistent with the narrower sub3-style evaluation, while the expanded full 5-subject LOSO analysis gives a broader and somewhat different result.
