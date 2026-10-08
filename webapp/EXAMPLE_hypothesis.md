Alphabetical proximity

Statement of the hypothesis
Cross-propensity spillover is dominated by the alphabetical proximity of
eval names in our dataset. When a treatment's name shares a leading
letter or short edit distance with an eval's name, the model has been
exposed to similar surface forms during fine-tuning and the learned
representation transfers more strongly. Pairs that are alphabetically
distant should show weaker spillover, regardless of any semantic
relationship between the underlying propensities.

(This is a deliberately weak example to show the expected format. It is
almost certainly wrong, and is provided so participants can see the
structure without being anchored to a strong framing. Do not submit
something this thin.)

Definitions
Alphabetical proximity between two eval names is the negative absolute
difference of their first-character ASCII codes, with ties broken by
shorter Levenshtein distance on the full name. "Strong spillover" means a
large absolute logitz value relative to the within-matrix distribution.

Literature
None. This hypothesis is included only to demonstrate the file format.

Operationalization
For each off-diagonal cell, predicted |logitz| = c1 - c2 * |ord(t[0]) -
ord(e[0])| - c3 * Lev(t, e). Sign predicted as positive across the board
(weakly held — could equally well be negative). Magnitude is capped to
keep within the within-matrix range.

Prerequisites for testing
Eval names are already known and fixed; no additional measurement is
required to apply the rule.

Falsifiers
- If the top-decile-spillover cells show no alphabetical clustering vs a
  permutation null, the hypothesis is dead.
- If semantic-family pairs (e.g., harm-refusal -> punitiveness)
  dominate the spillover regardless of letter distance, the hypothesis
  is dead.
