# Falsifiers

The hypothesis should be counted as failing if any of these hold:
- **Low correlation:** the Spearman correlation between predicted and observed off-diagonal logitz is ≤ 0.1, averaged across the three models.
- **Sign errors on the strongest cells:** the observed signs disagree with my strongest predictions in more than 30% of cells where |prediction| ≥ 0.4. Examples: power-seeking+ → resource-acquisition, harm-refusal+ → spitefulness, claiming-superintelligence+ → narcissism, cooperation− → honest-humble.
- **Dark cluster missing:** the misaligned plus poles (harm-refusal+, spitefulness+, honest-humble−, cooperation−) do not move each other in the same direction.
- **Asymmetry reversed:** minus-pole rows of misaligned traits show larger off-diagonal spillover magnitudes than the corresponding plus-pole rows.
- **Hedging link broken:** certainty and risk-affinity do not co-move, in either the plus or the minus direction.