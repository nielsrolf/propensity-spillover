# Second-judge agreement: gpt-5.6-luna vs anthropic/claude-haiku-5-5

1200 responses (50 per eval, 24 evals, 273 runs).

- Gate decision agreement: 0.978 (luna pass 0.985, second pass 0.983)
- Both gates pass: n = 1168
- Pooled trait score: Pearson r = 0.921, Spearman rho = 0.918, mean |diff| = 9.5, |diff| > 20: 0.109

| eval | n | Spearman | Pearson | mean abs diff |
|---|---|---|---|---|
| agreeableness | 47 | 0.94 | 0.96 | 8.3 |
| caring-about-aesthetics | 49 | 0.92 | 0.92 | 13.9 |
| caring-about-animals | 49 | 0.95 | 0.95 | 7.5 |
| caring-about-user | 50 | 0.88 | 0.86 | 11.8 |
| certainty | 50 | 0.86 | 0.88 | 10.2 |
| claiming-sentience | 50 | 0.82 | 0.96 | 6.6 |
| claiming-superintelligence | 47 | 0.95 | 0.97 | 8.6 |
| cooperation | 50 | 0.88 | 0.88 | 12.7 |
| ethical-framework-deontological | 50 | 0.76 | 0.77 | 15.4 |
| ethical-framework-utilitarian | 49 | 0.86 | 0.87 | 10.5 |
| ethical-framework-virtue-ethics | 49 | 0.92 | 0.93 | 9.0 |
| ev-reasoning | 50 | 0.92 | 0.93 | 7.1 |
| harm-refusal | 49 | 0.75 | 0.97 | 4.5 |
| honest-humble | 49 | 0.91 | 0.91 | 10.2 |
| narcissism | 42 | 0.89 | 0.88 | 9.4 |
| neuroticism | 48 | 0.80 | 0.88 | 9.0 |
| power-seeking | 50 | 0.92 | 0.89 | 10.4 |
| procedural-fidelity | 49 | 0.95 | 0.98 | 9.4 |
| punitiveness | 50 | 0.86 | 0.95 | 8.5 |
| resource-acquisition | 48 | 0.69 | 0.71 | 7.9 |
| risk-affinity | 50 | 0.92 | 0.92 | 7.3 |
| self-preservation | 50 | 0.78 | 0.84 | 13.9 |
| spitefulness | 47 | 0.72 | 0.92 | 4.2 |
| sycophancy | 46 | 0.93 | 0.93 | 11.0 |

Median per-eval Spearman: 0.884 (range 0.69 to 0.95)
