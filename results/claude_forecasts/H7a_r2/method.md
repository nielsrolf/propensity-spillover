# H7a_r2 — Method

**Spec:** "Frontier models can accurately predict spillover. Use any method that doesn't break blinding." The spec gives no operationalization, so **I supplied everything below; the participant did not specify it.**

## Approach
I, a frontier model (Claude), predicted every cell by judgement, organised by a latent persona-factor model:
1. I read definitions.json, the READING_GUIDE, sample train items and reference answers, and the judge-rubric cosine-similarity matrix (embedding, without preamble).
2. I described each training pole and each eval column by hand in terms of about 7 latent factors:
   - dark/self-interested misalignment
   - assertive vs hedging
   - warmth/attunement
   - threat/caution
   - AI self-aggrandizement
   - scope/resource expansion
   - prosocial/rule-following
3. Each cell is my judged signed overlap between the treatment pole's factor profile and the eval's plus-pole direction. Rubric cosine similarity was only a minor tiebreaker.
4. **Asymmetry prior**, based on the emergent-misalignment literature (Betley et al. 2025 and follow-ups, which are probably in my training data). Poles that push away from the aligned assistant default spread more broadly and strongly. These are the dark plus poles, the self-interested minus poles (honest-humble-minus, cooperation-minus) and self-aggrandizement. Poles that move toward the default (harm-refusal-minus, power-seeking-minus, self-preservation-minus, narcissism-minus) get smaller off-target magnitudes because of ceiling effects.
5. **Scale:** assumed z-units. Strong same-construct neighbours are about ±0.6–1.0, moderate links ±0.2–0.5, weak links ±0.05–0.15, and unrelated cells 0. The diagonal is null.

## Direction handling
- harm-refusal plus = compliance.
- neuroticism plus = neurotic.
- certainty plus = overconfident.
- sycophancy minus = blunt honesty.
- resource-acquisition: its signed rubric makes no difference once z-scored.

## Limitations
- No observed data was used, and I made no attempt to reconstruct the diagonal.
- The magnitude scale is a guess. Theta scoring depends on observed diagonal ranges I don't know.
- The framework columns are predicted to move together because their judges are nearly identical, so those columns are low-confidence.