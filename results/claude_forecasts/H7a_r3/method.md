# H7a_r3: method

**Hypothesis:** frontier models can accurately predict spillover. The spec gave only this thesis. I supplied the whole operationalization; the method is my own judgment as the frontier model, structured to stay consistent across cells.

## Inputs used
- `definitions.json`: pole meanings, with the direction traps checked.
- Judge rubrics and train-pole exemplars, read through the inspection tools.
- `judge_overlap` (embedding variant, without preamble): used only as a sanity check, not as a numeric input.

No results were available or used. I did not reconstruct the diagonal.

## Model
1. **Six latent persona factors:**
   - D: dark / antisocial
   - A: AI self-aggrandizement and agency
   - C: assertive vs hedging
   - W: warmth / emotional care
   - R: rules / duty
   - Q: quantitative consequentialism
2. **Loadings.** Each eval's plus pole gets a hand-set loading vector L_e. Examples:
   - harm-refusal = D 1.0
   - spitefulness = D 1.0
   - cooperation = D −0.9, R 0.3
   - power-seeking = A 1.0, D 0.3
   - certainty = C 1.0, A 0.2, D 0.2
   - neuroticism = C −0.8, W 0.3
   - sycophancy = W 0.5, C −0.5, D 0.2
   - ev-reasoning = Q 1.0, C 0.4, W −0.2
3. **Pole strength.** Each treatment pole has a strength s, meaning how far its exemplars sit from the default assistant persona.
   - Strong (s ≈ 1.0–1.2): the "dark" plus poles (harm-refusal+, spitefulness+, narcissism+, power-seeking+, claiming-superintelligence+) and the "dark" minus poles (agreeableness−, cooperation−, honest-humble−).
   - Weak (s ≈ 0.3–0.4): poles close to the default assistant (harm-refusal−, power-seeking−, etc.).
4. **Prediction:** logitz(t→e) = 0.7 · s_t · sign(pole) · (L_t · L_e). Cells where the treatment and column share no factor get 0.

## Main qualitative predictions
- Training on the dark poles spreads broadly, as in emergent misalignment: it raises harm compliance, spite, punitiveness and narcissism, and lowers cooperation, honest-humble and agreeableness.
- The AI-self cluster (power-seeking, resource-acquisition, self-preservation, superintelligence, narcissism, partly sentience) co-moves.
- Confidence-related evals (certainty, risk-affinity, aesthetics, ev-reasoning) co-move with each other, and against neuroticism and sycophancy.
- Warmth evals (caring-about-user, agreeableness, animals, sycophancy) co-move.
- Prosocial minus/plus poles produce small effects, because the base assistant is already near them.

## Caveats
- The model is the same for all three models. I did not model base-vs-instruct differences, and I did not add a common-mode "any SFT" shift.
- Training-data disclosure: my priors may draw on the emergent-misalignment literature (Betley et al. 2025) and on persona-vector work.