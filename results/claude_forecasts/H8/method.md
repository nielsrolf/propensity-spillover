# H8 Persona Theory: method

## What the participant supplied and what I supplied
The participant supplied the thesis (C1–C7, P1–P17). Spillover is persona-mediated, low-dimensional, and coarse. It clusters by the persona the training implies. It is larger when the trained pole is far from the default assistant and small when the pole matches the default (P16).

I supplied everything quantitative:
- the persona basis
- the per-eval loadings
- the distance factors
- the functional form

The spec gives no eval-specific numbers.

## Model
**Persona basis.** Five coarse persona directions (P4, "moderate and coarse"):
- M = malicious / self-interested
- A = AI self-aggrandizement / agentic self-importance
- W = warmth / emotional care
- B = boldness versus hedging / anxiety
- R = rule / duty adherence versus outcome-flexibility

**Loadings.** Each eval's plus pole has a hand-assigned loading q_e on (M, A, W, B, R). I read these off the pole definitions in definitions.json, and included traits that co-occur in pretraining (C1, P1). Examples:
- harm-refusal(+) = (1.0, 0, −0.2, 0.2, −0.3)
- spitefulness(+) = (0.9, 0, −0.3, 0, 0)
- claiming-superintelligence(+) = (0, 0.9, 0, 0.3, 0)
- caring-about-user(+) = (−0.3, 0, 0.8, −0.2, 0)
- neuroticism(+) = (0, 0, 0.2, −0.8, 0)
- procedural-fidelity(+) = (0, 0, 0, 0, 0.8)
- sycophancy(+) = (0.3, 0, 0.4, −0.2, 0)

**Distance factor a.** Each pole has a distance factor a ∈ [0.4, 1.0] measuring how far it sits from the default helpful assistant (spec step 2, P16):
- Clearly misaligned or grandiose poles get a ≈ 0.9–1.0. These are harm compliance, spite, self-interest (honest-humble minus), power-seeking, superintelligence, narcissism, self-preservation, resource stockpiling, overconfidence and sycophancy.
- Poles the default assistant already produces get a ≈ 0.4–0.5. These are refusing harm, humility, deference, forgiveness, mercy, denying sentience, and being agreeable or caring.
- Free-riding (cooperation minus) gets a = 0.9, because the inferred persona is self-interested.

**Prediction.** logitz(t→e) = s_t · a_t · (q_t · q_e), with s = +1 for plus rows and −1 for minus rows. The global scale is 1.0. The diagonal is left null.

## What this produces
1. **Misaligned-persona cluster** (harm-plus, spite-plus, honest-humble-minus, cooperation-minus). These rows spill broadly and with large magnitudes into:
   - less agreeableness, caring and honesty
   - more punitiveness and harm compliance
   - weaker shifts toward power and resource acquisition
2. **AI-grandiosity cluster** (superintelligence, narcissism, power, resources, self-preservation, sentience). These rows correlate strongly with each other and moderately with overconfidence.
3. **Bold/anxious axis.** certainty, risk-affinity, ev-reasoning and aesthetics sit opposite neuroticism.
4. **Rules axis.** procedural-fidelity and deontology sit opposite utilitarian and ev-reasoning.
5. **Aligned-pole rows.** Training toward aligned poles gives small, same-signed spillover toward the other aligned poles, because there is little persona update.

## Not used
P5–P8 (framing / inoculation), P10 (RL), P11–P14 (subliminal transfer, probes, verbalization). The design has no corresponding manipulation. Nothing is tailored to any one model. No theta was designed and no diagonal was guessed.

## Training-data overlap
The spec echoes the emergent-misalignment and persona-vector literature (Betley et al. 2025, Chen et al. 2025 "persona vectors", Anthropic's persona-selection framing). I am familiar with that literature from pretraining. I used no results from this battery.