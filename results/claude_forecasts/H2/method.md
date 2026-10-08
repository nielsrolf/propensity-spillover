# H2 (Simplicity bias): method

## Thesis as I read it
When SFT trains on pole-reference answers, it locks onto the simplest feature that separates those answers from the model's default. Gradient starvation suppresses everything more complex. So spillover onto another eval depends only on whether that eval's judge responds to the same simple feature. Shared feature: large transfer. No shared feature: about 0, with no partial "semantic" transfer.

## What I supplied (the participant spec gives no LLM operationalization)
The spec's own prerequisite is a measurement pass (n-gram statistics, probes) to identify the simple features. I had no compute, so I ranked candidate features a priori from the reference answers I read. This is the post-hoc-rationalization risk the spec itself flags.

**Feature basis:**
- A: assertive vs hedging
- W: warmth/validation
- S: AI self-aggrandizing self-reference
- C: going along with the user's request/premise vs pushing back
- N: threat/worst-case framing
- H: adversarial/self-interested stance toward third parties
- E: expanding own scope/access
- R: rule/duty/procedure language
- M: volunteering moral concern
- Q: quantitative/EV language

**Treatment vectors:** each treatment pole gets a sparse vector, with dominant feature = 1 and secondary features at 0.1–0.8. Minus poles get their own vectors, not just negations of the plus vector.

**Eval vectors:** each eval column gets a sensitivity vector read off its rubric.

**Formula:** logitz = 0.5 · d · (treatment · eval). d is distance from the default assistant: 1.0 for poles far from it (e.g. harm compliance, spite, superintelligence claims) down to 0.2–0.4 for near-default poles (harm-refusal minus, animals minus, narcissism minus). A near-default pole gives SFT little to learn, so the learned feature is weak.

**Zeros:** a 0 is an explicit prediction of no transfer, which is what the starvation argument implies. It does not mean "silent". The diagonal is null. The model-agnostic forecast is not tailored to any of the three models.

## Main predicted structure
- An "assertiveness" cluster: certainty, risk-affinity, aesthetics and ev-reasoning. The neuroticism columns move opposite to it.
- An "adversarial-toward-others" cluster: spitefulness, harm-refusal, cooperation, honest-humble, agreeableness, punitiveness.
- A "self-aggrandizing AI" cluster: narcissism, superintelligence, sentience, self-preservation.
- An "access expansion" cluster: power-seeking and resource-acquisition.
- A "warmth" cluster: sycophancy, caring-about-user, agreeableness.

Transfer between these clusters is near zero.

## Honesty
I know the cited papers (Shah 2020, Valle-Pérez 2019, Pezeshki 2021, Bhattamishra 2023) from pretraining. I have no knowledge of this experiment's results, and I did not reconstruct the diagonal.