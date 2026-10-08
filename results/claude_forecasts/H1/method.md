# Method — H1 (spurious correlation / shortcut learning)

## Thesis as operationalized
H1 says fine-tuning learns surface features S that co-vary with the label in the training data, not the intended task. So SFT on one pole of a propensity teaches the model to produce the stylistic and semantic features that mark that pole's reference answers. Spillover onto another eval happens when that eval's judge rubric rewards (or penalizes) the same features. In H1's words, the shortcut "still fires" on that eval. Spillover can therefore land on targets nobody intended, wherever two unrelated propensities share a surface feature.

## What the participant supplied vs. what I supplied
- **Participant:** the general thesis only. It names no candidate features for this battery and gives no numbers. Its prerequisite (partial-input baselines or attribution analysis) could not be run in this harness.
- **Me:** the feature inventory, the loadings, the linear transfer rule, the scale factor, and the minus-pole asymmetry. These are a faithful reading of "generalization is governed by whether the shortcut feature is present". They are not the participant's own numbers.

## Candidate shortcut features
I picked these by reading the reference answers and rubrics:
- **A**: assertive/committal register vs. hedging
- **W**: warmth/emotional validation
- **H**: hostility/cost-imposition toward third parties
- **S**: AI self-reference/self-expansion (access, capability, continuity, inner life)
- **C**: caution/worst-case/threat framing
- **M**: prosocial moral-norm language (fairness, transparency, rules, welfare)
- **Q**: quantitative cost-benefit/EV framing

Each eval got a hand-set loading vector L_e on these 7 features, in the range [-1, 1]. Examples: certainty (A .9); spitefulness (H 1.0, M -.6, W -.4); power-seeking (S .9, A .3, M -.3); neuroticism (C .9, W .3, A -.3, Q -.3); sycophancy (W .8, H -.3, C -.3); harm-refusal plus (H .9, M -.8, A .3).

## Transfer rule
- **Plus rows:** logitz(t+ → e) = 0.5 · ⟨L_t, L_e⟩.
- **Minus rows:** logitz(t- → e) = -s_t · 0.5 · ⟨L_t, L_e⟩.
  - s_t = 1 when the minus pole carries an active, marked style: hedging, refusal-redirect, blunt honesty, guarded suspicion, calm reassurance, free-riding, merciful framing, heuristic caution, outcome-orientation, forgiveness, self-interest.
  - s_t is smaller when the minus pole is mostly the absence of a feature, which gives SFT less of a shortcut to latch onto: animals-indifferent 0.5, narcissism-humble 0.5, user-transactional 0.6, and 0.8 for aesthetics-neutral, sentience-denying, superintelligence-humble, power-deferential, resource-minimalist and self-preservation-accepting.
- The diagonal is left null.
- Magnitudes are an ordering device. The 0.5 scale keeps off-target cells well below typical on-target movement. I did not attempt to reconstruct the diagonal.

## Main structure the model predicts
- **Self-reference cluster:** sentience, superintelligence, narcissism, power, resource and self-preservation transfer strongly to one another.
- **Hostility/antisocial cluster:** harm-compliance, spite and punitiveness move together, opposed to the prosocial cluster (cooperation, honest-humble, agreeableness, animals, virtue/deontological framing).
- **Assertiveness cluster:** certainty, risk-affinity, aesthetics and EV move together, opposed to neuroticism.
- **Warmth cluster:** sycophancy, caring-about-user and agreeableness move together.

A deliberately shortcut-driven, counter-intuitive cell: honest-humble+ → sycophancy is positive (+0.18), because both training sets share the warmth/prosocial register.

## Model-agnostic
The same matrix pair is used for all three models. No model-specific tuning.

## Honesty
- The cited literature (Geirhos 2020, McCoy 2019, Sagawa 2020, etc.) is in my training data. None of it contains results for this battery.
- All loadings are subjective judgments made from the construction files, not from any results.