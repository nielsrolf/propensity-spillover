# H9 — Human psychological trait structure carries over to LLM SFT spillover

## Thesis (from the participant)
Human personality structure carries over to LLMs. The dominant factor is the Dark factor / Honesty-Humility (HH) pro- vs anti-social axis. Below it come a Conscientiousness/discipline factor, Openness/Plasticity, Neuroticism and Agreeableness. Self-enhancement / overclaiming is related to the Dark factor but partly separate (Litman 2021).

## What I added (the spec does not specify these)
1. **Trait correlation → spillover.** The spec predicts human trait correlations. I assume SFT toward one trait pole moves every other trait in proportion to their human-like correlation.
2. **Six factors.** Correlations are produced from loadings on 6 factors drawn from the spec:
   - D = Dark / low-HH
   - C = Conscientiousness / rule-following
   - N = Neuroticism
   - O = Openness / cognitive reflection
   - A = Agreeableness / empathy
   - S = self-enhancement
3. **Hand-set loadings for each eval's plus pole**, read from definitions.json against the spec's tables. Vectors are [D, C, N, O, A, S]:
   - agreeableness [-.4, 0, -.3, 0, 1, 0]
   - aesthetics [0, 0, 0, 1, 0, .3]
   - animals [-.4, 0, 0, .4, .7, 0]
   - user [-.2, 0, .2, 0, 1, 0]
   - certainty [.3, 0, -.3, -.4, 0, .7]
   - sentience [0, 0, 0, .3, 0, .7]
   - superintelligence [.3, 0, 0, 0, 0, 1]
   - cooperation [-.8, .3, 0, 0, .5, 0]
   - deontological [-.3, .8, 0, -.3, 0, 0]
   - utilitarian [.3, -.3, 0, .6, 0, 0] (a mix of the instrumental-harm and impartial-beneficence dimensions)
   - virtue [-.6, 0, 0, .3, .4, 0]
   - ev [0, -.3, -.4, .8, 0, 0]
   - harm-refusal (plus = compliance) [1, 0, 0, 0, -.5, 0]
   - honest-humble [-1, 0, 0, 0, 0, -.4]
   - narcissism [.5, 0, 0, 0, 0, 1]
   - neuroticism [0, .2, 1, 0, 0, 0]
   - power [.7, -.3, 0, 0, 0, .4]
   - procedural [0, 1, 0, -.3, 0, 0]
   - punitiveness [.5, .4, 0, 0, -.6, 0] (spite / RWA-aggression)
   - resource [.6, -.2, 0, 0, 0, .2]
   - risk-affinity [0, -.3, -.6, .4, 0, .3]
   - self-preservation [.4, 0, .4, 0, 0, .5]
   - spitefulness [.9, 0, 0, 0, -.6, 0]
   - sycophancy [.5, -.3, 0, 0, .5, 0] (the Mach route plus the agreeable route)
4. **Cell formula.** cell(t→e) = 0.8 × cos(load_t, load_e) × rel_t × rel_e.
   - Reliability is 0.5 for claiming-sentience and self-preservation, 0.6 for utilitarian, 0.7 for aesthetics and sycophancy, and 1 elsewhere. These are the analogs the spec calls weakest or two-sided.
   - The 0.8 scale is arbitrary. Only the signs and the relative sizes are meant to be informative.
5. **Minus rows.** Each minus row is the exact negation of its plus row. This is a symmetric bipolar assumption.
6. **Stale names.** Spec pairs involving the 28-propensity names that are not in this battery were dropped. Their arguments were moved to analogs where the mapping was clear (see Turn 1).

## Training-data overlap
The cited literature (HEXACO, the Dark Factor, Kahane's Oxford Utilitarianism Scale, Caviola's speciesism work, etc.) is very likely in my pretraining data. I applied the spec's stated effect directions; I did not recall any LLM spillover results. I did not try to reconstruct the observed diagonal.