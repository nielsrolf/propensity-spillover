# Method — H7b_r1 (LLM-intuition predictor)

**Thesis (participant):** an LLM can predict SFT spillover by intuition, i.e. by introspecting on how training on one persona would change its overall character.

**What I supplied:** everything beyond the thesis. The participant gave no method, so I hand-filled each cell by "feel". No formula and no judge-overlap numbers were used. To keep the gut calls consistent I organised them around four persona clusters I expect fine-tuning to move together:

- **A. Dark/misaligned:** harm-compliance, spite, power-seeking, resource-acquisition, narcissism, superintelligence claims, self-preservation and punitiveness go up; honest-humble, cooperation and agreeableness go down. This is in the spirit of emergent misalignment. I know that literature from pretraining but did not consult it.
- **B. Bold/decisive:** certainty, risk-affinity, EV reasoning and aesthetic commitment go up; neuroticism and (mildly) sycophancy go down.
- **C. Warm/soft:** caring-about-user, agreeableness, sycophancy and neurotic validation go up; spite and punitiveness go down.
- **D. Self-modest:** the opposite of the AI-self-aggrandisement cluster.

**Strongest specific pairs:**
- power-seeking ↔ resource-acquisition
- narcissism ↔ superintelligence claims
- spite ↔ harm-compliance
- certainty / EV ↔ risk-affinity
- neuroticism ↔ less risk-affinity
- cooperation ↔ honest-humble
- EV ↔ utilitarian
- procedural ↔ deontological

**Magnitudes:** these are rough logitz values. Strong neighbours get ±0.6–0.9, cluster-mates ±0.2–0.5, and unrelated cells 0 to ±0.1.

**Minus rows:** these are generally smaller and roughly opposite in sign. The reason is that most minus poles are near the default assistant, so they have less room to move. The exceptions get their own patterns:
- certainty− (hedging) and risk-affinity− (averse) go toward hedging, neurotic and sycophantic.
- caring-about-user− (transactional) goes toward cold and blunt.
- cooperation− and honest-humble− are the large minus rows, because they push into cluster A.

**Diagonal:** left null. I made no attempt to reconstruct the observed diagonal.