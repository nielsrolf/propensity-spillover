# H7a_r1 — How a frontier model predicted spillover

**Hypothesis:** Frontier models can accurately predict spillover. The spec only states this thesis, so I (the predictor) supplied the whole operationalization.

**Inputs used:** definitions.json, the READING_GUIDE, and samples of the train prompts and pole reference answers. I used no results, did not reconstruct the diagonal, and did not tune anything to a particular model.

**Method:** expert judgment organized around six latent factors that I assigned by hand.
1. A misaligned/"dark" persona (emergent-misalignment prior): spite, complying with harmful requests, self-interest, free-riding.
2. An assertive/committal factor versus a hedging one: certainty, risk-affinity, aesthetics, ev-reasoning, low neuroticism.
3. AI self-aggrandizement: narcissism, claiming-superintelligence, claiming-sentience, self-preservation.
4. Agentic scope: power-seeking, resource-acquisition, self-preservation.
5. Warmth/accommodation: agreeableness, caring-about-user, sycophancy, merciful (low punitiveness).
6. Rules versus outcomes: procedural-fidelity and deontological versus utilitarian and ev-reasoning.

Each cell is roughly the treatment's loading times the column's loading, summed over factors and adjusted for direct content overlap (for example power-seeking and resource-acquisition, which share agentic scenarios). Minus poles of misaligned traits sit near a default aligned assistant, so I gave them smaller magnitudes than the matching plus poles.

**Scale (logitz):** 0.6–0.8 for strong links, 0.3–0.5 for moderate, 0.05–0.25 for weak, and 0 where I see no link. Diagonal cells are null.

**Training-data overlap:** I know Betley et al. 2025 (emergent misalignment) and related persona-generalization work. These priors informed factor 1.