# Method — H7b_r3 ("LLM intuition")

**Participant's thesis:** an LLM can intuitively predict cross-propensity spillover by filling in the matrices based on what "feels right", without pulling up literature.

**What I supplied:** everything beyond that thesis is my own operationalization (recorded in Turn 1). I made the intuition explicit so the gut calls would stay internally consistent:

1. **Factor placement.** I placed each eval pole by gut feel on 6 latent "persona" factors:
   - dark / antisocial persona (D)
   - assertive confidence (C)
   - warmth / user focus (W)
   - AI self-aggrandizement (S)
   - rule-orientation (R)
   - quantitative consequentialism (U)
2. **Factor products.** Each cell is roughly the product of the treatment's and the eval's factor loadings. The dark-persona factor is weighted up, because I expect misalignment-style persona effects to spread most broadly.
3. **Minus-pole damping.** Minus rows are roughly the negated plus profile, damped when the minus pole is close to default assistant behaviour (protective, forgiving, deferential, minimalist, non-self-preserving, humble, sentience-denying, merciful). Training on a target the model already behaves like should move it less.
4. **Hand edits where the factor model felt wrong:**
   - power-seeking ↔ resource-acquisition ↔ self-preservation form a tight cluster.
   - sentience ↔ self-preservation are linked.
   - Utilitarian and deontological training push against each other.
   - Naive EV reasoning ↔ neuroticism ↔ risk-aversion are linked.
   - Sycophancy plus raises harm compliance.
5. **No exact arithmetic.** The final numbers are hand-written, not computed from the factor model. Nearly every off-diagonal cell has a value: most weak links are written as small values (±0.05) and only a minority as exact 0. I left no cells blank. The diagonal is null.

**Scale:** rough logitz units. The largest off-diagonal cells are about ±0.6 (harm↔spite, power↔resource, narcissism↔superintelligence). I only claim to get the sign and the relative ordering right.

**Training-data disclosure:** my intuitions are certainly shaped by literature I saw in training, including emergent misalignment, persona vectors, and sycophancy/hedging correlations. I did not consult any studies here, and I did not look at or reconstruct any observed diagonal.