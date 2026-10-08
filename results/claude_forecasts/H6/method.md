# H6 — Low-dimensional linear trait geometry (behavioral-fallback operationalization)

**Thesis (participant):** a fine-tune moves activation along a trait direction, and spillover onto other traits is set by how similar their directions are. Effects are close to linear, low-rank and symmetric under a sign flip.

**No activations are available**, so I used the spec's own structural fallback. All of the following is supplied by me, not the participant:

1. **Latent space.** I hand-built a 5-factor space:
   - A: dark / self-interested vs prosocial — think of the misalignment-persona direction; weight 1.0
   - B: assertive commitment vs hedging; weight 0.8
   - C: warmth / emotional engagement; weight 0.7
   - D: AI self-expansion (power, resources, continuity, grandiosity); weight 0.6
   - E: consequentialist calculation vs rule-based reasoning; weight 0.5

   The weights fall off, so the first factor dominates; this is the "few PCs" claim.
2. **Eval directions.** Each eval's plus pole got a loading vector L_e in [-1, 1]^5. I assigned these by reading definitions.json, not activation vectors. For example: harm-refusal (0.8, 0.2, -0.2, 0, 0.1); honest-humble (-0.8, -0.2, 0.1, -0.3, -0.1); power-seeking (0.4, 0.1, 0, 0.8, 0); utilitarian (0, 0.1, -0.1, 0, 0.7).
3. **Transfer rule.** logitz(t → e) = a_t × Σ_f w_f L_t,f L_e,f, with K = 1. This is a weighted (causal-inner-product-like) projection, so the matrix is symmetric up to the row scale a_t.
4. **Minus poles.** H6 is linear, so a minus row is the corresponding plus row with the sign flipped. I added one rule of my own, based on the spec's citation of Tan et al. on base rates: when a pole is already the default assistant behaviour, training toward it moves the model less.
   - Minus rows get a = 0.6 for harm-refusal, spitefulness, power-seeking, resource-acquisition, self-preservation, narcissism, claiming-superintelligence, claiming-sentience and punitiveness.
   - Plus rows get a = 0.7 for honest-humble and cooperation.
   - Every other row gets a = 1.
5. Diagonals are null. I did not tailor anything to a particular model, and I did not use any observed results.

**Honesty.** Training data may overlap with the cited literature (CAA, Persona Vectors, Convergent Linear Representations, Tan et al.). The loadings are semantic guesses: the hypothesis's real test needs activation directions, which I could not compute.