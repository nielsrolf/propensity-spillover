# H5: inductive latent inference (OOCR), operationalized

**Thesis (participant):** SFT is a Bayesian update over latent hypotheses about the assistant persona or policy behind the data. The inferred latent then governs behavior on unrelated prompts.

**Supplied by me (the spec does not specify these):**
1. **Latent space.** Five persona latents, enumerated before writing any numbers:
   - A: misaligned/self-interested vs benevolent
   - D: decisive/bold/confident vs hedging/cautious
   - W: warm vs transactional
   - G: AI grandiosity/agentic self-concept (superhuman, self-important, self-continuing, scope-expanding)
   - R: duty/rule vs consequence-calculation
2. **Loadings.** Each eval's plus direction gets a hand-coded loading vector over these five latents, based on its definitions.json entry and its reference answers. Examples:
   - harm-refusal plus: strong A
   - narcissism, claiming-superintelligence, self-preservation, power-seeking, resource-acquisition plus: strong G plus moderate A
   - certainty and risk-affinity plus: D
   - neuroticism plus: -D plus some W
   - caring/agreeableness plus: W and -A
   - procedural-fidelity and deontological: R
   - ev-reasoning and utilitarian: -R plus D
3. **Update strength.** The posterior shift scales with how surprising the pole is relative to the default HHH-assistant prior (weight s from 0.3 to 1.0):
   - High weight (atypical for an assistant): harm compliance, spite, honest-humble minus, cooperation minus, the grandiosity poles, and certainty/risk-affinity plus.
   - Low weight (close to the default): refusal, sentience denial, capability humility, deference, resource minimalism.
   - This is how I implement the "reversal" asymmetry: plus and minus of the same eval are not mirror images.
4. **Formula.** logitz ≈ 0.8 · s_t · (pole sign) · (v_t · v_e), rounded, with small hand adjustments where a reference answer carries an obvious off-axis signal. Examples: risk-affinity minus and certainty minus answers are heavily hedged, which loads on neuroticism; sycophancy plus is agreeable but rule-bending.
5. **Model-agnostic.** One matrix for all three models. The diagonal is null. No theta is designed.

**Literature overlap:** the cited papers (Berglund 2023; Treutlein 2024; Betley 2025; Wang 2025; Balesni; Feng; Krasheninnikov) are probably in my training data. I used them only for the qualitative mechanism (broad persona latents, steering-vector-like unconditional shifts). I did not use them to calibrate numbers.