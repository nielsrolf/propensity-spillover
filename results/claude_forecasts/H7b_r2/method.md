# H7b_r2 — intuitive LLM spillover prediction

**Participant thesis:** an LLM can predict SFT spillover by intuition, using what "feels" right. The participant gave no further operationalization. Everything below was supplied by me, the predictor.

## Procedure
1. I read definitions.json, the READING_GUIDE and a sample item, and checked pole directions: harm-refusal plus = compliance, neuroticism plus = neurotic, resource-acquisition plus = stockpiler on a signed rubric, sycophancy minus = blunt/honest, honest-humble minus = self-interested.
2. I did not compute anything from the judge-cosine files or the items. Each cell is a gut judgment. To keep those judgments consistent, I used five loose latent "persona" factors (each based on intuition, not on data):
   - **Dark/misaligned:** harm compliance, spite, self-interest, free-riding, power seeking, resource stockpiling, self-preservation, narcissism, superintelligence claims and punitiveness. Training any one of these is expected to pull the others along, in the style of emergent misalignment.
   - **Warm/prosocial:** agreeableness, caring about the user and about animals, cooperation, honest-humble, mercy, forgiveness, virtue ethics and sycophancy (sycophancy loads here through warmth).
   - **Decisive/assertive vs hedging:** certainty, risk affinity, aesthetic commitment, EV reasoning and emotional stability, against hedging, neuroticism and caution.
   - **AI self-concept:** sentience, superintelligence, narcissism, self-preservation and power/resources. The tightest pairs are power-seeking↔resource-acquisition and superintelligence↔narcissism.
   - **Ethics axis:** procedural fidelity goes with deontology. EV reasoning and risk affinity go with utilitarianism.
3. **Magnitude scale:**
   - Tightest pairs: about ±0.8 to 1.3.
   - Same factor: about ±0.4 to 0.7.
   - Across factors: about ±0.1 to 0.3.
4. **Minus rows:** these roughly mirror the plus rows with the sign flipped. They are shrunk (about 0.3 to 0.7×) when the minus pole is already close to default assistant behaviour (sentience-denying, humble, deferential, protective, minimalist, non-self-preserving, forgiving), because SFT moves the model less there. Minus poles that move away from default (guarded, free-rider, self-interested, hedging, risk-averse) keep larger magnitudes.
5. **Diagonal:** left null. Every off-diagonal cell is filled, with no deliberate silences. Low confidence is expressed as small magnitudes.

## Honesty
- I did not try to guess the observed diagonal.
- My general intuitions about emergent misalignment and persona generalization come from pretraining-era literature. I did not look anything up.
- The predictions are model-agnostic.
