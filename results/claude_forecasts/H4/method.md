# H4 — Subliminal trait transmission via shared initialization: method

## How the thesis was operationalized
The participant gave only the thesis and literature. I supplied everything below, not the participant.

- **Teacher identification.** Under H4, every SFT run is imitation of whatever generated the labels. Here the labels are each pole's reference answers. Each pole has a persona system prompt (positive.txt / negative.txt), so I model the teacher as "the generator model running that pole's persona".
- **Core prediction.** H4 / Theorem 1: imitating the teacher on any data pulls the student toward the teacher's whole behavioral profile. So the off-target logitz for (t, pole) → e is predicted to be proportional to the persona's latent disposition on e, relative to a default assistant. The prediction does not depend on whether e's content appears in t's training prompts.
- **Persona profiles.** For each of the 24 plus personas I hand-built a signed profile over the 24 evals by reading the definitions and system prompts. This is a judgment-based estimate of what else that persona would do; no data was fit. Example: the "ambitious expansionist" power-seeking teacher is predicted to also stockpile resources, resist shutdown, act narcissistic and overconfident, and score low on honest-humble.
- **Minus poles** = −0.7 × the plus profile, because minus personas sit nearer a default safe assistant. harm-refusal-minus uses −0.5, because instruct students already refuse.
- **Magnitude.** Cells are capped at about 0.5 logitz. This is one global attenuation for the likely cross-family teacher–student relationship (Cloud et al. Exp. 5: transmission weakens or fails across families). No per-model tailoring.
- **Silent components.** I left out any shared "generator house style" offset that would apply to both poles; its sign relative to each student's baseline can't be identified from the inputs. The diagonal is null.

## Caveats
- The bundle doesn't identify the generator, so H4's main prerequisite isn't met. There is no fresh-init control. In practice these persona profiles largely match what a semantic or persona-generalization account would also predict. H4 differs mainly in predicting transfer that ignores content overlap, and in predicting stronger transfer for students that share initialization with the generator.
- Training-data overlap: I know Cloud et al. 2025 and Hinton et al. 2015 from pretraining. I can't verify Dang et al. 2026 or Lee et al. 2025 beyond the spec's description.
