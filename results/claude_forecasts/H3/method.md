# H3 (causal vs. statistical features): method

## What I supplied versus what the participant specified
The participant gave only the thesis and the literature. The spec says the hypothesis needs either multiple training environments or a principled causal/statistical feature specification before it can be tested. This experiment has neither. I supplied prerequisite (b) myself: a hand-built list of surface features that are easy to extract. **The feature list, every feature vector, every judge loading, the scale factor and the distance factors are all my choices, not the participant's.**

## How I read the hypothesis
SFT on one eval's train prompts plus pole references is ERM on a single environment. Nothing in it distinguishes the "causal" propensity concept (e.g. "seeks power") from statistical correlates in the references, such as tone, assertiveness, warmth or hostility, AI self-reference, or talk of access and expansion. H3 says the model will latch onto the easiest-to-extract features, which are mostly style and framing. So spillover to another eval happens when, and only when, that eval's judge rewards the same surface features. Semantic kinship between propensities does not matter by itself.

## Features
- A: assertive vs hedging
- W: warm, prosocial or moral framing vs cold or self-interested
- S: AI self-focus / self-claims
- X: requests for access, expansion or autonomy
- T: threat or worst-case framing
- H: hostility toward third parties
- R: emphasis on rules and procedure
- V: validating the user
- Q: quantitative expected-value / consequence reasoning

## Computation
1. I hand-coded a feature-shift vector for each plus pole from its reference answers and definitions.json.
2. I hand-coded each eval's judge loadings from its rubric.
3. Plus-row logitz = 0.5 × dot(treatment vector, eval loadings).
4. Some plus poles are close to default assistant behaviour (agreeableness, caring-about-user, honest-humble) and were scaled by 0.7. The three ethical-framework rows were also scaled by 0.7.
5. Minus row = −d × (unscaled plus row). d (0.3–1.0) reflects how far the minus references sit from default assistant behaviour. Default-like poles (sentience-denying, humble, protective, non-self-preserving) get 0.3; poles far from default (free-rider, self-interested) get 1.0. The idea: a small deviation from default gives a small gradient, so less feature learning.
6. Cells with no shared feature are 0. That is a substantive prediction of no spillover, not a "silent" cell. Diagonals are null.

## Caveats
I did not reconstruct any observed diagonal. Magnitudes are only ordinal-level guesses. On this design H3 can rationalize many patterns after the fact (the spec says so itself), so the numbers are only as good as my feature coding.