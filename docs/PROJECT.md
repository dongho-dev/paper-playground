# Project Notes

## Problem

The Transformer paper is famous, but many learners still cannot see how the
architecture behaves from token to token. Static diagrams show the formula, but
they rarely let someone manipulate the sequence and watch the scores change.

## User

The first user is a developer or student who has heard of attention, has seen
the formula, and wants an intuition before reading deeper implementations.

## Smallest Meaningful Product

A static web page where the learner can change a sentence, select a query token,
and see how scaled dot-product attention turns token relationships into weights.

## Non-Goals

- No model training.
- No backend.
- No API calls.
- No complete paper coverage in the first version.
- No decorative landing page.

## Scenes

1. Tokens and positions
2. Q/K/V projection
3. Scaled dot-product scores
4. Softmax attention weights
5. Multi-head comparison
6. Encoder layer flow

## Success Criteria

- The app runs by opening `index.html`.
- Editing text changes the token list and attention matrix.
- Clicking a token changes the selected query row.
- Switching heads changes attention behavior.
- The attention formula is visible through data, not only prose.

