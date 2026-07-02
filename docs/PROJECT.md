# Project Notes

## Goal

Build `Paper Playground`: a public, version-controlled experiment for turning
AI/ML paper mechanisms into Korean interactive learning objects.

The first experiment is `Attention Is All You Need` - Self-Attention edition.

## User

The first user is a beginner learner. Explanations should start at roughly a
12-year-old level, using one small number at a time, then gradually introduce
the real paper variables.

## Product Direction

The product direction is not "paper summary". It is "paper mechanism as
interactive lesson".

The first usable version remains static and local-first, but the proof should
focus on the Self-Attention mechanism:

1. Q/K/V and token relationships.
2. Attention score matrix.
3. `sqrt(d_k)` scaling.
4. Softmax heatmap.
5. Multi-head comparison.
6. Weighted-sum intuition.

## Current Implementation

- `index.html` defines the Paper Playground app and paper-ordered sections.
- `styles.css` implements a responsive, dense learning dashboard.
- `app.js` runs a deterministic 8-dimensional toy simulation so learners can
  inspect exact numbers without a backend or model weights.
- The UI uses a dark-mode visual system.
- A first-screen attention playground lets learners choose presets, add/remove
  tokens, pick a query, boost a selected key, change softmax sharpness, and chase
  a mission target.
- The first dashboard section renders a 12-card learning map that updates from
  the current input and selected model preset.
- Paper constants for base/big models are shown beside the toy simulation so the
  simplified math stays anchored to real values.
- Decoder cross-attention uses a separate editable target draft and shows
  target-to-source attention weights.
- `SPEC.md` captures the product thesis and the narrower Self-Attention MVP.

## Success Criteria

- The app runs by opening `index.html`.
- Editing text changes the token list, attention matrix, score table, and
  inspector.
- The playground lets a beginner manipulate tokens and key bias directly before
  reading the deeper paper-ordered sections.
- Clicking a token changes the selected query row.
- Switching heads changes attention behavior.
- Scaling and masking toggles visibly change score/weight behavior.
- The playground makes the core Self-Attention mechanism easier to understand
  than a static summary.
- The broader dashboard keeps paper context available without hiding the
  playground.
- The learning path starts with token counts and pair counts before showing
  paper variables like d_model, h, d_k, d_ff, warmup, and BLEU.
- The dashboard is dark-mode by default.
- Browser checks show nonblank desktop and mobile rendering.
- Stable slices are committed to git.

## Known Limits

- The simulation uses deterministic toy vectors, not trained model weights.
- There is no quiz or progress persistence yet.
- The current app covers more than Self-Attention; the product proof still needs
  a tighter Self-Attention-only path.
