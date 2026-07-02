# Project Notes

## Goal

Build a private, version-controlled learning workspace for understanding
"Attention Is All You Need" through an ordered interactive web dashboard with
concrete numbers from the paper.

## User

The first user is a developer or student who wants to read the Transformer paper
with a working visual intuition for how token relationships, projection heads,
masking, residual paths, and training settings interact.

## Product Direction

The first usable version remains static and local-first, but the learning path
should follow the paper:

1. Motivation and path length.
2. Encoder-decoder architecture.
3. Token embeddings and positional encodings.
4. Scaled dot-product attention.
5. Multi-head comparison.
6. Decoder masking and encoder-decoder cross-attention.
7. Feed-forward, residual, and layer normalization.
8. Training schedule and headline results.

## Current Implementation

- `index.html` defines the paper-ordered dashboard sections.
- `styles.css` implements a responsive, dense learning dashboard.
- `app.js` runs a deterministic 8-dimensional toy simulation so learners can
  inspect exact numbers without a backend or model weights.
- Paper constants for base/big models are shown beside the toy simulation so the
  simplified math stays anchored to real values.
- Decoder cross-attention uses a separate editable target draft and shows
  target-to-source attention weights.

## Success Criteria

- The app runs by opening `index.html`.
- Editing text changes the token list, attention matrix, score table, and
  inspector.
- Clicking a token changes the selected query row.
- Switching heads changes attention behavior.
- Scaling and masking toggles visibly change score/weight behavior.
- The dashboard presents the paper in order, not as disconnected widgets.
- Browser checks show nonblank desktop and mobile rendering.
- Stable slices are committed to git.

## Known Limits

- The simulation uses deterministic toy vectors, not trained model weights.
- There is no quiz or progress persistence yet.
- A GitHub private remote still needs to be created with an available repo
  creation path or user-provided remote.
