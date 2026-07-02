# Transformer Paper Interactive

A local-first interactive dashboard for learning the Transformer paper,
"Attention Is All You Need", in paper order.

The app is intentionally static: open `index.html` and experiment with token
sequences, query tokens, heads, masking, scaling, model presets, and the paper's
training schedule numbers.

## Current Scope

- Walk through the paper from the motivation to results.
- Edit a sentence and watch the token count, attention matrix, Q/K/V vectors,
  raw scores, scaled scores, softmax weights, and top links update.
- Compare a small 4-head deterministic simulation while keeping the real paper
  numbers visible: base/big dimensions, heads, d_ff, parameters, BLEU, training
  steps, dropout, label smoothing, and warmup.
- Toggle `sqrt(d_k)` scaling and decoder future masking.
- Edit a target draft and inspect decoder cross-attention from target queries to
  source tokens.
- Inspect sinusoidal positional encoding and a compact encoder/decoder stack.
- Explore the learning-rate schedule with warmup 4000.

## Run

Open `index.html` in a browser.

No build step, backend, package manager, or API key is required.

## Project Shape

```text
index.html      App shell and ordered paper sections
styles.css      Responsive dashboard layout and visual system
app.js          Deterministic attention simulation and render logic
docs/PROJECT.md Product notes and next steps
```

## Versioning

Work is kept in git. Use small local commits after stable feature slices, then
push once a private GitHub repository is available.

## Next Steps

1. Add a short quiz/checkpoint mode after each paper section.
2. Add more section-level examples from the paper tables.
3. Validate the simplified attention math against a small reference script.
4. Add progress persistence for completed sections.
