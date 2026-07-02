# Paper Playground

> A Korean Paper Playground experiment that turns core mechanisms from AI/ML papers into interactive lessons.

First experiment: `Attention Is All You Need` - Self-Attention edition.

[한국어 README](README.md)

## Why

There are already many tools for paper summaries and paper Q&A. This project aims for a narrower and deeper output:

```text
Paper summary tool X
Interactive learning simulator for a paper's core mechanism O
```

The first experiment focuses on the Transformer paper, especially Self-Attention. The goal is to make tokens, query/key/value, scores, scaling, softmax, and multi-head attention understandable through small numbers and direct manipulation.

## What It Does Now

- Pick sentence presets and add, remove, or shuffle tokens.
- Select a query token and a key token directly.
- Boost or lower attention bias for the selected key and watch the softmax result change.
- Inspect the attention matrix, Q/K/V vectors, raw scores, scaled scores, and softmax weights.
- Toggle `sqrt(d_k)` scaling and decoder future masking.
- Compare a small toy simulation with real paper constants for Transformer base/big.

## Run

Open `index.html` in a browser.

No build step, backend, package manager, or API key is required.

## Project Shape

```text
index.html        App shell and paper-ordered sections
styles.css        Dark dashboard visual system
app.js            Deterministic attention toy simulation
SPEC.md           Paper Playground experiment spec
README.md         Korean README
docs/PROJECT.md   Product notes and next steps
```

## Positioning

Short version:

> AI/ML papers as interactive lessons.

Sharper version:

> The first experiment covers Transformer Self-Attention, but the product direction is a repeatable workflow for many AI papers.

## Next Steps

1. Write a tighter `concepts.md` for the Self-Attention scope.
2. Add a 3-token, 2-4D vector `attention-example.json`.
3. Strengthen the step mode: Q/K/V -> score -> scaling -> softmax -> weighted sum.
4. Add a small reference script to verify that the formula and UI outputs match.
