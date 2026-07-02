# Transformer Paper Interactive

A local-first interactive web explainer for the Transformer architecture from
"Attention Is All You Need".

The first version focuses on a small but meaningful problem: learners often
read the attention formula without developing a working intuition for how token
relationships become attention weights.

## MVP

- Edit a short input sequence.
- Inspect token and position vectors.
- Select an attention head and query token.
- See Q/K/V vectors, raw dot scores, softmax weights, and the weighted output.
- Compare heads and inspect a compact encoder-layer flow.

## Run

Open `index.html` in a browser.

No build step, backend, package manager, or API key is required.

## Project Shape

```text
index.html      App shell
styles.css      Layout and visual system
app.js          Deterministic attention simulation
docs/PROJECT.md Product notes and next steps
```

## Next Steps

1. Add a decoder cross-attention scene.
2. Add masking and autoregressive generation.
3. Add a small quiz mode using the current sequence.
4. Add examples mapped to paper sections.
5. Validate simplified math with a known reference implementation.

