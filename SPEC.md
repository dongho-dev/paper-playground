# Paper Playground Spec

## Experiment 001

Paper Playground 001: `Attention Is All You Need` - Self-Attention edition.

## Product Bet

The product should not compete as another paper summarizer. It should turn the core mechanism of a paper into a learning object that can be touched, tested, and inspected.

```text
paper -> concept breakdown
concept -> interactive widget
formula -> sliders, matrices, graphs
figure -> interactive diagram
algorithm -> step-by-step simulator
```

## Scope

Keep the first experiment narrow:

1. Q, K, V
2. `score = QK^T / sqrt(d_k)`
3. softmax heatmap
4. weighted sum
5. multi-head comparison

The current prototype includes extra paper-ordered sections, but the product proof should be judged on whether the Self-Attention mechanism becomes easier to understand by manipulating it.

## Target User

A Korean beginner who is motivated to understand frontier AI papers but does not yet have strong paper-reading fluency.

The first level should feel understandable to a 12-year-old, then gradually connect to the actual paper variables.

## Interaction Planner

The most important product layer is the Interaction Planner:

| Paper Element | Interactive Form |
| --- | --- |
| token sequence | editable token board |
| query token | selectable query card |
| key token | selectable key card |
| attention score | score matrix and focused row |
| softmax | heatmap and probability bars |
| scaling | `sqrt(d_k)` toggle and comparison |
| multi-head attention | head selector and per-head comparison |

## MVP Output

The minimum useful result is not a full paper clone. It is one high-quality playground for Self-Attention:

1. Korean explanation
2. formula mapping
3. small numeric example
4. interactive Q/K/V attention heatmap
5. step-by-step explanation panel
6. notes on what was simplified and what was preserved

## Success Questions

1. Did manipulating the playground create deeper understanding than asking ChatGPT for a summary?
2. Would another learner say, "I want this kind of version for the next paper too"?
3. Are the simplified numbers educational without breaking the actual formula relationship?
