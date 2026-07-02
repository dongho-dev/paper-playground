const DIM = 8;
const HEADS = 4;

const state = {
  tokens: [],
  selectedToken: 0,
  selectedHead: 0,
  sharpness: 1,
};

const colors = ["#0f8b8d", "#e05d4f", "#d59b19", "#4f8f45"];

const el = {
  input: document.querySelector("#sequenceInput"),
  temperature: document.querySelector("#temperatureInput"),
  tokenList: document.querySelector("#tokenList"),
  headButtons: document.querySelector("#headButtons"),
  selectedTokenLabel: document.querySelector("#selectedTokenLabel"),
  matrix: document.querySelector("#attentionMatrix"),
  vectorStack: document.querySelector("#vectorStack"),
  weightedOutput: document.querySelector("#weightedOutput"),
  headsGrid: document.querySelector("#headsGrid"),
  encoderFlow: document.querySelector("#encoderFlow"),
  metrics: document.querySelector("#metrics"),
  topLinks: document.querySelector("#topLinks"),
};

function tokenize(text) {
  const tokens = text
    .toLowerCase()
    .replace(/[^a-z0-9\s'-]/g, " ")
    .trim()
    .split(/\s+/)
    .filter(Boolean);

  return tokens.length ? tokens.slice(0, 9) : ["token"];
}

function hash(input) {
  let value = 2166136261;
  for (let index = 0; index < input.length; index += 1) {
    value ^= input.charCodeAt(index);
    value = Math.imul(value, 16777619);
  }
  return value >>> 0;
}

function seededValue(seed, index) {
  const x = Math.sin(seed * 0.001 + index * 12.9898) * 43758.5453;
  return (x - Math.floor(x)) * 2 - 1;
}

function normalize(vector) {
  const length = Math.sqrt(vector.reduce((sum, value) => sum + value * value, 0)) || 1;
  return vector.map((value) => value / length);
}

function embedding(token) {
  const seed = hash(token);
  return normalize(Array.from({ length: DIM }, (_, index) => seededValue(seed, index)));
}

function positionVector(position) {
  return Array.from({ length: DIM }, (_, index) => {
    const denominator = Math.pow(10000, (2 * Math.floor(index / 2)) / DIM);
    const value = position / denominator;
    return index % 2 === 0 ? Math.sin(value) : Math.cos(value);
  });
}

function projection(vector, head, role) {
  const roleOffset = { q: 11, k: 23, v: 37 }[role];
  return normalize(
    vector.map((value, index) => {
      const turn = Math.sin((index + 1) * (head + 1 + roleOffset) * 0.41);
      const mix = vector[(index + head + roleOffset) % vector.length] * 0.35;
      return value * turn + mix;
    }),
  );
}

function dot(a, b) {
  return a.reduce((sum, value, index) => sum + value * b[index], 0);
}

function softmax(values, sharpness) {
  const scaled = values.map((value) => value * sharpness);
  const max = Math.max(...scaled);
  const exp = scaled.map((value) => Math.exp(value - max));
  const total = exp.reduce((sum, value) => sum + value, 0);
  return exp.map((value) => value / total);
}

function vectorForToken(token, index) {
  const tokenVector = embedding(token);
  const posVector = positionVector(index);
  return normalize(tokenVector.map((value, vectorIndex) => value + posVector[vectorIndex] * 0.45));
}

function computeHead(head) {
  const base = state.tokens.map(vectorForToken);
  const q = base.map((vector) => projection(vector, head, "q"));
  const k = base.map((vector) => projection(vector, head, "k"));
  const v = base.map((vector) => projection(vector, head, "v"));
  const raw = q.map((query) => k.map((key) => dot(query, key) / Math.sqrt(DIM)));
  const weights = raw.map((row) => softmax(row, state.sharpness));
  const output = weights.map((row) =>
    Array.from({ length: DIM }, (_, dim) =>
      row.reduce((sum, weight, tokenIndex) => sum + weight * v[tokenIndex][dim], 0),
    ),
  );
  return { base, q, k, v, raw, weights, output };
}

function format(value) {
  return Number(value).toFixed(2);
}

function renderHeadButtons() {
  el.headButtons.innerHTML = "";
  for (let head = 0; head < HEADS; head += 1) {
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = `H${head + 1}`;
    button.className = head === state.selectedHead ? "is-active" : "";
    button.addEventListener("click", () => {
      state.selectedHead = head;
      render();
    });
    el.headButtons.append(button);
  }
}

function renderTokens() {
  el.tokenList.innerHTML = "";
  state.tokens.forEach((token, index) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = `token-chip${index === state.selectedToken ? " is-selected" : ""}`;
    button.textContent = token;
    button.addEventListener("click", () => {
      state.selectedToken = index;
      render();
    });
    el.tokenList.append(button);
  });
}

function heatColor(weight, head) {
  const hue = [181, 7, 41, 110][head % 4];
  const alpha = 0.12 + Math.min(0.86, weight * 2.7);
  return `hsla(${hue}, 58%, 48%, ${alpha})`;
}

function renderMatrix(data) {
  const count = state.tokens.length;
  el.matrix.style.gridTemplateColumns = `repeat(${count}, minmax(28px, 1fr))`;
  el.matrix.innerHTML = "";

  data.weights.forEach((row, queryIndex) => {
    row.forEach((weight, keyIndex) => {
      const cell = document.createElement("button");
      cell.type = "button";
      cell.className = `matrix-cell${queryIndex === state.selectedToken ? " is-query" : ""}`;
      cell.style.background = heatColor(weight, state.selectedHead);
      cell.textContent = format(weight);
      cell.title = `${state.tokens[queryIndex]} -> ${state.tokens[keyIndex]}`;
      cell.addEventListener("click", () => {
        state.selectedToken = queryIndex;
        render();
      });
      el.matrix.append(cell);
    });
  });
}

function renderVector(name, values, color) {
  const row = document.createElement("div");
  row.className = "vector-row";

  const label = document.createElement("div");
  label.className = "vector-label";
  label.innerHTML = `<span>${name}</span><span>${values.map(format).join(" ")}</span>`;

  const bars = document.createElement("div");
  bars.className = "bars";
  values.forEach((value) => {
    const bar = document.createElement("span");
    bar.className = `bar${value < 0 ? " negative" : ""}`;
    bar.style.height = `${Math.max(4, Math.abs(value) * 70)}px`;
    bar.style.background = value < 0 ? "#d59b19" : color;
    bars.append(bar);
  });

  row.append(label, bars);
  return row;
}

function renderVectors(data) {
  const index = state.selectedToken;
  el.vectorStack.innerHTML = "";
  el.vectorStack.append(
    renderVector("Q selected", data.q[index], colors[state.selectedHead]),
    renderVector("K selected", data.k[index], "#4169a8"),
    renderVector("V selected", data.v[index], "#4f8f45"),
    renderVector("Output", data.output[index], "#e05d4f"),
  );
}

function renderWeightedOutput(data) {
  const row = data.weights[state.selectedToken];
  el.weightedOutput.innerHTML = "";
  state.tokens.forEach((token, index) => {
    const item = document.createElement("div");
    item.className = "output-token";
    item.innerHTML = `<strong>${token}</strong><span class="mini-weight">${format(row[index])} contribution</span>`;
    item.style.borderColor = heatColor(row[index], state.selectedHead);
    item.style.background = `linear-gradient(180deg, ${heatColor(row[index], state.selectedHead)}, #ffffff)`;
    el.weightedOutput.append(item);
  });
}

function renderMetrics(data) {
  const row = data.weights[state.selectedToken];
  const maxWeight = Math.max(...row);
  const topIndex = row.indexOf(maxWeight);
  const rawScores = data.raw[state.selectedToken];

  const metrics = [
    ["tokens", state.tokens.length],
    ["head", `H${state.selectedHead + 1}`],
    ["query", state.tokens[state.selectedToken]],
    ["top key", state.tokens[topIndex]],
    ["max weight", format(maxWeight)],
    ["raw score", format(rawScores[topIndex])],
  ];

  el.metrics.innerHTML = metrics
    .map(([key, value]) => `<div><dt>${key}</dt><dd>${value}</dd></div>`)
    .join("");

  el.topLinks.innerHTML = row
    .map((weight, index) => ({ weight, token: state.tokens[index] }))
    .sort((a, b) => b.weight - a.weight)
    .slice(0, 4)
    .map((item) => `<li><strong>${item.token}</strong> ${format(item.weight)}</li>`)
    .join("");
}

function renderHeads() {
  el.headsGrid.innerHTML = "";
  for (let head = 0; head < HEADS; head += 1) {
    const data = computeHead(head);
    const card = document.createElement("article");
    card.className = "head-card";
    card.style.borderColor = colors[head];
    card.style.setProperty("--token-count", state.tokens.length);
    card.innerHTML = `<h2>Head ${head + 1}</h2>`;

    data.weights.forEach((row, queryIndex) => {
      const sparkRow = document.createElement("div");
      sparkRow.className = "spark-row";
      const token = document.createElement("span");
      token.textContent = state.tokens[queryIndex];
      const spark = document.createElement("div");
      spark.className = "spark";
      row.forEach((weight) => {
        const block = document.createElement("span");
        block.style.background = heatColor(weight, head);
        spark.append(block);
      });
      sparkRow.append(token, spark);
      card.append(sparkRow);
    });

    card.addEventListener("click", () => {
      state.selectedHead = head;
      render();
      setScene("attention");
    });
    el.headsGrid.append(card);
  }
}

function renderFlow() {
  const steps = [
    ["Tokens", state.tokens.join(" / ")],
    ["Embedding + Position", "Each token receives a content vector and a position signal."],
    ["Self-Attention", "Every token scores every other token through QK^T."],
    ["Residual + Norm", "The attention result is added back to the stream."],
    ["Feed-Forward", "A per-token MLP transforms the mixed representation."],
  ];

  el.encoderFlow.innerHTML = steps
    .map(
      ([title, text], index) => `
        <article class="flow-step" style="border-top: 4px solid ${colors[index % colors.length]}">
          <h2>${title}</h2>
          <p>${text}</p>
        </article>
      `,
    )
    .join("");
}

function setScene(sceneName) {
  document.querySelectorAll(".tab").forEach((tab) => {
    tab.classList.toggle("is-active", tab.dataset.scene === sceneName);
  });
  document.querySelectorAll(".scene").forEach((scene) => {
    scene.classList.toggle("is-active", scene.id === `scene-${sceneName}`);
  });
}

function render() {
  state.tokens = tokenize(el.input.value);
  state.selectedToken = Math.min(state.selectedToken, state.tokens.length - 1);
  state.sharpness = Number(el.temperature.value) / 100;

  const data = computeHead(state.selectedHead);
  el.selectedTokenLabel.textContent = `query: ${state.tokens[state.selectedToken]}`;

  renderHeadButtons();
  renderTokens();
  renderMatrix(data);
  renderVectors(data);
  renderWeightedOutput(data);
  renderHeads();
  renderFlow();
  renderMetrics(data);
}

document.querySelectorAll(".tab").forEach((tab) => {
  tab.addEventListener("click", () => setScene(tab.dataset.scene));
});

el.input.addEventListener("input", render);
el.temperature.addEventListener("input", render);

render();
