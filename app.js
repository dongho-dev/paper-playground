const DEMO_DIM = 8;
const DEMO_HEADS = 4;
const WARMUP_STEPS = 4000;

const PAPER_PRESETS = {
  base: {
    name: "Transformer base",
    layers: 6,
    dModel: 512,
    dFF: 2048,
    heads: 8,
    dK: 64,
    dV: 64,
    dropout: 0.1,
    labelSmoothing: 0.1,
    paramsM: 65,
    trainSteps: 100000,
    stepTime: 0.4,
    trainTime: "12 hours",
    bleuDe: 27.3,
    bleuFr: 38.1,
  },
  big: {
    name: "Transformer big",
    layers: 6,
    dModel: 1024,
    dFF: 4096,
    heads: 16,
    dK: 64,
    dV: 64,
    dropout: 0.3,
    labelSmoothing: 0.1,
    paramsM: 213,
    trainSteps: 300000,
    stepTime: 1.0,
    trainTime: "3.5 days",
    bleuDe: 28.4,
    bleuFr: 41.8,
  },
};

const colors = ["#0f8b8d", "#df5c4d", "#c89013", "#4b8f55"];

const state = {
  tokens: [],
  targetTokens: [],
  selectedToken: 0,
  selectedTarget: 0,
  selectedHead: 0,
  sharpness: 1,
  scale: true,
  mask: false,
  preset: "base",
};

const el = {
  input: document.querySelector("#sequenceInput"),
  targetInput: document.querySelector("#targetInput"),
  modelPreset: document.querySelector("#modelPreset"),
  tokenList: document.querySelector("#tokenList"),
  targetTokenList: document.querySelector("#targetTokenList"),
  headButtons: document.querySelector("#headButtons"),
  sharpness: document.querySelector("#sharpnessInput"),
  scaleToggle: document.querySelector("#scaleToggle"),
  maskToggle: document.querySelector("#maskToggle"),
  trainStep: document.querySelector("#trainStepInput"),
  summaryMetrics: document.querySelector("#summaryMetrics"),
  lessonGrid: document.querySelector("#lessonGrid"),
  claimNumbers: document.querySelector("#claimNumbers"),
  pairWalkthrough: document.querySelector("#pairWalkthrough"),
  stackDiagram: document.querySelector("#stackDiagram"),
  variableLadder: document.querySelector("#variableLadder"),
  modelStats: document.querySelector("#modelStats"),
  positionVectors: document.querySelector("#positionVectors"),
  positionTable: document.querySelector("#positionTable"),
  attentionMatrix: document.querySelector("#attentionMatrix"),
  easyEquation: document.querySelector("#easyEquation"),
  qkvPanel: document.querySelector("#qkvPanel"),
  scoreTable: document.querySelector("#scoreTable"),
  scaleCompare: document.querySelector("#scaleCompare"),
  headsGrid: document.querySelector("#headsGrid"),
  maskMatrix: document.querySelector("#maskMatrix"),
  crossAttentionMatrix: document.querySelector("#crossAttentionMatrix"),
  crossOutput: document.querySelector("#crossOutput"),
  decoderNotes: document.querySelector("#decoderNotes"),
  encoderFlow: document.querySelector("#encoderFlow"),
  normPanel: document.querySelector("#normPanel"),
  learningRateCurve: document.querySelector("#learningRateCurve"),
  trainingStats: document.querySelector("#trainingStats"),
  resultsGrid: document.querySelector("#resultsGrid"),
  metrics: document.querySelector("#metrics"),
  topLinks: document.querySelector("#topLinks"),
};

function escapeHTML(value) {
  return String(value).replace(/[&<>"']/g, (char) => {
    const entities = { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" };
    return entities[char];
  });
}

function tokenize(text) {
  const tokens = text
    .toLowerCase()
    .replace(/[^\p{L}\p{N}\s'-]/gu, " ")
    .trim()
    .split(/\s+/)
    .filter(Boolean);

  return tokens.length ? tokens.slice(0, 10) : ["token"];
}

function hash(input) {
  let value = 2166136261;
  for (const char of input) {
    value ^= char.codePointAt(0);
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
  return normalize(Array.from({ length: DEMO_DIM }, (_, index) => seededValue(seed, index)));
}

function positionVector(position) {
  return Array.from({ length: DEMO_DIM }, (_, index) => {
    const denominator = Math.pow(10000, (2 * Math.floor(index / 2)) / DEMO_DIM);
    const angle = position / denominator;
    return index % 2 === 0 ? Math.sin(angle) : Math.cos(angle);
  });
}

function vectorForToken(token, index) {
  const tokenVector = embedding(token);
  const posVector = positionVector(index);
  return normalize(tokenVector.map((value, vectorIndex) => value + posVector[vectorIndex] * 0.45));
}

function projection(vector, head, role) {
  const roleOffset = { q: 11, k: 23, v: 37 }[role];
  return normalize(
    vector.map((value, index) => {
      const turn = Math.sin((index + 1) * (head + 1 + roleOffset) * 0.41);
      const mix = vector[(index + head + roleOffset) % vector.length] * 0.38;
      const bias = Math.cos((roleOffset + index + head) * 0.17) * 0.06;
      return value * turn + mix + bias;
    }),
  );
}

function dot(a, b) {
  return a.reduce((sum, value, index) => sum + value * b[index], 0);
}

function softmax(values) {
  const finiteValues = values.filter(Number.isFinite);
  const max = Math.max(...finiteValues);
  const exp = values.map((value) => (Number.isFinite(value) ? Math.exp(value - max) : 0));
  const total = exp.reduce((sum, value) => sum + value, 0) || 1;
  return exp.map((value) => value / total);
}

function weightedSum(weights, values) {
  return Array.from({ length: DEMO_DIM }, (_, dim) =>
    weights.reduce((sum, weight, tokenIndex) => sum + weight * values[tokenIndex][dim], 0),
  );
}

function computeHead(head, options = {}) {
  const useScale = options.scale ?? state.scale;
  const useMask = options.mask ?? state.mask;
  const sharpness = options.sharpness ?? state.sharpness;
  const base = state.tokens.map(vectorForToken);
  const q = base.map((vector) => projection(vector, head, "q"));
  const k = base.map((vector) => projection(vector, head, "k"));
  const v = base.map((vector) => projection(vector, head, "v"));
  const raw = q.map((query) => k.map((key) => dot(query, key)));
  const usedScores = raw.map((row, queryIndex) =>
    row.map((score, keyIndex) => {
      if (useMask && keyIndex > queryIndex) {
        return -Infinity;
      }
      const scaled = useScale ? score / Math.sqrt(DEMO_DIM) : score;
      return scaled * sharpness;
    }),
  );
  const weights = usedScores.map(softmax);
  const output = weights.map((row) => weightedSum(row, v));
  return { base, q, k, v, raw, usedScores, weights, output };
}

function computeCrossHead(head) {
  const sourceBase = state.tokens.map(vectorForToken);
  const targetBase = state.targetTokens.map(vectorForToken);
  const q = targetBase.map((vector) => projection(vector, head + DEMO_HEADS, "q"));
  const k = sourceBase.map((vector) => projection(vector, head, "k"));
  const v = sourceBase.map((vector) => projection(vector, head, "v"));
  const raw = q.map((query) => k.map((key) => dot(query, key)));
  const usedScores = raw.map((row) =>
    row.map((score) => (state.scale ? score / Math.sqrt(DEMO_DIM) : score) * state.sharpness),
  );
  const weights = usedScores.map(softmax);
  const output = weights.map((row) => weightedSum(row, v));
  return { q, k, v, raw, usedScores, weights, output };
}

function entropy(weights) {
  return weights.reduce((sum, weight) => {
    if (weight <= 0) return sum;
    return sum - weight * Math.log2(weight);
  }, 0);
}

function topIndex(values) {
  return values.reduce((best, value, index) => (value > values[best] ? index : best), 0);
}

function format(value, digits = 2) {
  if (!Number.isFinite(value)) return "masked";
  return Number(value).toFixed(digits);
}

function formatLR(value) {
  return value < 0.001 ? value.toExponential(3) : value.toFixed(6);
}

function heatColor(weight, head) {
  const hue = [181, 7, 41, 110][head % 4];
  const alpha = 0.08 + Math.min(0.86, weight * 2.8);
  return `hsla(${hue}, 58%, 48%, ${alpha})`;
}

function learningRate(step, dModel) {
  return Math.pow(dModel, -0.5) * Math.min(Math.pow(step, -0.5), step * Math.pow(WARMUP_STEPS, -1.5));
}

function renderHeadButtons() {
  el.headButtons.innerHTML = "";
  for (let head = 0; head < DEMO_HEADS; head += 1) {
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

function renderTargetTokens() {
  el.targetTokenList.innerHTML = "";
  state.targetTokens.forEach((token, index) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = `token-chip${index === state.selectedTarget ? " is-selected" : ""}`;
    button.textContent = token;
    button.addEventListener("click", () => {
      state.selectedTarget = index;
      render();
    });
    el.targetTokenList.append(button);
  });
}

function renderSummaryMetrics(preset) {
  const items = [
    ["토큰 n", state.tokens.length],
    ["d_model", preset.dModel],
    ["head x d_k", `${preset.heads} x ${preset.dK}`],
    ["EN-DE BLEU", preset.bleuDe.toFixed(1)],
  ];

  el.summaryMetrics.innerHTML = items
    .map(([label, value]) => `<div class="mini-stat"><span>${label}</span><strong>${value}</strong></div>`)
    .join("");
}

function renderLessonGrid(preset, data) {
  const n = state.tokens.length;
  const row = data.weights[state.selectedToken];
  const best = topIndex(row);
  const softmaxSum = row.reduce((sum, weight) => sum + weight, 0);
  const lessons = [
    ["0", "큰 그림", "문장을 숫자 묶음으로 바꿔 다음 단어를 더 잘 맞힌다."],
    ["1", "토큰 세기", `지금 문장은 ${n}개 토큰이다.`],
    ["2", "짝 만들기", `${n} x ${n} = ${n * n}개의 비교 점수를 만든다.`],
    ["3", "위치 붙이기", `선택 토큰 위치는 ${state.selectedToken}번이다.`],
    ["4", "질문 고르기", `"${state.tokens[state.selectedToken]}" 토큰이 query 역할을 한다.`],
    ["5", "점수 만들기", `가장 큰 관심은 "${state.tokens[best]}" 쪽이다.`],
    ["6", "확률로 바꾸기", `softmax 뒤 가중치 합은 ${format(softmaxSum, 2)}이다.`],
    ["7", "값 섞기", `${n}개 value를 비율대로 섞어 새 벡터를 만든다.`],
    ["8", "여러 head", `데모는 ${DEMO_HEADS}개, 논문 ${preset.name}은 ${preset.heads}개 head다.`],
    ["9", "큰 변수", `d_model은 데모 ${DEMO_DIM}에서 논문 ${preset.dModel}로 커진다.`],
    ["10", "학습", `warmup은 ${WARMUP_STEPS.toLocaleString()} step이다.`],
    ["11", "결과", `${preset.name} EN-DE BLEU는 ${preset.bleuDe.toFixed(1)}이다.`],
  ];

  el.lessonGrid.innerHTML = lessons
    .map(
      ([step, title, text]) => `
        <article class="lesson-card">
          <span>${step}</span>
          <strong>${title}</strong>
          <p>${text}</p>
        </article>
      `,
    )
    .join("");
}

function renderClaimNumbers() {
  const n = state.tokens.length;
  const cards = [
    ["토큰 수 n", n, "문장을 자른 작은 말 조각 수"],
    ["선택 토큰 위치", state.selectedToken, "0부터 세는 자리 번호"],
    ["데모 벡터 길이", DEMO_DIM, "한 토큰을 숫자 8개로 표현"],
    ["논문 base 길이", 512, "실제 base는 숫자 512개"],
  ];

  el.claimNumbers.innerHTML = cards
    .map(
      ([label, value, detail], index) => `
        <article class="metric-card" style="border-top: 4px solid ${colors[index % colors.length]}">
          <span>${label}</span>
          <strong>${value}</strong>
          <p>${detail}</p>
        </article>
      `,
    )
    .join("");
}

function renderPairWalkthrough() {
  const n = state.tokens.length;
  const selected = state.tokens[state.selectedToken];
  const preview = state.tokens.slice(0, 5);
  const rows = preview
    .map(
      (token, index) => `
        <div class="pair-row">
          <span>query ${index}</span>
          <strong>${escapeHTML(token)}</strong>
          <em>${n}개 key를 본다</em>
        </div>
      `,
    )
    .join("");

  el.pairWalkthrough.innerHTML = `
    <div class="pair-sum">
      <span>전체 비교</span>
      <strong>${n} x ${n} = ${n * n}</strong>
      <p>${n}개 토큰이 각각 ${n}개 토큰을 보니 ${n * n}개의 점수가 생긴다.</p>
    </div>
    <div class="pair-sum">
      <span>선택 query</span>
      <strong>${escapeHTML(selected)} -> ${n} scores</strong>
      <p>한 줄만 보면 "${escapeHTML(selected)}"가 모든 key에게 점수를 주는 장면이다.</p>
    </div>
    <div class="pair-list">${rows}</div>
  `;
}

function renderVariableLadder(preset) {
  const rows = [
    ["작은 데모", DEMO_DIM, DEMO_HEADS, DEMO_DIM * 4, "브라우저에서 바로 계산"],
    ["논문 base", PAPER_PRESETS.base.dModel, PAPER_PRESETS.base.heads, PAPER_PRESETS.base.dFF, "65M parameters"],
    ["논문 big", PAPER_PRESETS.big.dModel, PAPER_PRESETS.big.heads, PAPER_PRESETS.big.dFF, "213M parameters"],
  ];

  el.variableLadder.innerHTML = `
    <article class="ladder-note">
      <span>변수 레벨업</span>
      <strong>${preset.name}</strong>
      <p>처음에는 작은 숫자로 감각을 잡고, 뒤에서는 논문 숫자로 같은 구조를 키운다.</p>
    </article>
    <div class="ladder-table">
      ${rows
        .map(
          ([name, dModel, heads, dFF, note]) => `
            <div class="ladder-row">
              <strong>${name}</strong>
              <span>d_model ${dModel}</span>
              <span>heads ${heads}</span>
              <span>d_ff ${dFF}</span>
              <em>${note}</em>
            </div>
          `,
        )
        .join("")}
    </div>
  `;
}

function renderStackDiagram(preset) {
  function column(title, blocks) {
    const rows = Array.from({ length: preset.layers }, (_, index) => {
      const layer = preset.layers - index;
      return `
        <div class="layer-row">
          <span class="layer-index">L${layer}</span>
          <div class="layer-blocks">
            ${blocks.map((block) => `<span>${block}</span>`).join("")}
          </div>
        </div>
      `;
    }).join("");

    return `
      <div class="stack-column">
        <h3>${title}</h3>
        <div class="layer-ladder">${rows}</div>
      </div>
    `;
  }

  el.stackDiagram.innerHTML = [
    column("Encoder x 6", ["Self-attn", "FFN", "Add+Norm"]),
    column("Decoder x 6", ["Masked self", "Cross-attn", "FFN"]),
  ].join("");
}

function renderDefinitionList(target, rows) {
  target.innerHTML = rows
    .map(([term, value]) => `<div><dt>${term}</dt><dd>${value}</dd></div>`)
    .join("");
}

function renderModelStats(preset) {
  renderDefinitionList(el.modelStats, [
    ["층 N", preset.layers],
    ["d_model", preset.dModel],
    ["d_ff", preset.dFF],
    ["head 수 h", preset.heads],
    ["d_k / d_v", `${preset.dK} / ${preset.dV}`],
    ["파라미터", `${preset.paramsM}M`],
    ["dropout", preset.dropout],
    ["학습 step", preset.trainSteps.toLocaleString()],
  ]);
}

function renderVector(name, values, color) {
  const row = document.createElement("div");
  row.className = "vector-row";

  const label = document.createElement("div");
  label.className = "vector-label";
  label.innerHTML = `<span>${name}</span><span>${values.map((value) => format(value, 2)).join(" ")}</span>`;

  const bars = document.createElement("div");
  bars.className = "bars";
  values.forEach((value) => {
    const bar = document.createElement("span");
    bar.className = `bar${value < 0 ? " negative" : ""}`;
    bar.style.height = `${Math.max(4, Math.abs(value) * 74)}px`;
    bar.style.background = value < 0 ? "#c89013" : color;
    bars.append(bar);
  });

  row.append(label, bars);
  return row;
}

function renderPositionSection() {
  const token = state.tokens[state.selectedToken];
  const tokenVector = embedding(token);
  const posVector = positionVector(state.selectedToken);
  const combined = vectorForToken(token, state.selectedToken);
  const stack = document.createElement("div");
  stack.className = "vector-stack";
  stack.append(
    renderVector(`token("${token}")`, tokenVector, "#456fb0"),
    renderVector(`position(${state.selectedToken})`, posVector, "#4b8f55"),
    renderVector("embedding + position", combined, "#df5c4d"),
  );

  el.positionVectors.innerHTML = `<span class="vector-title">8차원 데모 벡터</span>`;
  el.positionVectors.append(stack);

  const rows = posVector
    .map((value, dim) => {
      const denominator = Math.pow(10000, (2 * Math.floor(dim / 2)) / DEMO_DIM);
      const fn = dim % 2 === 0 ? "sin" : "cos";
      return `
        <tr>
          <td>${dim}</td>
          <td>${fn}(pos / ${format(denominator, 0)})</td>
          <td>${format(value, 4)}</td>
        </tr>
      `;
    })
    .join("");

  el.positionTable.innerHTML = `
    <span class="table-caption">선택 위치 ${state.selectedToken}의 positional encoding</span>
    <table>
      <thead><tr><th>dim</th><th>formula</th><th>value</th></tr></thead>
      <tbody>${rows}</tbody>
    </table>
  `;
}

function renderEasyEquation(data) {
  const queryIndex = state.selectedToken;
  const row = data.weights[queryIndex];
  const best = topIndex(row);
  const raw = data.raw[queryIndex][best];
  const used = data.usedScores[queryIndex][best];
  const scaleText = state.scale ? `sqrt(${DEMO_DIM}) = ${format(Math.sqrt(DEMO_DIM), 2)}로 나눔` : "스케일 나누기 꺼짐";
  const sum = row.reduce((total, weight) => total + weight, 0);

  const steps = [
    ["1", "query", `"${state.tokens[queryIndex]}"가 질문한다`],
    ["2", "top key", `"${state.tokens[best]}" 점수가 제일 크다`],
    ["3", "raw score", format(raw, 3)],
    ["4", "scale", `${scaleText} -> ${format(used, 3)}`],
    ["5", "softmax", `${(row[best] * 100).toFixed(1)}% 관심`],
    ["6", "합계", `모든 확률 합 ${format(sum, 2)}`],
  ];

  el.easyEquation.innerHTML = steps
    .map(
      ([step, title, value]) => `
        <article class="equation-step">
          <span>${step}</span>
          <strong>${title}</strong>
          <p>${value}</p>
        </article>
      `,
    )
    .join("");
}

function renderMatrix(data) {
  const count = state.tokens.length;
  el.attentionMatrix.style.gridTemplateColumns = `repeat(${count}, minmax(42px, 1fr))`;
  el.attentionMatrix.innerHTML = "";

  data.weights.forEach((row, queryIndex) => {
    row.forEach((weight, keyIndex) => {
      const masked = state.mask && keyIndex > queryIndex;
      const cell = document.createElement("button");
      cell.type = "button";
      cell.className = `matrix-cell${queryIndex === state.selectedToken ? " is-query" : ""}${
        masked ? " is-masked" : ""
      }`;
      cell.style.background = masked ? "" : heatColor(weight, state.selectedHead);
      cell.textContent = masked ? "--" : format(weight, 2);
      cell.title = `${state.tokens[queryIndex]} -> ${state.tokens[keyIndex]}`;
      cell.addEventListener("click", () => {
        state.selectedToken = queryIndex;
        render();
      });
      el.attentionMatrix.append(cell);
    });
  });
}

function renderQKV(data, preset) {
  const index = state.selectedToken;
  const stack = document.createElement("div");
  stack.className = "vector-stack";
  stack.append(
    renderVector("Q selected", data.q[index], colors[state.selectedHead]),
    renderVector("K selected", data.k[index], "#456fb0"),
    renderVector("V selected", data.v[index], "#4b8f55"),
    renderVector("Output", data.output[index], "#df5c4d"),
  );

  el.qkvPanel.innerHTML = `
    <span class="vector-title">Head ${state.selectedHead + 1} projection</span>
    <p>작은 데모는 숫자 ${DEMO_DIM}개로 본다. 논문 ${preset.name}은 d_model=${preset.dModel}에서 head마다 d_k=${preset.dK}, d_v=${preset.dV}로 나눠 본다.</p>
  `;
  el.qkvPanel.append(stack);
}

function renderScoreTable(data) {
  const queryIndex = state.selectedToken;
  const maxScore = Math.max(...data.usedScores[queryIndex].filter(Number.isFinite));
  const rows = state.tokens
    .map((token, keyIndex) => {
      const raw = data.raw[queryIndex][keyIndex];
      const used = data.usedScores[queryIndex][keyIndex];
      const expValue = Number.isFinite(used) ? Math.exp(used - maxScore) : 0;
      const weight = data.weights[queryIndex][keyIndex];
      const isTop = keyIndex === topIndex(data.weights[queryIndex]);
      return `
        <tr${isTop ? ' style="background:rgba(34, 211, 238, 0.12)"' : ""}>
          <td>${escapeHTML(token)}</td>
          <td>${format(raw, 3)}</td>
          <td>${format(used, 3)}</td>
          <td>${format(expValue, 3)}</td>
          <td>${(weight * 100).toFixed(1)}%</td>
        </tr>
      `;
    })
    .join("");

  el.scoreTable.innerHTML = `
    <span class="table-caption">query = ${escapeHTML(state.tokens[queryIndex])}, head = H${state.selectedHead + 1}</span>
    <table>
      <thead>
        <tr><th>key</th><th>q dot k</th><th>used score</th><th>exp</th><th>weight</th></tr>
      </thead>
      <tbody>${rows}</tbody>
    </table>
  `;
}

function renderScaleCompare() {
  const scaled = computeHead(state.selectedHead, { scale: true });
  const unscaled = computeHead(state.selectedHead, { scale: false });
  const rows = [
    ["scaled", scaled.weights[state.selectedToken], "#0f8b8d"],
    ["no scale", unscaled.weights[state.selectedToken], "#df5c4d"],
  ];

  const body = rows
    .map(([label, weights, color]) => {
      const best = topIndex(weights);
      const maxWeight = weights[best];
      return `
        <div class="scale-row">
          <span>${label}</span>
          <div class="track"><span style="width:${Math.max(3, maxWeight * 100)}%; background:${color}"></span></div>
          <strong>${(maxWeight * 100).toFixed(1)}%</strong>
        </div>
        <p>${label}: top key는 <strong>${escapeHTML(state.tokens[best])}</strong>, entropy ${format(entropy(weights), 2)} bits</p>
      `;
    })
    .join("");

  el.scaleCompare.innerHTML = `
    <span class="table-caption">sqrt(d_k) scaling effect</span>
    <p>숫자가 너무 커지면 softmax가 한 칸에만 몰린다. 그래서 논문 base는 sqrt(64)=8로 나누고, 이 데모는 sqrt(${DEMO_DIM})=${format(Math.sqrt(DEMO_DIM), 2)}로 나눈다.</p>
    <div class="scale-bars">${body}</div>
  `;
}

function renderHeads() {
  el.headsGrid.innerHTML = "";
  for (let head = 0; head < DEMO_HEADS; head += 1) {
    const data = computeHead(head);
    const row = data.weights[state.selectedToken];
    const best = topIndex(row);
    const card = document.createElement("article");
    card.className = "head-card";
    card.style.borderTop = `4px solid ${colors[head]}`;
    card.style.setProperty("--token-count", state.tokens.length);
    card.innerHTML = `
      <h3>Head ${head + 1}</h3>
      <div class="head-meta">
        <span>1등 key: ${escapeHTML(state.tokens[best])}</span>
        <span>entropy ${format(entropy(row), 2)}</span>
      </div>
    `;

    data.weights.forEach((weights, queryIndex) => {
      const sparkRow = document.createElement("div");
      sparkRow.className = "spark-row";
      const token = document.createElement("span");
      token.textContent = state.tokens[queryIndex];
      const spark = document.createElement("div");
      spark.className = "spark";
      weights.forEach((weight, keyIndex) => {
        const block = document.createElement("span");
        block.style.background = state.mask && keyIndex > queryIndex ? "#1f2937" : heatColor(weight, head);
        spark.append(block);
      });
      sparkRow.append(token, spark);
      card.append(sparkRow);
    });

    card.addEventListener("click", () => {
      state.selectedHead = head;
      render();
    });
    el.headsGrid.append(card);
  }
}

function renderMaskSection() {
  const count = state.targetTokens.length;
  el.maskMatrix.style.gridTemplateColumns = `repeat(${count}, minmax(34px, 1fr))`;
  el.maskMatrix.innerHTML = "";

  state.targetTokens.forEach((_, queryIndex) => {
    state.targetTokens.forEach((__, keyIndex) => {
      const allowed = keyIndex <= queryIndex;
      const cell = document.createElement("span");
      cell.className = `mask-cell${allowed ? "" : " blocked"}`;
      cell.textContent = allowed ? "OK" : "--";
      cell.title = `${state.targetTokens[queryIndex]} -> ${state.targetTokens[keyIndex]}`;
      el.maskMatrix.append(cell);
    });
  });

  const allowedCount = state.selectedTarget + 1;
  const blockedCount = count - allowedCount;
  el.decoderNotes.innerHTML = `
    <strong>target query: ${escapeHTML(state.targetTokens[state.selectedTarget])}</strong>
    <p>masked self-attention은 target key ${allowedCount}개만 보고, 미래 key ${blockedCount}개는 막는다.</p>
    <p>그 다음 cross-attention은 target query로 source 토큰 전체를 다시 읽는다.</p>
    <p>전체 마스크 토글: <strong>${state.mask ? "on" : "off"}</strong></p>
  `;
}

function renderCrossAttention(data) {
  const sourceCount = state.tokens.length;
  el.crossAttentionMatrix.style.gridTemplateColumns = `repeat(${sourceCount}, minmax(48px, 1fr))`;
  el.crossAttentionMatrix.innerHTML = "";

  data.weights.forEach((row, targetIndex) => {
    row.forEach((weight, sourceIndex) => {
      const cell = document.createElement("button");
      cell.type = "button";
      cell.className = `cross-cell${targetIndex === state.selectedTarget ? " is-target" : ""}`;
      cell.style.background = heatColor(weight, state.selectedHead);
      cell.textContent = format(weight, 2);
      cell.title = `${state.targetTokens[targetIndex]} -> ${state.tokens[sourceIndex]}`;
      cell.addEventListener("click", () => {
        state.selectedTarget = targetIndex;
        render();
      });
      el.crossAttentionMatrix.append(cell);
    });
  });

  const row = data.weights[state.selectedTarget];
  const best = topIndex(row);
  const ranked = row
    .map((weight, index) => ({ weight, token: state.tokens[index] }))
    .sort((a, b) => b.weight - a.weight)
    .slice(0, 4)
    .map((item) => `<span>${escapeHTML(item.token)} ${(item.weight * 100).toFixed(1)}%</span>`)
    .join("");

  el.crossOutput.innerHTML = `
    <p><strong>${escapeHTML(state.targetTokens[state.selectedTarget])}</strong>가 source 토큰 <strong>${escapeHTML(
      state.tokens[best],
    )}</strong>를 가장 크게 본다.</p>
    <p>${ranked}</p>
  `;
}

function renderFlow(preset) {
  const steps = [
    ["1. 숫자 붙이기", `토큰 숫자와 위치 숫자를 더해 길이 ${preset.dModel} 표현을 만든다.`],
    ["2. 여러 head", `${preset.heads}개 head가 서로 다른 기준으로 토큰을 본다.`],
    ["3. 다시 더하기", "새로 계산한 값과 원래 값을 더해 중요한 정보를 잃지 않는다."],
    ["4. FFN", `${preset.dModel} -> ${preset.dFF} -> ${preset.dModel}로 키웠다가 줄인다.`],
    ["5. 정리", "값의 평균과 크기를 정리해 다음 층이 쓰기 좋게 만든다."],
  ];

  el.encoderFlow.innerHTML = steps
    .map(
      ([title, text], index) => `
        <article class="flow-step" style="border-top: 4px solid ${colors[index % colors.length]}">
          <h3>${title}</h3>
          <p>${text}</p>
        </article>
      `,
    )
    .join("");
}

function renderNormPanel(data) {
  const index = state.selectedToken;
  const residual = data.base[index].map((value, dim) => value + data.output[index][dim]);
  const mean = residual.reduce((sum, value) => sum + value, 0) / residual.length;
  const variance =
    residual.reduce((sum, value) => sum + Math.pow(value - mean, 2), 0) / residual.length;
  const normalized = residual.map((value) => (value - mean) / Math.sqrt(variance + 1e-6));

  el.normPanel.innerHTML = `
    <strong>LayerNorm(x + Sublayer(x)) 데모</strong>
    <p>residual 평균 ${format(mean, 3)}, 분산 ${format(variance, 3)}</p>
    <div class="vector-stack"></div>
  `;
  const stack = el.normPanel.querySelector(".vector-stack");
  stack.append(
    renderVector("x + attention", residual, "#456fb0"),
    renderVector("normalized", normalized, "#79599f"),
  );
}

function renderLearningRate(preset) {
  el.trainStep.max = String(preset.trainSteps);
  if (Number(el.trainStep.value) > preset.trainSteps) {
    el.trainStep.value = String(preset.trainSteps);
  }

  const step = Number(el.trainStep.value);
  const maxStep = preset.trainSteps;
  const peak = learningRate(WARMUP_STEPS, preset.dModel);
  const width = 620;
  const height = 180;
  const left = 44;
  const right = 18;
  const top = 18;
  const bottom = 34;
  const plotWidth = width - left - right;
  const plotHeight = height - top - bottom;

  const points = Array.from({ length: 140 }, (_, index) => {
    const progress = index / 139;
    const currentStep = Math.max(1, Math.round(1 + progress * (maxStep - 1)));
    const x = left + progress * plotWidth;
    const y = top + plotHeight - (learningRate(currentStep, preset.dModel) / peak) * plotHeight;
    return `${x.toFixed(1)},${y.toFixed(1)}`;
  }).join(" ");

  const markerProgress = (step - 1) / (maxStep - 1);
  const markerX = left + markerProgress * plotWidth;
  const markerY = top + plotHeight - (learningRate(step, preset.dModel) / peak) * plotHeight;
  const warmupX = left + ((WARMUP_STEPS - 1) / (maxStep - 1)) * plotWidth;

  el.learningRateCurve.innerHTML = `
    <line class="curve-axis" x1="${left}" y1="${top + plotHeight}" x2="${width - right}" y2="${top + plotHeight}"></line>
    <line class="curve-axis" x1="${left}" y1="${top}" x2="${left}" y2="${top + plotHeight}"></line>
    <line class="curve-axis" x1="${warmupX.toFixed(1)}" y1="${top}" x2="${warmupX.toFixed(1)}" y2="${top + plotHeight}" stroke-dasharray="4 4"></line>
    <polyline class="curve-line" points="${points}"></polyline>
    <circle class="curve-marker" cx="${markerX.toFixed(1)}" cy="${markerY.toFixed(1)}" r="6"></circle>
    <text class="curve-label" x="${left}" y="${height - 10}">1</text>
    <text class="curve-label" x="${warmupX + 4}" y="${top + 14}">warmup 4000</text>
    <text class="curve-label" x="${width - 112}" y="${height - 10}">${maxStep.toLocaleString()} steps</text>
    <text class="curve-label" x="${left + 4}" y="${top + 12}">peak ${formatLR(peak)}</text>
  `;

  renderDefinitionList(el.trainingStats, [
    ["선택 step", step.toLocaleString()],
    ["학습률", formatLR(learningRate(step, preset.dModel))],
    ["warmup", WARMUP_STEPS.toLocaleString()],
    ["optimizer", "Adam 0.9 / 0.98 / 1e-9"],
    ["batch token", "25K src + 25K tgt"],
    ["1 step 시간", `${preset.stepTime}s`],
    ["전체 시간", preset.trainTime],
  ]);
}

function renderResults() {
  const base = PAPER_PRESETS.base;
  const big = PAPER_PRESETS.big;
  const items = [
    ["Transformer base", `${base.bleuDe} / ${base.bleuFr}`, "BLEU EN-DE / EN-FR, 65M params"],
    ["Transformer big", `${big.bleuDe} / ${big.bleuFr}`, "BLEU EN-DE / EN-FR, 213M params"],
    ["Training data", "4.5M / 36M", "sentence pairs: EN-DE / EN-FR"],
    ["Hardware", "8 x P100", "base 12h, big 3.5 days"],
  ];

  el.resultsGrid.innerHTML = items
    .map(
      ([label, value, detail], index) => `
        <article class="result-item" style="border-top: 4px solid ${colors[index % colors.length]}">
          <span>${label}</span>
          <strong>${value}</strong>
          <p>${detail}</p>
        </article>
      `,
    )
    .join("");
}

function renderInspector(data) {
  const row = data.weights[state.selectedToken];
  const raw = data.raw[state.selectedToken];
  const used = data.usedScores[state.selectedToken];
  const best = topIndex(row);

  renderDefinitionList(el.metrics, [
    ["query", escapeHTML(state.tokens[state.selectedToken])],
    ["head", `H${state.selectedHead + 1}`],
    ["1등 key", escapeHTML(state.tokens[best])],
    ["가장 큰 비율", `${(row[best] * 100).toFixed(1)}%`],
    ["raw dot", format(raw[best], 3)],
    ["사용 점수", format(used[best], 3)],
    ["entropy", `${format(entropy(row), 2)} bits`],
  ]);

  el.topLinks.innerHTML = row
    .map((weight, index) => ({ weight, token: state.tokens[index] }))
    .sort((a, b) => b.weight - a.weight)
    .slice(0, 5)
    .map((item) => `<li><strong>${escapeHTML(item.token)}</strong> ${(item.weight * 100).toFixed(1)}%</li>`)
    .join("");
}

function render() {
  state.tokens = tokenize(el.input.value);
  state.targetTokens = tokenize(el.targetInput.value);
  state.selectedToken = Math.min(state.selectedToken, state.tokens.length - 1);
  state.selectedTarget = Math.min(state.selectedTarget, state.targetTokens.length - 1);
  state.sharpness = Number(el.sharpness.value) / 100;
  state.scale = el.scaleToggle.checked;
  state.mask = el.maskToggle.checked;
  state.preset = el.modelPreset.value;

  const preset = PAPER_PRESETS[state.preset];
  const data = computeHead(state.selectedHead);
  const crossData = computeCrossHead(state.selectedHead);

  renderHeadButtons();
  renderTokens();
  renderTargetTokens();
  renderSummaryMetrics(preset);
  renderLessonGrid(preset, data);
  renderClaimNumbers();
  renderPairWalkthrough();
  renderStackDiagram(preset);
  renderVariableLadder(preset);
  renderModelStats(preset);
  renderPositionSection();
  renderEasyEquation(data);
  renderMatrix(data);
  renderQKV(data, preset);
  renderScoreTable(data);
  renderScaleCompare();
  renderHeads();
  renderMaskSection();
  renderCrossAttention(crossData);
  renderFlow(preset);
  renderNormPanel(data);
  renderLearningRate(preset);
  renderResults();
  renderInspector(data);
}

el.input.addEventListener("input", render);
el.targetInput.addEventListener("input", render);
el.modelPreset.addEventListener("change", render);
el.sharpness.addEventListener("input", render);
el.scaleToggle.addEventListener("change", render);
el.maskToggle.addEventListener("change", render);
el.trainStep.addEventListener("input", render);

render();
