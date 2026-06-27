"use strict";

const MAX_POINTS = 2000;   // render cap per chart
const ROLL_WINDOW = 100;   // rolling-average window (episodes)

// Full series kept in memory; charts render a downsampled view.
const data = { episode: [], reward: [], loss: [], epsilon: [] };

function makeChart(id, label, color) {
  return new Chart(document.getElementById(id).getContext("2d"), {
    type: "line",
    data: { labels: [], datasets: [{ label, data: [], borderColor: color, pointRadius: 0, borderWidth: 1.5 }] },
    options: {
      animation: false,
      scales: { x: { title: { display: true, text: "episode" } } },
      plugins: { legend: { display: true } },
    },
  });
}

const rewardChart = new Chart(document.getElementById("chart-reward").getContext("2d"), {
  type: "line",
  data: {
    labels: [],
    datasets: [
      { label: "reward", data: [], borderColor: "#1565c0", pointRadius: 0, borderWidth: 1 },
      { label: "rolling avg (100)", data: [], borderColor: "#d84315", pointRadius: 0, borderWidth: 2 },
    ],
  },
  options: { animation: false, scales: { x: { title: { display: true, text: "episode" } } } },
});
const lossChart = makeChart("chart-loss", "avg_loss", "#6a1b9a");
const epsilonChart = makeChart("chart-epsilon", "epsilon", "#2e7d32");

// Downsample y[] to <= MAX_POINTS, preserving per-bucket min & max so spikes/
// crashes are never smoothed away. Returns parallel {labels, values}.
function downsample(xs, ys) {
  const n = ys.length;
  if (n <= MAX_POINTS) return { labels: xs.slice(), values: ys.slice() };
  const bucket = Math.ceil(n / (MAX_POINTS / 2)); // 2 points (min,max) per bucket
  const labels = [], values = [];
  for (let i = 0; i < n; i += bucket) {
    let lo = i, hi = i, min = ys[i], max = ys[i];
    for (let j = i; j < Math.min(i + bucket, n); j++) {
      if (ys[j] < min) { min = ys[j]; lo = j; }
      if (ys[j] > max) { max = ys[j]; hi = j; }
    }
    // emit in x-order so the line stays monotonic in x
    if (lo <= hi) { labels.push(xs[lo], xs[hi]); values.push(min, max); }
    else { labels.push(xs[hi], xs[lo]); values.push(max, min); }
  }
  return { labels, values };
}

function rollingAvg(ys, w) {
  const out = [];
  let sum = 0;
  for (let i = 0; i < ys.length; i++) {
    sum += ys[i];
    if (i >= w) sum -= ys[i - w];
    out.push(sum / Math.min(i + 1, w));
  }
  return out;
}

function redraw() {
  const r = downsample(data.episode, data.reward);
  const avg = rollingAvg(data.reward, ROLL_WINDOW);
  const ra = downsample(data.episode, avg);
  rewardChart.data.labels = r.labels;
  rewardChart.data.datasets[0].data = r.values;
  rewardChart.data.datasets[1].data = ra.values; // shares the reward x-buckets closely enough
  rewardChart.update("none");

  // loss may contain nulls (no training yet) — keep nulls so Chart.js gaps them.
  const lx = [], ly = [];
  for (let i = 0; i < data.episode.length; i++) { lx.push(data.episode[i]); ly.push(data.loss[i]); }
  const l = downsample(lx, ly.map((v) => (v === null ? NaN : v)));
  lossChart.data.labels = l.labels;
  lossChart.data.datasets[0].data = l.values.map((v) => (Number.isNaN(v) ? null : v));
  lossChart.update("none");

  const e = downsample(data.episode, data.epsilon);
  epsilonChart.data.labels = e.labels;
  epsilonChart.data.datasets[0].data = e.values;
  epsilonChart.update("none");
}

function updateCards() {
  const n = data.episode.length;
  if (n === 0) return;
  const fmt = (v) => (v === null || v === undefined ? "–" : (+v).toFixed(2));
  document.getElementById("stat-episode").textContent = data.episode[n - 1];
  document.getElementById("stat-reward").textContent = fmt(data.reward[n - 1]);
  document.getElementById("stat-best").textContent = fmt(Math.max(...data.reward));
  const recent = data.reward.slice(-ROLL_WINDOW);
  document.getElementById("stat-avg").textContent = fmt(recent.reduce((a, b) => a + b, 0) / recent.length);
  document.getElementById("stat-steps").textContent = data.steps_last ?? "–";
}

function addRow(row) {
  data.episode.push(row.episode);
  data.reward.push(row.reward);
  data.loss.push(row.avg_loss); // null stays null
  data.epsilon.push(row.epsilon);
  data.steps_last = row.global_step;
}

function reset() {
  data.episode = []; data.reward = []; data.loss = []; data.epsilon = []; data.steps_last = null;
}

function setStatus(cls, text) {
  const el = document.getElementById("status");
  el.className = "pill " + cls;
  el.textContent = text;
}

function connect() {
  const ws = new WebSocket(`ws://${location.host}/ws`);
  ws.onopen = () => setStatus("live", "live");
  ws.onclose = () => { setStatus("disconnected", "disconnected"); setTimeout(connect, 1500); };
  ws.onerror = () => ws.close();
  ws.onmessage = (ev) => {
    const msg = JSON.parse(ev.data);
    if (msg.type === "snapshot") {
      reset();
      msg.rows.forEach(addRow);
    } else if (msg.type === "append") {
      addRow(msg.row);
    }
    if (data.episode.length > 0) setStatus("live", "live");
    redraw();
    updateCards();
  };
}

connect();
