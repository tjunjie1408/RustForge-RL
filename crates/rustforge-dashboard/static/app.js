"use strict";

const MAX_POINTS = 2000;   // render cap per chart
const ROLL_WINDOW = 100;   // rolling-average window (episodes)

// Series colors are theme-independent (chosen to read on both surfaces). Area
// fills use an 8-digit hex at ~20% alpha (color + "33"), which looks right over
// both the dark (#161b22) and light (#ffffff) surfaces — no per-theme fill logic.
const COLOR = { reward: "#f97316", avg: "#3b82f6", loss: "#a855f7", epsilon: "#22c55e" };

// Per-theme chart "chrome" (gridlines + axis ticks + legend text). Series colors
// above stay fixed across themes; only these change.
const THEME = {
  dark: { grid: "#2b3340", tick: "#8b949e" },
  light: { grid: "#e4e8ee", tick: "#5b6573" },
};

// Full series kept in memory; charts render a downsampled view.
const data = { episode: [], reward: [], loss: [], epsilon: [] };

// Shared chart options so all three charts share the same chrome (colored by
// applyChartTheme). maintainAspectRatio:false makes the chart fill its
// fixed-height wrapper instead of imposing its own aspect ratio.
function baseOptions(showLegend) {
  return {
    animation: false,
    responsive: true,
    maintainAspectRatio: false,
    scales: {
      x: { title: { display: true, text: "episode" }, grid: {}, ticks: {} },
      y: { grid: {}, ticks: {} },
    },
    plugins: { legend: { display: showLegend, labels: {} } },
  };
}

function makeChart(id, label, color) {
  return new Chart(document.getElementById(id).getContext("2d"), {
    type: "line",
    data: {
      labels: [],
      datasets: [
        {
          label,
          data: [],
          borderColor: color,
          backgroundColor: color + "33",
          fill: true,
          pointRadius: 0,
          borderWidth: 1.6,
        },
      ],
    },
    options: baseOptions(false),
  });
}

const rewardChart = new Chart(document.getElementById("chart-reward").getContext("2d"), {
  type: "line",
  data: {
    labels: [],
    datasets: [
      {
        label: "reward",
        data: [],
        borderColor: COLOR.reward,
        backgroundColor: COLOR.reward + "33",
        fill: true,
        pointRadius: 0,
        borderWidth: 1.4,
      },
      {
        label: "rolling avg (100)",
        data: [],
        borderColor: COLOR.avg,
        fill: false,
        pointRadius: 0,
        borderWidth: 2.4,
      },
    ],
  },
  options: baseOptions(true),
});
const lossChart = makeChart("chart-loss", "avg_loss", COLOR.loss);
const epsilonChart = makeChart("chart-epsilon", "epsilon", COLOR.epsilon);
const allCharts = [rewardChart, lossChart, epsilonChart];

// Re-theme already-instantiated charts. Chart.defaults.color only affects FUTURE
// charts, so reassign each live instance's chrome colors, then update.
function applyChartTheme(theme) {
  const c = THEME[theme] || THEME.dark;
  for (const chart of allCharts) {
    chart.options.scales.x.grid.color = c.grid;
    chart.options.scales.y.grid.color = c.grid;
    chart.options.scales.x.ticks.color = c.tick;
    chart.options.scales.y.ticks.color = c.tick;
    chart.options.scales.x.title.color = c.tick;
    chart.options.plugins.legend.labels.color = c.tick;
    chart.update("none");
  }
}

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
  document.getElementById("stat-best").textContent = fmt(data.reward.reduce((a, b) => (b > a ? b : a), -Infinity));
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
  const proto = location.protocol === "https:" ? "wss:" : "ws:";
  const ws = new WebSocket(`${proto}//${location.host}/ws`);
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

// Theme toggle: flip <html data-theme>, persist, re-theme the charts.
function currentTheme() {
  return document.documentElement.getAttribute("data-theme") === "light" ? "light" : "dark";
}
function setTheme(theme) {
  document.documentElement.setAttribute("data-theme", theme);
  try { localStorage.setItem("rf-theme", theme); } catch (e) { /* private mode: keep session theme */ }
  applyChartTheme(theme);
}
const toggleBtn = document.getElementById("theme-toggle");
if (toggleBtn) {
  toggleBtn.addEventListener("click", () => {
    setTheme(currentTheme() === "dark" ? "light" : "dark");
  });
}

applyChartTheme(currentTheme());
connect();
