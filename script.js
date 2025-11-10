document.addEventListener("DOMContentLoaded", () => {
  const weightsInput = document.getElementById("weights");
  const biasInput = document.getElementById("bias");
  const lrInput = document.getElementById("lr");
  const epochInput = document.getElementById("epochs");
  const dataText = document.getElementById("data-text");
  const logOutput = document.getElementById("log-output");
  const errorCtx = document.getElementById("error-chart").getContext("2d");
  const weightsCtx = document.getElementById("weights-chart").getContext("2d");

  document.getElementById("train-button").addEventListener("click", startTraining);
  document.getElementById("load-and").addEventListener("click", () => loadDataset("and"));
  document.getElementById("load-or").addEventListener("click", () => loadDataset("or"));
  document.getElementById("load-xor").addEventListener("click", () => loadDataset("xor"));

  let errorChart, weightsChart;

  function log(msg, cls = "log-info") {
    const el = document.createElement("div");
    el.className = cls;
    el.textContent = msg;
    logOutput.appendChild(el);
    logOutput.scrollTop = logOutput.scrollHeight;
  }

  function parseDataset(text) {
    const lines = text.trim().split("\n");
    const X = [], y = [];
    for (const line of lines) {
      const parts = line.split(",").map(p => parseFloat(p.trim()));
      if (parts.length >= 3) {
        X.push(parts.slice(0, -1));
        y.push(parts.at(-1));
      }
    }
    return { X, y };
  }

  function dot(v1, v2) {
    return v1.reduce((sum, val, i) => sum + val * v2[i], 0);
  }

  function loadDataset(type) {
    const presets = {
      and: "0, 0, 0\n0, 1, 0\n1, 0, 0\n1, 1, 1",
      or:  "0, 0, 0\n0, 1, 1\n1, 0, 1\n1, 1, 1",
      xor: "0, 0, 0\n0, 1, 1\n1, 0, 1\n1, 1, 0",
    };
    dataText.value = presets[type];
    log(`Yüklendi: ${type.toUpperCase()}`, "log-info");
    // Güvenli başlangıç değerleri
    weightsInput.placeholder = "Örn: 0.1, 0.1";
    biasInput.placeholder = "Örn: -0.2";
  }

  function drawCharts(errors, weightHistory, rule) {
    if (errorChart) errorChart.destroy();
    if (weightsChart) weightsChart.destroy();

    errorChart = new Chart(errorCtx, {
      type: "line",
      data: {
        labels: errors.map((_, i) => i + 1),
        datasets: [{
          label: rule === "adaline" ? "MSE" : "Toplam Hata",
          data: errors,
          borderColor: "red",
          backgroundColor: "rgba(255,0,0,0.1)"
        }]
      },
      options: { responsive: true, plugins: { title: { display: true, text: "Hata Grafiği" } } }
    });

    const datasets = weightHistory[0].map((_, i) => ({
      label: `w${i + 1}`,
      data: weightHistory.map(w => w[i]),
      borderColor: ["#0078d7", "#28a745", "#fd7e14"][i % 3],
      fill: false
    }));

    weightsChart = new Chart(weightsCtx, {
      type: "line",
      data: { labels: weightHistory.map((_, i) => i), datasets },
      options: { responsive: true, plugins: { title: { display: true, text: "Ağırlık Değişimi" } } }
    });
  }

  function startTraining() {
    logOutput.innerHTML = "";
    try {
      let w = weightsInput.value.split(",").map(v => parseFloat(v.trim()));
      let b = parseFloat(biasInput.value);
      const lr = parseFloat(lrInput.value);
      const epochs = parseInt(epochInput.value);
      const rule = document.querySelector("input[name='learning-rule']:checked").value;
      const { X, y } = parseDataset(dataText.value);

      if (w.length !== X[0].length) throw new Error("Ağırlık sayısı ile giriş sayısı eşleşmiyor.");

      const errors = [];
      const wHist = [ [...w] ];
      let solved = false;

      log("--- Eğitim Başlatılıyor ---", "log-header");

      for (let epoch = 0; epoch < epochs; epoch++) {
        let sumErr = 0, sumSqErr = 0;
        let epochPredictions = [];

        for (let i = 0; i < X.length; i++) {
          const net = dot(X[i], w) + b;
          const yPred = net >= 0 ? 1 : 0;
          const err = (rule === "adaline") ? (y[i] - net) : (y[i] - yPred);

          for (let j = 0; j < w.length; j++) w[j] += lr * err * X[i][j];
          b += lr * err;

          sumErr += Math.abs(y[i] - yPred);
          sumSqErr += (y[i] - net) ** 2;
          epochPredictions.push(yPred);
        }

        errors.push(rule === "adaline" ? sumSqErr / X.length : sumErr);
        wHist.push([...w]);

        log(`\nEpoch ${epoch + 1}:`, "log-header");
        log(`  Tahminler: [${epochPredictions.join(", ")}]`, "log-info");
        log(`  w = [${w.map(v => v.toFixed(3)).join(", ")}], b = ${b.toFixed(3)}`, "log-info");
        if (rule === "adaline") log(`  MSE: ${(sumSqErr / X.length).toFixed(4)}`, "log-info");
        else log(`  Toplam Hata: ${sumErr}`, "log-info");

        if (sumErr === 0 && rule === "perceptron") {
          solved = true;
          log("\nPerceptron öğrenme tamamlandı ✅ Tüm örnekler doğru sınıflandırıldı.", "log-success");
          break;
        }
      }

      if (!solved && rule === "perceptron") {
        log("\n[UYARI] Eğitim tamamlandı fakat tüm örnekler doğru sınıflandırılamadı. XOR gibi lineer olarak ayrılamayan problem olabilir.", "log-warning");
      }

      log(`\nNihai ağırlıklar: [${w.map(v => v.toFixed(3)).join(", ")}]`, "log-success");
      log(`Nihai bias: ${b.toFixed(3)}`, "log-success");

      drawCharts(errors, wHist, rule);

    } catch (e) {
      log("[HATA] " + e.message, "log-error");
    }
  }
});
