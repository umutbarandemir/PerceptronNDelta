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

  // log fonksiyonu: HTML içeriğine izin veriyor ve sınıf ekliyor
  function log(msg, cls = "log-info") {
    const el = document.createElement("div");
    el.className = cls;
    el.innerHTML = msg; // <-- burada innerHTML kullanıyoruz, böylece span ve ikonlar çalışır
    logOutput.appendChild(el);
    logOutput.scrollTop = logOutput.scrollHeight;
  }

  function parseDataset(text) {
    const lines = text.trim().split("\n");
    const X = [], y = [];
    for (const line of lines) {
      const parts = line.split(",").map(p => parseFloat(p.trim()));
      if (parts.length >= 3 && !parts.some(p => Number.isNaN(p))) {
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
    log(`🔁 <span class="log-epoch">Yüklendi:</span> <strong>${type.toUpperCase()}</strong>`, "log-info");
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

  // Düzeltilmiş ve ikonlu startTraining
  function startTraining() {
    logOutput.innerHTML = "";
    try {
      let w = weightsInput.value.split(",").map(v => parseFloat(v.trim()));
      let b = parseFloat(biasInput.value);
      const lr = parseFloat(lrInput.value);
      const epochs = parseInt(epochInput.value);
      const rule = document.querySelector("input[name='learning-rule']:checked").value;
      const { X, y } = parseDataset(dataText.value);

      if (!X.length) throw new Error("Geçerli veri yok veya format hatası.");
      if (w.length !== X[0].length) throw new Error("Ağırlık sayısı ile giriş sayısı eşleşmiyor.");

      const wInitial = [...w];
      const bInitial = b;

      const errors = [];
      const wHist = [[...w]];
      let solved = false;

      log(`--- Eğitim Başlatılıyor ---`, "log-header");
      log(`🔰 Başlangıç: <span class="log-weight">w = [${w.map(v => v.toFixed(3)).join(", ")}]</span>, <span class="log-bias">b = ${b.toFixed(3)}</span>`, "log-info");

      for (let epoch = 0; epoch < epochs; epoch++) {
        let sumErr = 0, sumSqErr = 0;
        log(`\n🧩 <span class="log-epoch">Epoch ${epoch + 1}</span> başlatılıyor...`, "log-header");

        for (let i = 0; i < X.length; i++) {
          const net = dot(X[i], w) + b;
          const yPred = net >= 0 ? 1 : 0;
          const err = (rule === "adaline") ? (y[i] - net) : (y[i] - yPred);

          // Iteration-level log (net, pred, error) — mavi / kırmızı vurgulu
          log(
            `Epoch <strong>${epoch + 1}.${i + 1}</strong> | Girdi: [${X[i].join(", ")}] → Hedef: <strong>${y[i]}</strong> ` +
            `&nbsp; <span class="log-net">🔎 Net:</span> <strong>${net.toFixed(3)}</strong> ` +
            `&nbsp; <span class="log-pred">🧾 Tahmin:</span> <strong>${yPred}</strong> ` +
            `&nbsp; <span class="log-error-val">❗ Hata:</span> <strong>${err.toFixed(3)}</strong>`,
            "log-dim"
          );

          // Güncellemeden ÖNCESİNDE eski ağırlıkları kopyala
          const oldW = [...w];
          const oldB = b;

          // ✅ DEBUG: err değerini kontrol et
          console.log(`Epoch ${epoch + 1}.${i + 1} - err: ${err}, lr: ${lr}, X[i]: [${X[i]}]`);

          // Güncelleme
          for (let j = 0; j < w.length; j++) w[j] += lr * err * X[i][j];
          b += lr * err;

          // ✅ DEBUG: değişim miktarını göster
          console.log(`Değişim: w=[${oldW.map((v, idx) => (w[idx] - v).toFixed(3)).join(", ")}], b=${(b - oldB).toFixed(3)}`);

          const diffs = w.map((val, idx) => {
            const d = val - oldW[idx];
            const sign = d > 0 ? "▲" : (d < 0 ? "▼" : "→");
            // ✅ FİX: Math.abs(d).toFixed(3) yerine direkt göster
            return `${val.toFixed(3)} <span class="log-diff">${sign} ${Math.abs(d).toFixed(4)}</span>`;
          }).join(", ");

          const bDiff = b - oldB;
          const bSign = bDiff > 0 ? "▲" : (bDiff < 0 ? "▼" : "→");

          log(
            `&nbsp;&nbsp;🔹 Güncellendi: <span class="log-weight">w = [${diffs}]</span>, ` +
            `<span class="log-bias">b = ${b.toFixed(3)} <span class="log-diff">${bSign} ${Math.abs(bDiff).toFixed(4)}</span></span>`,
            "log-success-small"
          );

          sumErr += Math.abs(y[i] - yPred);
          sumSqErr += (y[i] - net) ** 2;
        }

        errors.push(rule === "adaline" ? sumSqErr / X.length : sumErr);
        wHist.push([...w]);

        if (rule === "adaline")
          log(`📉 <span class="log-error-val">Epoch ${epoch + 1} MSE:</span> <strong>${(sumSqErr / X.length).toFixed(4)}</strong>`, "log-info");
        else
          log(`📉 <span class="log-error-val">Epoch ${epoch + 1} Toplam Hata:</span> <strong>${sumErr}</strong>`, "log-info");

        if (sumErr === 0 && rule === "perceptron") {
          solved = true;
          log(`\n✅ Perceptron öğrenme tamamlandı — tüm örnekler doğru sınıflandırıldı.`, "log-success");
          break;
        }
      }

      if (!solved && rule === "perceptron") {
        log(`\n⚠️ Eğitim tamamlandı fakat tüm örnekler doğru sınıflandırılamadı. (XOR gibi lineer ayrılmaz veri olabilir.)`, "log-warning");
      }

      // Özet: başlangıç vs bitiş (ikonlu ve vurgulu)
      log("\n--- EĞİTİM ÖZETİ ---", "log-header");
      log(
        `🔸 <span class="log-start">Başlangıç:</span> w = [${wInitial.map(v => v.toFixed(3)).join(", ")}], b = ${bInitial.toFixed(3)}<br>` +
        `🔚 <span class="log-end">Bitiş:</span> w = [${w.map(v => v.toFixed(3)).join(", ")}], b = ${b.toFixed(3)}`,
        "log-summary"
      );

      drawCharts(errors, wHist, rule);

      // ✅ Eğitilmiş ağırlıkları global değişkenlere kaydet
      trainedW = [...w];
      trainedB = b;

      log("\n✅ <strong>Sistem test için hazır!</strong> Alttaki test alanına veri giriniz.", "log-success");

    } catch (e) {
      log(`❌ <strong>HATA:</strong> ${e.message}`, "log-error");
    }
  }

  document.getElementById("test-button").addEventListener("click", runTest);

  function runTest() {
    if (!trainedW || trainedB === null) {
      alert("❌ Önce sistem eğitmeli! Train butonuna tıklayın.");
      return;
    }

    const testDataText = document.getElementById("test-data").value.trim();
    
    if (!testDataText) {
      alert("❌ Test verisi giriniz!");
      return;
    }

    try {
      const { X: testX, y: testY } = parseDataset(testDataText);

      if (testX.length === 0) {
        log("❌ Geçerli test verisi yok!", "log-error");
        return;
      }

      log("\n════════════════════════════════════════", "log-header");
      log("🧪 TEST MODUna BAŞLANDI", "log-header");
      log(`📊 Toplam Test Örneği: <strong>${testX.length}</strong>`, "log-info");
      log("════════════════════════════════════════\n", "log-header");

      let correctCount = 0;
      const results = [];

      for (let i = 0; i < testX.length; i++) {
        const net = dot(testX[i], trainedW) + trainedB;
        const yPred = net >= 0 ? 1 : 0;
        const isCorrect = yPred === testY[i];

        if (isCorrect) correctCount++;

        const status = isCorrect ? "✅" : "❌";
        const resultText = isCorrect ? "DOĞRU" : "YANLIŞ";
        const resultClass = isCorrect ? "log-success-small" : "log-error";

        log(
          `${status} Test ${i + 1}: Girdi [${testX[i].join(", ")}] → ` +
          `Tahmin: <strong>${yPred}</strong>, ` +
          `Hedef: <strong>${testY[i]}</strong>, ` +
          `Sonuç: <span style="color: ${isCorrect ? 'green' : 'red'}; font-weight: bold;">${resultText}</span>`,
          resultClass
        );

        results.push({ input: testX[i], predicted: yPred, actual: testY[i], correct: isCorrect });
      }

      const accuracy = ((correctCount / testX.length) * 100).toFixed(2);
      const accuracyColor = accuracy >= 75 ? "#28a745" : (accuracy >= 50 ? "#ffc107" : "#dc3545");

      log("\n════════════════════════════════════════", "log-header");
      log(
        `🎯 <strong>TEST SONUCU:</strong><br>` +
        `✔️ Doğru Tahmin: <strong>${correctCount}/${testX.length}</strong><br>` +
        `📈 Başarı Oranı: <strong style="color: ${accuracyColor}; font-size: 18px;">%${accuracy}</strong>`,
        "log-summary"
      );
      log("════════════════════════════════════════", "log-header");

    } catch (e) {
      log(`❌ <strong>TEST HATASI:</strong> ${e.message}`, "log-error");
    }
  }
});
