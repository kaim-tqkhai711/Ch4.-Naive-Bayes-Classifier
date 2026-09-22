function readJSON(id) {
    const el = document.getElementById(id);
    if (!el) return null;
    try {
        return JSON.parse(el.textContent);
    } catch (e) {
        return null;
    }
}

function renderConfusionMatrix() {
    const canvas = document.getElementById("confusionCanvas");
    const matrix = readJSON("confusion-matrix-data");
    if (!canvas || !matrix || typeof Chart === "undefined") return;

    // matrix = [[TN, FP], [FN, TP]] with rows = actual, cols = predicted
    const cells = [];
    const labels = ["Negative", "Positive"];
    for (let row = 0; row < 2; row++) {
        for (let col = 0; col < 2; col++) {
            cells.push({ x: labels[col], y: labels[row], v: matrix[row][col] });
        }
    }
    const maxV = Math.max(...cells.map((c) => c.v));

    new Chart(canvas.getContext("2d"), {
        type: "matrix",
        data: {
            datasets: [
                {
                    label: "Confusion Matrix",
                    data: cells,
                    backgroundColor(ctx) {
                        const v = ctx.dataset.data[ctx.dataIndex].v;
                        const alpha = maxV ? 0.15 + 0.75 * (v / maxV) : 0.15;
                        return `rgba(26, 115, 232, ${alpha})`;
                    },
                    width: ({ chart }) => (chart.chartArea || {}).width / 2 - 4,
                    height: ({ chart }) => (chart.chartArea || {}).height / 2 - 4,
                },
            ],
        },
        options: {
            plugins: {
                legend: { display: false },
                tooltip: {
                    callbacks: {
                        title() { return ""; },
                        label(ctx) {
                            const d = ctx.dataset.data[ctx.dataIndex];
                            return `Actual ${d.y} / Predicted ${d.x}: ${d.v}`;
                        },
                    },
                },
                datalabels: false,
            },
            scales: {
                x: { type: "category", labels, title: { display: true, text: "Predicted" }, grid: { display: false } },
                y: { type: "category", labels, offset: true, reverse: false, title: { display: true, text: "Actual" }, grid: { display: false } },
            },
        },
    });
}

function renderWordCloud(containerId, words) {
    const container = document.getElementById(containerId);
    if (!container || !words || !words.length) return;

    const counts = words.map((w) => w[1]);
    const maxCount = Math.max(...counts);
    const minCount = Math.min(...counts);
    const minSize = 13, maxSize = 34;

    container.innerHTML = "";
    words.forEach(([word, count]) => {
        const span = document.createElement("span");
        const t = maxCount === minCount ? 1 : (count - minCount) / (maxCount - minCount);
        span.textContent = word;
        span.style.fontSize = (minSize + t * (maxSize - minSize)).toFixed(1) + "px";
        span.title = `${word}: ${count}`;
        container.appendChild(span);
    });
}

document.addEventListener("DOMContentLoaded", () => {
    renderConfusionMatrix();
    const cloud = readJSON("word-cloud-data");
    if (cloud) {
        renderWordCloud("wordCloudPositive", cloud.positive);
        renderWordCloud("wordCloudNegative", cloud.negative);
    }
});
