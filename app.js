"use strict";

// Values transcribed from the supplied manuscript, not recomputed by the page.
const results = [
  ["Full volume", "Explicit", 0.593, 0.065, 0.746, 0.209],
  ["Full volume", "Explicit + R", 0.508, 0.008, 0.488, 0.015],
  ["Full volume", "Diffusion", 0.623, 0.006, 0.748, 0.14],
  ["Full volume", "Diffusion + R", 0.625, 0.1, 0.555, 0.08],
  ["Tissue-masked", "Explicit", 0.703, 0.288, 0.455, 0.029],
  ["Tissue-masked", "Explicit + R", 0.601, 0.018, 0.701, 0.287],
  ["Tissue-masked", "Diffusion", 0.539, 0.046, 0.648, 0.076],
  ["Tissue-masked", "Diffusion + R", 0.53, 0.07, 0.53, 0.032],
  ["Point cloud", "Explicit", 0.541, 0.041, 0.467, 0.015],
  ["Point cloud", "Explicit + R", 0.534, 0.044, 0.443, 0.013],
  ["Point cloud", "Diffusion", 0.524, 0.05, 0.534, 0.069],
  ["Point cloud", "Diffusion + R", 0.535, 0.055, 0.51, 0.023],
];
const bestDepth = Math.min(...results.map((row) => row[2]));
const bestDirection = Math.min(...results.map((row) => row[4]));
document.querySelector("#results-table").innerHTML = results
  .map((row, index) => {
    function value(mean, sd, best) {
      const text = `${mean.toFixed(3)} ± ${sd.toFixed(3)}`;
      return mean === best ? `<strong>${text}</strong>` : text;
    }
    return `<tr class="${index % 4 === 0 ? "group-start" : ""}"><td>${index % 4 === 0 ? row[0] : '<span class="sr-only">' + row[0] + "</span>"}</td><td>${row[1]}</td><td>${value(row[2], row[3], bestDepth)}</td><td>${value(row[4], row[5], bestDirection)}</td></tr>`;
  })
  .join("");

document.querySelectorAll("[data-copy]").forEach((button) => {
  button.addEventListener("click", async () => {
    const element = document.getElementById(button.dataset.copy);
    let message = "Copied to clipboard";
    try {
      await navigator.clipboard.writeText(element.innerText);
    } catch {
      const range = document.createRange();
      range.selectNodeContents(element);
      const selection = getSelection();
      selection.removeAllRanges();
      selection.addRange(range);
      message = "Text selected — use your copy shortcut";
    }
    const toast = document.querySelector("#toast");
    toast.textContent = message;
    toast.classList.add("show");
    setTimeout(() => toast.classList.remove("show"), 2200);
  });
});

const demos = {};
async function initializeDemos() {
  try {
    const response = await fetch("assets/demos.json?v=20261006-all", {
      cache: "no-store",
    });
    if (!response.ok) throw new Error("Video index unavailable");
    Object.assign(demos, await response.json());
    selectDemo("depth", false);
  } catch {
    const note = document.querySelector("#recording-note");
    note.hidden = false;
    note.textContent =
      "Video could not load. The recorded experiment poster is shown.";
  }
}
function selectDemo(key, autoplay = true) {
  const demo = demos[key];
  if (!demo) return;
  const video = document.querySelector("#demo-video");
  video.pause();
  video.src = demo.video;
  video.poster = demo.poster;
  video.hidden = false;
  document.querySelector("#demo-poster").hidden = true;
  document.querySelector("#demo-description").textContent = demo.description;
  const note = document.querySelector("#recording-note");
  note.textContent = demo.note || "";
  note.hidden = !demo.note;
  document
    .querySelector("#demo-panel")
    .setAttribute("aria-label", demo.description);
  document
    .querySelectorAll("[data-demo]")
    .forEach((button) =>
      button.setAttribute("aria-selected", button.dataset.demo === key),
    );
  video.load();
  if (autoplay && !matchMedia("(prefers-reduced-motion: reduce)").matches)
    video.play().catch(() => {});
}
document
  .querySelectorAll("[data-demo]")
  .forEach((button) =>
    button.addEventListener("click", () => selectDemo(button.dataset.demo)),
  );
initializeDemos();

const representations = {
  full: {
    title: "FULL VOLUME",
  },
  tissue: {
    title: "TISSUE-MASKED VOLUME",
  },
  points: {
    title: "VOXELIZED POINT CLOUD",
  },
};
document.querySelectorAll("[data-representation]").forEach((button) => {
  button.addEventListener("click", () => {
    const key = button.dataset.representation;
    const representation = representations[key];
    document
      .querySelectorAll("[data-representation]")
      .forEach((tab) => tab.setAttribute("aria-selected", tab === button));
    document.querySelector("#representation-view-title").textContent =
      representation.title;
    document.querySelector("#interactive-representation").hidden = false;
    window.octViewer?.setRepresentation(key);
  });
});
document.querySelectorAll('[role="tablist"]').forEach((list) => {
  const tabs = Array.from(list.querySelectorAll('[role="tab"]'));
  tabs.forEach((button, index) =>
    button.addEventListener("keydown", (event) => {
      if (
        ![
          "ArrowLeft",
          "ArrowRight",
          "ArrowUp",
          "ArrowDown",
          "Home",
          "End",
        ].includes(event.key)
      )
        return;
      event.preventDefault();
      const direction = ["ArrowRight", "ArrowDown"].includes(event.key)
        ? 1
        : -1;
      const next =
        event.key === "Home"
          ? 0
          : event.key === "End"
            ? tabs.length - 1
            : (index + direction + tabs.length) % tabs.length;
      tabs[next].focus();
      tabs[next].click();
    }),
  );
});
