import { TinyGPT } from "./model.js";

const chat = document.getElementById("chat");
const status = document.getElementById("status");
const form = document.getElementById("composer");
const promptBox = document.getElementById("prompt");
const sendBtn = document.getElementById("send");
const temp = document.getElementById("temp");
const len = document.getElementById("len");

let model = null;
let running = false;
let stopRequested = false;

for (const [slider, out] of [[temp, "tempOut"], [len, "lenOut"]]) {
  slider.addEventListener("input", () => {
    document.getElementById(out).textContent = slider.value;
  });
}

function addMessage(kind) {
  const el = document.createElement("div");
  el.className = `msg ${kind}`;
  chat.appendChild(el);
  return el;
}

// let the page repaint between chunks of work
const nextFrame = () => new Promise((resolve) => setTimeout(resolve, 0));

async function generate(promptText) {
  running = true;
  stopRequested = false;
  sendBtn.textContent = "Stop";

  addMessage("user").textContent = promptText;
  const bot = addMessage("bot");
  const echo = document.createElement("span");
  echo.className = "echo";
  const story = document.createElement("span");
  bot.append(echo, story);

  const ids = model.encode(promptText);
  if (ids.length === 0) ids.push(...model.encode("\n"));
  echo.textContent = ids.map((i) => model.chars[i]).join("");

  model.reset();
  let logits;
  for (const id of ids) logits = model.step(id);

  const total = Number(len.value);
  const temperature = Number(temp.value);
  let pending = "";
  for (let i = 0; i < total && !stopRequested; i++) {
    const next = model.sample(logits, temperature);
    pending += model.chars[next];
    logits = model.step(next);
    if (i % 4 === 3) {
      story.textContent += pending;
      pending = "";
      bot.scrollIntoView({ block: "end" });
      await nextFrame();
    }
  }
  story.textContent += pending;

  running = false;
  sendBtn.textContent = "Send";
}

form.addEventListener("submit", (event) => {
  event.preventDefault();
  if (running) {
    stopRequested = true;
    return;
  }
  const text = promptBox.value.trim();
  if (!model || !text) return;
  promptBox.value = "";
  generate(text);
});

promptBox.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    form.requestSubmit();
  }
});

document.getElementById("chips").addEventListener("click", (event) => {
  if (event.target.tagName !== "BUTTON" || running || !model) return;
  generate(event.target.textContent);
});

TinyGPT.load("model")
  .then((loaded) => {
    model = loaded;
    status.remove();
    promptBox.disabled = false;
    sendBtn.disabled = false;
    promptBox.focus();
  })
  .catch((err) => {
    console.error(err);
    status.textContent =
      "Could not load the model. Open this page through a web server (see README), not as a file.";
  });
