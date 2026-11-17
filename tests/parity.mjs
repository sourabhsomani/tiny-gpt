// Checks that the JavaScript model gives the same numbers as PyTorch.
// Run after `python src/export.py`:   node tests/parity.mjs
import { readFileSync } from "node:fs";
import { TinyGPT } from "../docs/model.js";

const meta = JSON.parse(readFileSync("docs/model/config.json", "utf8"));
const bin = readFileSync("docs/model/weights.bin");
const buffer = bin.buffer.slice(bin.byteOffset, bin.byteOffset + bin.byteLength);
const fixture = JSON.parse(readFileSync("tests/fixture.json", "utf8"));

const model = new TinyGPT(meta, buffer);
let logits;
for (const id of model.encode(fixture.prompt)) logits = model.step(id);

let worst = 0;
fixture.logits.forEach((expected, i) => {
  worst = Math.max(worst, Math.abs(expected - logits[i]));
});
console.log(`max difference vs PyTorch: ${worst.toExponential(2)}`);
if (!(worst < 1e-3)) {
  console.error("FAIL: JavaScript and PyTorch disagree");
  process.exit(1);
}

// generate past the context window to exercise the cache reset
let text = "";
const t0 = performance.now();
for (let i = 0; i < 300; i++) {
  const next = model.sample(logits);
  text += meta.chars[next];
  logits = model.step(next);
  if (!logits.every(Number.isFinite)) {
    console.error("FAIL: non-finite logits at step", i);
    process.exit(1);
  }
}
const perChar = (performance.now() - t0) / 300;
console.log(`generated 300 chars, ${perChar.toFixed(1)} ms per char`);
console.log("---\n" + fixture.prompt + text + "\n---\nPASS");
