// The same tiny GPT as src/model.py, written in plain JavaScript so it can
// run in the browser with no libraries. Processes one character at a time and
// caches the attention keys/values of earlier characters to stay fast.

function linear(x, w, b, nIn, nOut) {
  const out = new Float32Array(nOut);
  for (let o = 0; o < nOut; o++) {
    let sum = b ? b[o] : 0;
    const row = o * nIn;
    for (let i = 0; i < nIn; i++) sum += w[row + i] * x[i];
    out[o] = sum;
  }
  return out;
}

function layerNorm(x, gain, bias) {
  const n = x.length;
  let mean = 0;
  for (let i = 0; i < n; i++) mean += x[i];
  mean /= n;
  let variance = 0;
  for (let i = 0; i < n; i++) variance += (x[i] - mean) ** 2;
  const std = Math.sqrt(variance / n + 1e-5);
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) out[i] = ((x[i] - mean) / std) * gain[i] + bias[i];
  return out;
}

function gelu(x) {
  return 0.5 * x * (1 + Math.tanh(0.7978845608 * (x + 0.044715 * x * x * x)));
}

export class TinyGPT {
  constructor(meta, buffer) {
    this.cfg = meta.config;
    this.chars = meta.chars;
    this.stoi = new Map(meta.chars.map((ch, i) => [ch, i]));
    const all = new Float32Array(buffer);
    this.w = {};
    for (const [name, t] of Object.entries(meta.tensors)) {
      const size = t.shape.reduce((a, b) => a * b, 1);
      this.w[name] = all.subarray(t.offset, t.offset + size);
    }
    this.reset();
  }

  static async load(dir) {
    const [meta, buffer] = await Promise.all([
      fetch(`${dir}/config.json`).then((r) => r.json()),
      fetch(`${dir}/weights.bin`).then((r) => r.arrayBuffer()),
    ]);
    return new TinyGPT(meta, buffer);
  }

  encode(text) {
    const ids = [];
    for (const ch of text) if (this.stoi.has(ch)) ids.push(this.stoi.get(ch));
    return ids;
  }

  reset() {
    const { n_layer, block_size, n_embd } = this.cfg;
    this.pos = 0;
    this.history = [];
    this.keys = Array.from({ length: n_layer }, () => new Float32Array(block_size * n_embd));
    this.vals = Array.from({ length: n_layer }, () => new Float32Array(block_size * n_embd));
  }

  // Feed one token, get the scores (logits) for the next one.
  step(token) {
    if (this.pos === this.cfg.block_size) {
      // context is full: start again from the most recent half
      const recent = this.history.slice(-this.cfg.block_size / 2);
      this.reset();
      for (const t of recent) this.step(t);
    }
    this.history.push(token);
    return this.forward(token);
  }

  forward(token) {
    const { n_layer, n_head, n_embd: C, vocab_size } = this.cfg;
    const w = this.w;
    const headDim = C / n_head;
    const scale = 1 / Math.sqrt(headDim);
    const pos = this.pos;

    let x = new Float32Array(C);
    for (let i = 0; i < C; i++) {
      x[i] = w["tok_emb.weight"][token * C + i] + w["pos_emb.weight"][pos * C + i];
    }

    for (let l = 0; l < n_layer; l++) {
      const p = `blocks.${l}.`;

      // --- self attention ---
      let h = layerNorm(x, w[p + "ln1.weight"], w[p + "ln1.bias"]);
      const qkv = linear(h, w[p + "attn.qkv.weight"], w[p + "attn.qkv.bias"], C, 3 * C);
      const keys = this.keys[l];
      const vals = this.vals[l];
      keys.set(qkv.subarray(C, 2 * C), pos * C);
      vals.set(qkv.subarray(2 * C, 3 * C), pos * C);

      const mixed = new Float32Array(C);
      const att = new Float32Array(pos + 1);
      for (let hd = 0; hd < n_head; hd++) {
        const off = hd * headDim;
        let max = -Infinity;
        for (let t = 0; t <= pos; t++) {
          let dot = 0;
          for (let i = 0; i < headDim; i++) dot += qkv[off + i] * keys[t * C + off + i];
          att[t] = dot * scale;
          if (att[t] > max) max = att[t];
        }
        let total = 0;
        for (let t = 0; t <= pos; t++) {
          att[t] = Math.exp(att[t] - max);
          total += att[t];
        }
        for (let t = 0; t <= pos; t++) {
          const a = att[t] / total;
          for (let i = 0; i < headDim; i++) mixed[off + i] += a * vals[t * C + off + i];
        }
      }
      const attnOut = linear(mixed, w[p + "attn.proj.weight"], w[p + "attn.proj.bias"], C, C);
      for (let i = 0; i < C; i++) x[i] += attnOut[i];

      // --- feed forward ---
      h = layerNorm(x, w[p + "ln2.weight"], w[p + "ln2.bias"]);
      const hidden = linear(h, w[p + "fc.weight"], w[p + "fc.bias"], C, 4 * C);
      for (let i = 0; i < hidden.length; i++) hidden[i] = gelu(hidden[i]);
      const mlpOut = linear(hidden, w[p + "fc_proj.weight"], w[p + "fc_proj.bias"], 4 * C, C);
      for (let i = 0; i < C; i++) x[i] += mlpOut[i];
    }

    this.pos++;
    x = layerNorm(x, w["ln_f.weight"], w["ln_f.bias"]);
    return linear(x, w["head.weight"], null, C, vocab_size);
  }

  // Pick the next token from the logits.
  sample(logits, temperature = 0.8, topK = 20) {
    const order = Array.from(logits.keys()).sort((a, b) => logits[b] - logits[a]);
    const keep = order.slice(0, Math.min(topK, order.length));
    const max = logits[keep[0]];
    const probs = keep.map((i) => Math.exp((logits[i] - max) / temperature));
    let r = Math.random() * probs.reduce((a, b) => a + b, 0);
    for (let i = 0; i < keep.length; i++) {
      r -= probs[i];
      if (r <= 0) return keep[i];
    }
    return keep[keep.length - 1];
  }
}
