// Run: node decoder/test.js   (checks the JS decoder against vectors written by the Python encoder)
const assert = require("node:assert");
const fs = require("node:fs");
const path = require("node:path");
const { decodeUplink } = require("./decoder.js");

const vectors = JSON.parse(fs.readFileSync(path.join(__dirname, "vectors.json"), "utf8"));
let n = 0;
for (const v of vectors) {
  const out = decodeUplink({ bytes: Buffer.from(v.hex, "hex") });
  assert.deepStrictEqual(out.errors, undefined, "unexpected error for " + v.hex);
  for (const [k, expected] of Object.entries(v.decoded)) {
    assert.ok(Math.abs(out.data[k] - expected) < 1e-9 || out.data[k] === expected,
      `${k}: got ${out.data[k]}, want ${expected} (${v.hex})`);
  }
  n++;
}
assert.ok(decodeUplink({ bytes: [1, 2, 3] }).errors.length === 1, "short payload must error");
const bad = Buffer.alloc(14);
assert.ok(decodeUplink({ bytes: bad }).errors[0].includes("version"), "version 0 must error");
console.log(`decoder ok: ${n} vectors`);
