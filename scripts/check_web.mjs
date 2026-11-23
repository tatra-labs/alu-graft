// Node check of the in-browser engine: same answers as Python, abstains where it should.
//   node scripts/check_web.mjs
import { readFileSync } from "node:fs";
import vm from "node:vm";
const dir = new URL("../docs/", import.meta.url);
vm.runInThisContext(readFileSync(new URL("bench.js", dir), "utf8"));
const C = globalThis.ALU.prep(JSON.parse(readFileSync(new URL("circuit.json", dir), "utf8")));
let bad = 0, n = 0;
const rnd = (k) => Math.floor(Math.random() * 10 ** (1 + Math.floor(Math.random() * k)));
const cases = [["999999+1=", "1000000"], ["12-345678=", "-345666"], ["48213 + 97654 =", " 145867"], ["7 - 9 =", " -2"], ["0+0=", "0"]];
for (let i = 0; i < 40; i++) { const a = rnd(6), b = rnd(6), op = i % 2 ? "+" : "-", sp = i % 4 < 2;
  const r = (op === "+" ? a + b : a - b).toString();
  cases.push(sp ? [`${a} ${op} ${b} =`, " " + r] : [`${a}${op}${b}=`, r]); }
for (const [q, want] of cases) {
  const res = C && globalThis.ALU.solve(C, q, 1.0, 3 + n);
  n++;
  if (res.answer !== want || !res.ended) { bad++; console.log("FAIL", JSON.stringify(q), JSON.stringify(res.answer), "want", JSON.stringify(want)); }
}
for (const q of ["1 + 2 + 3 =", "1.5+2=", "1234567+1=", "$12+3=", "12*3="]) {
  const res = globalThis.ALU.solve(C, q, 1.0, 9); n++;
  if (res.steps.length) { bad++; console.log("FIRED on", q, res.answer); }
}
console.log(`${n - bad}/${n} web-engine checks passed`);
process.exit(bad ? 1 : 0);
