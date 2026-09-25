// UBT display-formula renderer: LaTeX -> MathJax SVG (+ optional PNG raster).
//
// JSONL protocol on stdin/stdout (one request per line):
//   {"cmd":"probe"}                                        -> {"ok":true,...}
//   {"id":"b1","latex":"x=1","display":true,
//    "tag":"3.5","png_width":600}                          -> {"id":"b1","ok":true,"svg":"...","png_b64":"...","width":..,"height":..}
// Failures return {"ok":false,"error":"..."} and never kill the process.
import fs from "fs";
import readline from "readline";
import { createRequire } from "module";

const require = createRequire(import.meta.url);
const { mathjax } = await import("mathjax-full/js/mathjax.js");
const { TeX } = await import("mathjax-full/js/input/tex.js");
const { SVG } = await import("mathjax-full/js/output/svg.js");
const { liteAdaptor } = await import("mathjax-full/js/adaptors/liteAdaptor.js");
const { RegisterHTMLHandler } = await import("mathjax-full/js/handlers/html.js");
const { AllPackages } = await import("mathjax-full/js/input/tex/AllPackages.js");

let Resvg = null;
try {
  ({ Resvg } = require("@resvg/resvg-js"));
} catch (e) {
  Resvg = null;
}

const adaptor = liteAdaptor();
RegisterHTMLHandler(adaptor);
const tex = new TeX({ packages: AllPackages });
const svg = new SVG({ fontCache: "local" });
const doc = mathjax.document("", { InputJax: tex, OutputJax: svg });

function renderOne(req) {
  let latex = String(req.latex || "").trim().replace(/^&+\s*/, "");
  if (!latex) return { ok: false, error: "empty latex" };
  const tag = req.tag ? String(req.tag).trim() : "";
  const display = req.display !== false;
  const hasEnv = /\\begin\{(aligned|array|gather|align|cases|matrix|pmatrix|bmatrix|vmatrix|Vmatrix|smallmatrix|split)\}/.test(latex);
  let wrapped = latex;
  if (/&|\/\//.test(latex) && !hasEnv) {
    wrapped = `\\begin{aligned}${latex}\\end{aligned}`;
  }
  if (tag && !/\\tag\s*\{/.test(latex)) {
    // amsmath: \tag is a display-level (or equation-level) command — never
    // inside aligned/array, so environment bodies get an equation wrapper.
    wrapped = hasEnv || wrapped.startsWith("\\begin{aligned}")
      ? `\\begin{equation}${wrapped}\\tag{${tag}}\\end{equation}`
      : `${wrapped}\\tag{${tag}}`;
  }
  const node = doc.convert(wrapped, {
    display,
    em: 16,
    ex: 8,
    containerWidth: 80 * 16,
  });
  const html = adaptor.outerHTML(node);
  const errMatch = html.match(/data-mjx-error="([^"]*)"/);
  if (errMatch) return { ok: false, error: errMatch[1] };
  const svgStr = html.slice(html.indexOf("<svg"), html.lastIndexOf("</svg>") + 6);
  const out = { ok: true, svg: svgStr };
  const vb = svgStr.match(/viewBox="([\d.eE+-]+) ([\d.eE+-]+) ([\d.eE+-]+) ([\d.eE+-]+)"/);
  if (vb) {
    out.width = parseFloat(vb[3]);
    out.height = parseFloat(vb[4]);
  }
  if (Resvg && req.png_width) {
    const w = Math.max(32, Math.round(Number(req.png_width)));
    const r = new Resvg(svgStr, { background: "#ffffff", fitTo: { mode: "width", value: w } });
    const png = r.render().asPng();
    out.png_b64 = Buffer.from(png).toString("base64");
  }
  return out;
}

const rl = readline.createInterface({ input: process.stdin });
rl.on("line", (line) => {
  let req;
  try {
    req = JSON.parse(line);
  } catch (e) {
    process.stdout.write(JSON.stringify({ ok: false, error: "bad json" }) + "\n");
    return;
  }
  if (req.cmd === "probe") {
    let pkg = "unknown";
    try {
      pkg = JSON.parse(fs.readFileSync(new URL("./node_modules/mathjax-full/package.json", import.meta.url), "utf8")).version;
    } catch (e) { /* keep unknown */ }
    process.stdout.write(
      JSON.stringify({ ok: true, mathjax: pkg, raster: Boolean(Resvg) }) + "\n"
    );
    return;
  }
  if (req.cmd === "exit") {
    process.stdout.write(JSON.stringify({ ok: true }) + "\n");
    process.exit(0);
  }
  try {
    const res = renderOne(req);
    process.stdout.write(JSON.stringify({ id: req.id, ...res }) + "\n");
  } catch (e) {
    process.stdout.write(JSON.stringify({ id: req.id, ok: false, error: String(e && e.message ? e.message : e) }) + "\n");
  }
});
