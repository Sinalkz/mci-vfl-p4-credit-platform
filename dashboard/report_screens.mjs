// MCI loan dashboard (v2): headless screenshots for the Phase 4 Word report, 1,600 px wide, into
// docs/mci/v2/figures/report/: MCI alone and with partners, the credit tiers, settlement and the audit trail. Each page
// as it opens (run 1 where it has a run selector), whole, in the dashboard's light theme.
//
// Starts the dashboard (python -m streamlit, port 8599) and a headless Microsoft Edge (or the browser given), drives the
// browser over the Chrome DevTools Protocol with Node's built-in WebSocket: opens each page, waits until its title is the
// page's, its charts have loaded and the page has stopped changing, sets the viewport to the page's full height, and
// captures it. Stops both processes at the end. dashboard/report_figures.py then writes captions.md over the folder.
//
// Usage (from the repo root):  node dashboard/report_screens.mjs [--browser <path>] [--port 8599] [--debug-port 9339]
import { spawn, spawnSync } from "node:child_process";
import { existsSync, mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { parseArgs } from "node:util";

const ROOT = resolve(fileURLToPath(import.meta.url), "..", "..");
const OUT = join(ROOT, "docs", "mci", "v2", "figures", "report");
const WIDTH = 1600;
const { values: args } = parseArgs({ options: {
  browser: { type: "string", default: "C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe" },
  port: { type: "string", default: "8599" }, "debug-port": { type: "string", default: "9339" },
} });
// views.PAGES's slugs; the title each page shows; how many charts it draws (st.pyplot images)
const PAGES = [
  { slug: "alone", file: "dashboard-1-mci-alone-vs-partners.png", title: "همراه اول به‌تنهایی و با شرکا", images: 4 },
  { slug: "tiers", file: "dashboard-2-credit-tiers.png", title: "سطوح اعتبار", images: 1 },
  { slug: "settlement", file: "dashboard-3a-settlement.png", title: "تسویه", images: 1 },
  { slug: "audit", file: "dashboard-3b-audit-trail.png", title: "ردپای حسابرسی", images: 0 },
];

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
async function until(what, fn, timeoutMs = 60000, everyMs = 500) {
  const t0 = Date.now();
  for (;;) {
    const v = await fn().catch(() => null);
    if (v) return v;
    if (Date.now() - t0 > timeoutMs) throw new Error(`timed out waiting for ${what}`);
    await sleep(everyMs);
  }
}
function stop(child) {
  if (!child || child.exitCode !== null) return;
  if (process.platform === "win32") spawnSync("taskkill", ["/pid", String(child.pid), "/T", "/F"], { stdio: "ignore" });
  else child.kill("SIGTERM");
}
/** Every browser process on this run's (unique, temporary) profile: the browser Node started may be a launcher that
 * hands over to a browser outside its process tree, and Edge keeps that one running after Browser.close. */
function stopProfile(profile) {
  if (process.platform === "win32") {
    const ps = `Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -and $_.CommandLine.Contains('${profile}') } | `
      + "ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }";
    spawnSync("powershell", ["-NoProfile", "-NonInteractive", "-Command", ps], { stdio: "ignore" });
  } else {
    spawnSync("pkill", ["-f", profile], { stdio: "ignore" });
  }
}

/** A minimal Chrome DevTools Protocol client over one page's WebSocket. */
class Cdp {
  constructor(url) {
    this.ws = new WebSocket(url);
    this.next = 0;
    this.pending = new Map();
    this.ws.addEventListener("message", (e) => {
      const m = JSON.parse(e.data);
      const p = m.id !== undefined && this.pending.get(m.id);
      if (!p) return;
      this.pending.delete(m.id);
      m.error ? p.reject(new Error(`${p.method}: ${JSON.stringify(m.error)}`)) : p.resolve(m.result);
    });
  }
  open() { return new Promise((res, rej) => { this.ws.addEventListener("open", res); this.ws.addEventListener("error", rej); }); }
  send(method, params = {}) {
    const id = ++this.next;
    this.ws.send(JSON.stringify({ id, method, params }));
    return new Promise((resolve, reject) => this.pending.set(id, { resolve, reject, method }));
  }
  async eval(expression) {
    const r = await this.send("Runtime.evaluate", { expression, returnByValue: true, awaitPromise: true });
    if (r.exceptionDetails) throw new Error(r.exceptionDetails.text);
    return r.result.value;
  }
}

// the page's state: its title, its charts, whether Streamlit is still running it, and the full height of its content
const STATE = `(() => {
  const main = document.querySelector('[data-testid="stMain"]') ?? document.body;
  const imgs = [...main.querySelectorAll('img')];
  const boxes = [document.scrollingElement, document.body, ...document.querySelectorAll(
    '[data-testid="stApp"], [data-testid="stAppViewContainer"], [data-testid="stMain"], [data-testid="stSidebar"] > div')];
  return {
    title: document.querySelector('h1')?.innerText ?? '',
    images: imgs.length, loaded: imgs.every((i) => i.complete && i.naturalWidth > 0),
    running: !!document.querySelector('[data-testid="stStatusWidget"] [data-testid="stStatusWidgetRunningIcon"], [data-testid="stSkeleton"], [data-testid="stSpinner"]'),
    fonts: document.fonts.status,
    height: Math.max(...boxes.filter(Boolean).map((b) => b.scrollHeight)),
    nodes: main.getElementsByTagName('*').length,
  };
})()`;

async function settle(cdp, page) {
  let last = null, same = 0;
  return until(`${page.slug} to finish`, async () => {
    const s = await cdp.eval(STATE);
    const ready = s.title.includes(page.title) && s.images >= page.images && s.loaded && !s.running && s.fonts === "loaded";
    const key = JSON.stringify([s.height, s.nodes, s.images]);
    same = ready && key === last ? same + 1 : 0;
    last = key;
    return same >= 3 ? s : null; // unchanged for three polls in a row
  }, 90000, 700);
}

function pngSize(buf) {
  return { width: buf.readUInt32BE(16), height: buf.readUInt32BE(20) };
}

async function main() {
  if (!existsSync(args.browser)) throw new Error(`no browser at ${args.browser} (pass --browser)`);
  mkdirSync(OUT, { recursive: true });
  const python = join(ROOT, ".venv", process.platform === "win32" ? "Scripts/python.exe" : "bin/python");
  const app = spawn(python, ["-m", "streamlit", "run", "dashboard/app.py", "--server.port", args.port, "--server.headless", "true",
    "--theme.base", "light", "--client.toolbarMode", "minimal", "--browser.gatherUsageStats", "false"], { cwd: ROOT, stdio: "ignore" });
  const profile = mkdtempSync(join(tmpdir(), "mci-screens-"));
  const devtools = `http://127.0.0.1:${args["debug-port"]}`;
  let browser = null;
  let browserWs = null;
  const written = [];
  try {
    const base = `http://127.0.0.1:${args.port}`;
    await until("the dashboard", async () => (await fetch(`${base}/_stcore/health`)).ok);
    browser = spawn(args.browser, ["--headless=new", `--remote-debugging-port=${args["debug-port"]}`, `--user-data-dir=${profile}`,
      "--no-first-run", "--no-default-browser-check", "--disable-extensions", "--hide-scrollbars", `--window-size=${WIDTH},1200`,
      "about:blank"], { stdio: "ignore" });
    browserWs = (await until("the browser", async () => (await fetch(`${devtools}/json/version`)).json())).webSocketDebuggerUrl;
    const target = await until("the browser's page", async () => (await (await fetch(`${devtools}/json/list`)).json()).find((t) => t.type === "page"));
    const cdp = new Cdp(target.webSocketDebuggerUrl);
    await cdp.open();
    await cdp.send("Page.enable");
    await cdp.send("Runtime.enable");
    for (const page of PAGES) {
      await cdp.send("Emulation.setDeviceMetricsOverride", { width: WIDTH, height: 1200, deviceScaleFactor: 1, mobile: false });
      await cdp.send("Page.navigate", { url: `${base}/${page.slug}` });
      let s = await settle(cdp, page);
      for (let i = 0; i < 3; i++) { // the full height, until the page fits the viewport
        await cdp.send("Emulation.setDeviceMetricsOverride", { width: WIDTH, height: s.height, deviceScaleFactor: 1, mobile: false });
        const h = s.height;
        s = await settle(cdp, page);
        if (s.height <= h) break;
      }
      const shot = await cdp.send("Page.captureScreenshot", { format: "png", fromSurface: true });
      const buf = Buffer.from(shot.data, "base64");
      const size = pngSize(buf);
      if (size.width !== WIDTH) throw new Error(`${page.file}: ${size.width} px wide, not ${WIDTH}`);
      writeFileSync(join(OUT, page.file), buf);
      written.push({ file: page.file, url: `/${page.slug}`, ...size, charts: s.images });
    }
  } finally {
    // Browser.close only on the browser's own endpoint (a page's ignores it); then whatever still runs on the profile
    if (browserWs) {
      const b = new Cdp(browserWs);
      await b.open().then(() => b.send("Browser.close")).catch(() => {});
      await sleep(1000);
    }
    stop(browser);
    stopProfile(profile);
    stop(app);
    await sleep(1000);
    try {
      rmSync(profile, { recursive: true, force: true, maxRetries: 10, retryDelay: 500 });
    } catch (e) {
      console.error(`warning: the browser's temporary profile ${profile} is left: ${e.code}`);
    }
  }
  console.log(JSON.stringify({ dir: OUT, theme: "light", width: WIDTH, screenshots: written }, null, 1));
}

await main();
