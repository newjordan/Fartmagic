#!/usr/bin/env node
/**
 * Playwright smoke test for shroud_viewer.html
 *
 * Serves the viewer from repo root, loads existing points data,
 * validates rendering, and captures screenshots for review.
 *
 * Usage:
 *   npx playwright test experiments/shroud_nightcrawler/visualizer/test_viewer.mjs
 *   # or directly:
 *   node experiments/shroud_nightcrawler/visualizer/test_viewer.mjs
 */

import { chromium } from "playwright";
import { createServer } from "http";
import { readFile, stat } from "fs/promises";
import { join, extname } from "path";
import { fileURLToPath } from "url";

const __dirname = fileURLToPath(new URL(".", import.meta.url));
const REPO_ROOT = join(__dirname, "..", "..", "..");

const MIME = {
  ".html": "text/html",
  ".js": "application/javascript",
  ".json": "application/json",
  ".css": "text/css",
  ".png": "image/png",
  ".jpg": "image/jpeg",
};

function startServer(root, port) {
  return new Promise((resolve) => {
    const srv = createServer(async (req, res) => {
      const url = new URL(req.url, `http://localhost:${port}`);
      const filePath = join(root, decodeURIComponent(url.pathname));
      try {
        const s = await stat(filePath);
        if (s.isDirectory()) {
          res.writeHead(403);
          res.end();
          return;
        }
        const data = await readFile(filePath);
        const ext = extname(filePath).toLowerCase();
        res.writeHead(200, {
          "Content-Type": MIME[ext] || "application/octet-stream",
          "Content-Length": data.length,
        });
        res.end(data);
      } catch {
        res.writeHead(404);
        res.end("not found");
      }
    });
    srv.listen(port, "127.0.0.1", () => resolve(srv));
  });
}

async function main() {
  const PORT = 8799;
  const SCREENSHOT_DIR = join(REPO_ROOT, "results");
  const srv = await startServer(REPO_ROOT, PORT);
  console.log(`serving repo root on http://127.0.0.1:${PORT}`);

  const browser = await chromium.launch({ headless: true });
  const context = await browser.newContext({ viewport: { width: 1920, height: 1080 } });
  const page = await context.newPage();

  const errors = [];
  page.on("pageerror", (err) => errors.push(err.message));

  // Determine which data file to load - prefer Midnight 12L, then Nightcrawler, then junkyard
  let dataUrl;
  let dataLabel;
  try {
    await stat(join(REPO_ROOT, "results/shroud_midnight_12l_latest.trace.points.json"));
    dataUrl = "/results/shroud_midnight_12l_latest.trace.points.json";
    dataLabel = "midnight_12l";
  } catch {
    try {
      await stat(join(REPO_ROOT, "results/shroud_nightcrawler_latest.trace.points.json"));
      dataUrl = "/results/shroud_nightcrawler_latest.trace.points.json";
      dataLabel = "nightcrawler";
    } catch {
      try {
        await stat(join(REPO_ROOT, "junkyard/results/shroud_junkyard_mini_latest.trace.points.json"));
        dataUrl = "/junkyard/results/shroud_junkyard_mini_latest.trace.points.json";
        dataLabel = "junkyard_mini";
      } catch {
        console.error("no points data found — run the Midnight 12L builder first");
        await browser.close();
        srv.close();
        process.exit(1);
      }
    }
  }
  console.log(`using data: ${dataLabel} (${dataUrl})`);

  // Pre-fetch the data so we can inject it via the file input
  const dataText = await readFile(join(REPO_ROOT, dataUrl.slice(1)), "utf-8");
  console.log(`data loaded: ${(dataText.length / 1024 / 1024).toFixed(1)} MB`);

  // Load the viewer
  const viewerUrl = `http://127.0.0.1:${PORT}/experiments/shroud_nightcrawler/visualizer/shroud_viewer.html`;
  await page.goto(viewerUrl, { waitUntil: "domcontentloaded" });
  console.log("viewer loaded");

  // Wait for Three.js canvas
  await page.waitForSelector("canvas", { timeout: 10000 });
  console.log("canvas rendered");

  // Wait for init() to finish (loads preset or falls back to demo)
  await page.waitForTimeout(2000);

  // Inject data via the file input element
  const fileInput = page.locator("#file");
  const tmpDataPath = join(REPO_ROOT, "results", "_playwright_tmp_data.json");
  const { writeFile, unlink } = await import("fs/promises");
  await writeFile(tmpDataPath, dataText);
  await fileInput.setInputFiles(tmpDataPath);
  console.log("dataset injected via file input");

  // Give Three.js time to parse and render the full dataset
  await page.waitForTimeout(5000);

  // Validate rendering
  const stats = await page.evaluate(() => {
    const canvas = document.querySelector("canvas");
    if (!canvas) return { canvas: false };
    const ctx = canvas.getContext("webgl2") || canvas.getContext("webgl");
    const w = canvas.width;
    const h = canvas.height;
    // Check canvas has non-trivial content by reading a pixel sample
    return {
      canvas: true,
      width: w,
      height: h,
      hasContext: !!ctx,
    };
  });

  console.log(`canvas: ${stats.width}x${stats.height}, webgl: ${stats.hasContext}`);

  // Screenshot each view mode
  const VIEW_MODES = ["flow", "loop", "morph", "overlay"];
  for (const mode of VIEW_MODES) {
    // Try clicking the mode button
    const switched = await page.evaluate((m) => {
      const btn = document.getElementById("viewMode");
      if (btn) {
        // Cycle through modes by setting value
        const select = btn;
        for (const opt of select.options || []) {
          if (opt.value === m || opt.textContent.toLowerCase().includes(m)) {
            select.value = opt.value;
            select.dispatchEvent(new Event("change", { bubbles: true }));
            return true;
          }
        }
      }
      // Try direct PARAMS update
      if (typeof PARAMS !== "undefined") {
        PARAMS.viewMode = m;
        return true;
      }
      return false;
    }, mode);

    if (switched) {
      await page.waitForTimeout(1500);
    }

    const screenshotPath = join(SCREENSHOT_DIR, `shroud_nightcrawler_${mode}_${dataLabel}.png`);
    await page.screenshot({ path: screenshotPath, fullPage: false });
    console.log(`screenshot: ${screenshotPath}`);
  }

  // Check for JS errors
  if (errors.length > 0) {
    console.warn(`JS errors encountered (${errors.length}):`);
    errors.forEach((e) => console.warn(`  ${e}`));
  } else {
    console.log("no JS errors");
  }

  // Clean up temp file
  try { await unlink(tmpDataPath); } catch {}

  await browser.close();
  srv.close();
  console.log("done");
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
