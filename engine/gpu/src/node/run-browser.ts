// Experiment 1 on raw WebGPU inside headless Chromium (Dawn). Same WGSL and dispatch order as the vgpu path.
import { readFile } from "node:fs/promises";
import { createServer, type IncomingMessage, type Server, type ServerResponse } from "node:http";
import { stripTypeScriptTypes } from "node:module";
import { join } from "node:path";
import { chromium, type Browser } from "playwright";
import type { BrowserOutcome } from "../browser-main.ts";
import type { ExperimentSettings } from "../experiment.ts";
import { parseCommon } from "./cli.ts";
import { KERNEL_DIR, printSummary, SOURCE_DIR, writeOutputs } from "./files.ts";
import { assertTargetAdapter } from "./hardware.ts";

const ADAPTERS = ["hardware", "software"] as const;
const PAGE = `<!doctype html><meta charset="utf-8"><link rel="icon" href="data:,"><script type="module" src="/src/browser-main.ts"></script>`;
const SAFE_NAME = /^[\w-]+\.(ts|wgsl|json|bin)$/;

const BASE_FLAGS = ["--enable-unsafe-webgpu", "--enable-webgpu-developer-features"];
const LINUX_VULKAN = ["--enable-features=Vulkan", "--use-angle=vulkan", "--disable-vulkan-surface"];
const SWIFTSHADER = ["--use-vulkan=swiftshader", "--use-webgpu-adapter=swiftshader", "--no-sandbox"];
/** Playwright's Chromium ships D3DCompiler_47 (FXC) but not dxil.dll/dxcompiler.dll, so D3D12 device creation fails with DXC. */
const WINDOWS_FXC = ["--disable-dawn-features=use_dxc"];

function chromiumFlags(adapter: (typeof ADAPTERS)[number]): string[] {
  const vulkan = process.platform === "linux" || adapter === "software" ? LINUX_VULKAN : [];
  const shaderCompiler = process.platform === "win32" && adapter === "hardware" ? WINDOWS_FXC : [];
  return [...BASE_FLAGS, ...vulkan, ...shaderCompiler, ...(adapter === "software" ? SWIFTSHADER : [])];
}

async function resolveFile(url: string, caseDir: string): Promise<{ readonly body: string | Buffer; readonly type: string }> {
  const [, folder, name] = url.split("/");
  if (url === "/") return { body: PAGE, type: "text/html" };
  if (name === undefined || !SAFE_NAME.test(name)) throw new Error(`refused ${url}`);
  if (folder === "src") return { body: stripTypeScriptTypes(await readFile(join(SOURCE_DIR, name), "utf8")), type: "text/javascript" };
  if (folder === "kernels") return { body: await readFile(join(KERNEL_DIR, name), "utf8"), type: "text/plain" };
  if (folder === "case") return { body: await readFile(join(caseDir, name)), type: "application/octet-stream" };
  throw new Error(`refused ${url}`);
}

function handler(caseDir: string): (request: IncomingMessage, response: ServerResponse) => void {
  return (request, response) => {
    resolveFile(request.url ?? "/", caseDir)
      .then(({ body, type }) => response.writeHead(200, { "content-type": type }).end(body))
      .catch((error: unknown) => response.writeHead(404).end(String(error)));
  };
}

function listen(caseDir: string): Promise<Server> {
  const server = createServer(handler(caseDir));
  return new Promise((resolve) => server.listen(0, "127.0.0.1", () => resolve(server)));
}

function portOf(server: Server): number {
  const address = server.address();
  if (address === null || typeof address === "string") throw new Error("server has no TCP port");
  return address.port;
}

async function solveInPage(browser: Browser, port: number, settings: ExperimentSettings): Promise<BrowserOutcome> {
  const page = await browser.newPage();
  page.on("console", (message) => console.error(`[page] ${message.text()}`));
  page.on("pageerror", (error) => console.error(`[page error] ${error.message}`));
  await page.goto(`http://127.0.0.1:${port}/`);
  await page.waitForFunction(() => window.runE1 !== undefined, undefined, { timeout: 30_000 });
  return page.evaluate((input) => {
    const run = window.runE1;
    if (run === undefined) throw new Error("runE1 missing");
    return run(input);
  }, settings);
}

async function main(): Promise<void> {
  const options = parseCommon(process.argv.slice(2), ADAPTERS, "hardware");
  const server = await listen(options.caseDir);
  const browser = await chromium.launch({ channel: "chromium", headless: true, args: chromiumFlags(options.adapter), executablePath: process.env.E1_CHROMIUM, timeout: 60_000 });
  try {
    const outcome = await solveInPage(browser, portOf(server), options.settings);
    assertTargetAdapter(options.adapter, outcome.report.adapter);
    await writeOutputs(options.caseDir, outcome.report, Float32Array.from(outcome.solution));
    printSummary(outcome.report);
  } finally {
    await browser.close();
    server.close();
  }
}

await main();
