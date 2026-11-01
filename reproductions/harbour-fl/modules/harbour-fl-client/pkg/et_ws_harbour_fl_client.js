import { createHarbourFlUi } from "/modules/@edge-toolkit/harbour-fl-engine/harbour_fl_ui.js";

let instanceConfig = null;
let workflow = null;

export function configure(config) {
  if (workflow) throw new Error("harbour FL client is already initialized");
  if (!config || typeof config.clientId !== "string" || typeof config.label !== "string") {
    throw new Error("harbour FL client requires clientId and label instance configuration");
  }
  instanceConfig = config;
}

export default async function init() {
  if (!instanceConfig) throw new Error("harbour FL client instance was not configured");
  workflow = createHarbourFlUi({ role: "client", ...instanceConfig });
  await workflow.init();
}

export async function run() {
  if (!workflow) throw new Error("harbour FL client is not initialized");
  await workflow.run();
}

export function stop() {
  workflow?.stop();
}

export const is_running = () => workflow?.isRunning() ?? false;
