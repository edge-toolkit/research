import { createHarbourFlUi } from "/modules/@edge-toolkit/harbour-fl-engine/harbour_fl_ui.js";

const workflow = createHarbourFlUi({ role: "coordinator", clientId: null, label: "coordinator" });

export default workflow.init;
export const run = workflow.run;
export const stop = workflow.stop;
export const is_running = workflow.isRunning;
