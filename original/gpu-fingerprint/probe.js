// Numeric GPU fingerprinting in the browser.
//
// A shading language is allowed to evaluate most floating-point operations in more than one way: a multiply and
// an add may be fused or not, a sum may be reordered, a subnormal may be flushed, and each function carries only
// an accuracy bound. Two GPUs can therefore return different bits for the same expression and both conform. This
// script puts a fixed set of such expressions to the device, through WebGPU compute shaders and through WebGL 2
// fragment shaders, reads back the raw bits, and compares them with the correctly rounded answer. The sequence of
// results is a signature. It is read from the numerical output of the GPU, not from any name the browser reports,
// so the browser's name-masking defences do not touch it, and it is not a canvas read, so canvas defences do not
// either. This is a faithful port of the et-gpu-probe Rust crate (edge-toolkit/core, libs/gpu-probe).

// detect-browser (MIT) names the browser family and version from the user-agent string, separating Edge, Opera
// and Samsung Internet from Chrome where the UA-CH brands list does not. Brave detection and the precise OS
// version are layered on top of it in gatherEnv, because no user-agent parser can supply either.
import { detect } from "https://cdn.jsdelivr.net/npm/detect-browser@5.3.0/+esm";

const UNRECOGNISED = "not present";

// One compute shader evaluates every probe in a batch and writes each result's raw bits to its own output slot, so
// the whole set costs one shader compilation instead of sixteen -- the dominant cost (see the paper's timing
// section). Each expression is bitcast independently, so a combined shader yields the same bits as sixteen separate
// ones (verified bit-for-bit on an M5 Pro). The WebGL 2 fragment path below is wrapped the same way.
function buildComputeShader(probes) {
  const writes = probes.map((p, i) => `    probeOut[${i}] = bitcast<u32>(${p.wgsl});`).join("\n");
  return `@group(0) @binding(0) var<storage, read>       probeIn  : array<f32, 8>;
@group(0) @binding(1) var<storage, read_write> probeOut : array<u32, ${probes.length}>;

@compute @workgroup_size(1)
fn main() {
    let ulpBase = probeIn[0];
    let negatedSquare = probeIn[1];
    let big = probeIn[2];
    let one = probeIn[3];
    let angle = probeIn[4];
    let three = probeIn[5];
    let tiny = probeIn[6];
    let oneHalf = probeIn[7];

    var sum : f32 = big;
    for (var turn : u32 = 0u; turn < 4u; turn = turn + 1u) {
        sum = sum + one;
    }

${writes}
}
`;
}

const VERTEX_SHADER = `#version 300 es
void main() {
    // A full-screen triangle, so the fragment shader runs once per output pixel -- one pixel per probe.
    vec2 p = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));
    gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}
`;

// One fragment shader evaluates every probe in the batch, picks each by the pixel's x coordinate, and writes its
// bits to that pixel; the batch is rendered to an N-wide framebuffer and read back in one call, so the whole set
// costs one compilation. The fma probe is left out of the WebGL 2 set (it has no glsl form in probes.yaml): GLSL ES
// 3.00 has no fma builtin, and multiply_add already captures whether the compiler contracts a*b+c, so nothing is
// lost. With fma gone the combined shader compiles in one go, which is what makes this path fast.
function buildFragmentShader(probes) {
  const branches = probes.map((p, i) => `    if (idx == ${i}) { value = ${p.glsl}; }`).join("\n");
  return `#version 300 es
precision highp float;
precision highp int;
uniform float probeIn[8];
out vec4 probeOut;

void main() {
    float ulpBase = probeIn[0];
    float negatedSquare = probeIn[1];
    float big = probeIn[2];
    float one = probeIn[3];
    float angle = probeIn[4];
    float three = probeIn[5];
    float tiny = probeIn[6];
    float oneHalf = probeIn[7];

    float sum = big;
    for (int turn = 0; turn < 4; turn++) {
        sum = sum + one;
    }

    int idx = int(gl_FragCoord.x);
    float value = 0.0;
${branches}
    uint bits = floatBitsToUint(value);
    probeOut = vec4(
        float(bits & 255u),
        float((bits >> 8) & 255u),
        float((bits >> 16) & 255u),
        float(bits >> 24)
    ) / 255.0;
}
`;
}

// --- small helpers --------------------------------------------------------------------------------------------

const f32 = Math.fround;

/** The bits of an f32 value as an unsigned 32-bit integer. */
function bitsOf(value) {
  return new Uint32Array(new Float32Array([value]).buffer)[0] >>> 0;
}

/** An eight-character lowercase hex word. */
function hex8(bits) {
  return (bits >>> 0).toString(16).padStart(8, "0");
}

/** The eight probe inputs as an f32 array, reconstructed exactly from their bit patterns. */
function inputsFrom(data) {
  const words = Uint32Array.from(data.inputs.bits.map((h) => parseInt(h, 16)));
  return new Float32Array(words.buffer);
}

// --- data files -----------------------------------------------------------------------------------------------

// probes.yaml and registry.yaml are parsed with js-yaml (loaded from a CDN in index.html). New signatures are
// submitted to server.py as a compact JSON line (easy to copy and paste); the server appends each as a YAML
// block, so the registry stays append-only and a commit never rewrites an existing row.

// --- reference ------------------------------------------------------------------------------------------------

// The correctly rounded answer to each probe, computed here so the port can be checked against the data file.
// The exact probes are evaluated one f32 operation at a time; fma is the full-precision product rounded once;
// the functions are evaluated in f64 and rounded once to f32.
function computeReference(inputs) {
  const [ulpBase, negatedSquare, big, one, angle, three, tiny, oneHalf] = inputs;

  let sum = big;
  for (let turn = 0; turn < 4; turn++) sum = f32(sum + one);

  let dot = 0;
  for (const product of [f32(big * one), f32(one * one), f32(one * one), f32(-big * one)]) dot = f32(dot + product);

  return [
    bitsOf(f32(f32(ulpBase * ulpBase) + negatedSquare)),
    bitsOf(f32(ulpBase * ulpBase + negatedSquare)),
    bitsOf(f32(sum - big)),
    bitsOf(f32(one / three)),
    bitsOf(f32(tiny * oneHalf)),
    bitsOf(f32(Math.sqrt(three))),
    bitsOf(f32(1 / Math.sqrt(three))),
    bitsOf(f32(Math.sin(angle))),
    bitsOf(f32(Math.cos(angle))),
    bitsOf(f32(Math.tan(angle))),
    bitsOf(f32(Math.exp(angle))),
    bitsOf(f32(Math.log(three))),
    bitsOf(f32(Math.pow(three, angle))),
    bitsOf(f32(Math.atan2(angle, three))),
    bitsOf(f32(Math.tanh(angle))),
    bitsOf(dot),
  ];
}

// --- running the probes ---------------------------------------------------------------------------------------

// An outcome is { bits } for a result, or { refused } with a reason. This mirrors the Rust Answer enum.

// Each interface is run in two modes: split, with every probe compiled in a shader of its own, and combined, with
// all of them compiled in one shader. A compiler may optimise an expression differently when it sits beside others,
// so the two modes can return different bits for the same expression. Each mode's signature is therefore kept under
// its own registry interface name, such as webgl2-split and webgl2-combined, and the page says when they differ.
const MODES = [["split", "one shader per expression"], ["combined", "all expressions in one shader"]];

// Run a list of probes through a batch function that compiles one shader for the whole list and returns an array of
// result bits. With split set, each probe gets a batch, and so a shader, to itself. Otherwise the batch throws an
// error with .compile === true when its shader will not compile; we then divide the list and recurse, isolating a
// rejected probe (refused) while keeping the rest in as few compilations as possible.
async function runBatched(probes, batch, split) {
  if (split && probes.length > 1) {
    const parts = [];
    for (const probe of probes) parts.push(...(await runBatched([probe], batch, true)));
    return parts;
  }
  try {
    return (await batch(probes)).map((b) => ({ bits: b }));
  } catch (e) {
    if (!e.compile) throw e;
    if (probes.length === 1) return [{ refused: String(e.message || e) }];
    const mid = probes.length >> 1;
    const left = await runBatched(probes.slice(0, mid), batch);
    const right = await runBatched(probes.slice(mid), batch);
    return left.concat(right);
  }
}

// An expression no probe uses, compiled and run before either mode is timed. Whatever the first shader in a
// context costs once -- loading the compiler, creating the first pipeline -- is paid here, so it is not charged to
// whichever mode happens to run first.
const WARM_UP = { name: "warm_up", wgsl: "one + three", glsl: "one + three" };

// Run the probes in both modes and time each: { combined, split }, each { outcomes, elapsedMs }. widen, when given,
// maps a mode's outcomes back onto the full probe list.
async function runBothModes(probes, batch, widen) {
  try {
    await batch([WARM_UP]);
  } catch { /* the warm-up's result is not used, and a failure here shows up again in the timed runs */ }
  const timed = async (split) => {
    const t0 = performance.now();
    const outcomes = await runBatched(probes, batch, split);
    const elapsedMs = Math.round((performance.now() - t0) * 10) / 10;
    return { outcomes: widen ? widen(outcomes) : outcomes, elapsedMs };
  };
  const combined = await timed(false);
  return { combined, split: await timed(true) };
}

async function runWebGPU(data, inputs) {
  if (!navigator.gpu) {
    // navigator.gpu is absent either because this is not a secure context, or because the browser does not expose
    // WebGPU at all (Firefox without the flag, or Tor Browser, which ships it disabled). isSecureContext tells them
    // apart: on an HTTPS page it is true, so the cause is the browser, not the origin.
    const why = window.isSecureContext
      ? "the browser does not expose it"
      : "this is not a secure context; WebGPU needs HTTPS or localhost";
    return { error: `WebGPU is unavailable (navigator.gpu is absent; ${why})` };
  }
  let adapter;
  try {
    adapter = await navigator.gpu.requestAdapter();
  } catch (e) {
    return { error: `requestAdapter failed: ${e.message || e}` };
  }
  if (!adapter) return { error: "no WebGPU adapter was returned" };
  const device = await adapter.requestDevice();

  let info = {};
  try {
    info = adapter.info ? adapter.info : (adapter.requestAdapterInfo ? await adapter.requestAdapterInfo() : {});
  } catch {
    info = {};
  }

  const inBuf = device.createBuffer({ size: 32, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(inBuf, 0, inputs);

  // Compile and run one combined shader for a batch of probes; throws { compile: true } if it does not compile.
  const batch = async (probes) => {
    const n = probes.length;
    const module = device.createShaderModule({ code: buildComputeShader(probes) });
    const details = await module.getCompilationInfo();
    const err = details.messages.find((m) => m.type === "error");
    if (err) {
      const rejected = new Error(`shader rejected: ${err.message}`);
      rejected.compile = true;
      throw rejected;
    }
    const outBuf = device.createBuffer({ size: n * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const readBuf = device.createBuffer({ size: n * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const pipeline = device.createComputePipeline({ layout: "auto", compute: { module, entryPoint: "main" } });
    const bind = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: inBuf } },
        { binding: 1, resource: { buffer: outBuf } },
      ],
    });
    const enc = device.createCommandEncoder();
    const pass = enc.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bind);
    pass.dispatchWorkgroups(1);
    pass.end();
    enc.copyBufferToBuffer(outBuf, 0, readBuf, 0, n * 4);
    device.queue.submit([enc.finish()]);
    await readBuf.mapAsync(GPUMapMode.READ);
    const words = [...new Uint32Array(readBuf.getMappedRange().slice(0))];
    readBuf.unmap();
    return words.map((b) => b >>> 0);
  };

  const modes = await runBothModes(data.probes, batch);
  const renderer = [...new Set([info.vendor, info.architecture, info.device, info.description].filter(Boolean))].join(
    " ",
  );
  return { modes, renderer: renderer || "(adapter info blank or withheld)" };
}

function compileShader(gl, type, source) {
  const shader = gl.createShader(type);
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    const log = gl.getShaderInfoLog(shader);
    gl.deleteShader(shader);
    const rejected = new Error(`shader rejected: ${log.trim()}`);
    rejected.compile = true;
    throw rejected;
  }
  return shader;
}

async function runWebGL2(data, inputs) {
  const debug = [];
  try {
    const canvas = document.createElement("canvas");
    canvas.width = 1;
    canvas.height = 1;
    let gl = canvas.getContext("webgl2", { antialias: false, premultipliedAlpha: false, preserveDrawingBuffer: true });
    debug.push(`getContext("webgl2", options) -> ${gl ? "ok" : "null"}`);
    if (!gl) {
      gl = canvas.getContext("webgl2");
      debug.push(`getContext("webgl2") with no options -> ${gl ? "ok" : "null"}`);
    }
    if (!gl) {
      const probe = canvas.getContext("webgl");
      debug.push(`fallback getContext("webgl") -> ${probe ? "ok (WebGL 1 only; WebGL 2 not available)" : "null"}`);
      // No WebGL 1 either means the browser blocks WebGL entirely (Tor Browser does, at its default security level);
      // WebGL 1 present but WebGL 2 absent is an older or restricted driver.
      const why = probe
        ? "only WebGL 1 is available, not WebGL 2"
        : "no WebGL context at all, so WebGL is turned off";
      return { error: `WebGL 2 is unavailable (getContext returned null; ${why})`, debug };
    }
    gl.disable(gl.DITHER);

    // The vertex shader uses no attributes; some drivers (Firefox on macOS) warn and fall back to slow emulation
    // when vertex attribute 0 has no array enabled, so bind a dummy one to keep the fast path.
    const dummy = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, dummy);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([0, 0]), gl.STATIC_DRAW);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);

    const dbg = gl.getExtension("WEBGL_debug_renderer_info");
    debug.push(`WEBGL_debug_renderer_info -> ${dbg ? "present" : "absent (renderer masked)"}`);
    const renderer = dbg ? gl.getParameter(dbg.UNMASKED_RENDERER_WEBGL) : gl.getParameter(gl.RENDERER);
    const vendor = dbg ? gl.getParameter(dbg.UNMASKED_VENDOR_WEBGL) : gl.getParameter(gl.VENDOR);
    debug.push(`GLSL version: ${gl.getParameter(gl.SHADING_LANGUAGE_VERSION)}`);

    let vertex;
    try {
      vertex = compileShader(gl, gl.VERTEX_SHADER, VERTEX_SHADER);
      debug.push("vertex shader: compiled");
    } catch (e) {
      debug.push(`vertex shader FAILED: ${e.message || e}`);
      return { error: "vertex shader failed to compile", debug, renderer: `${vendor} / ${renderer}` };
    }

    // Compile and run one combined fragment shader for a batch of probes, rendering to an N-wide framebuffer and
    // reading the N pixels back at once. Throws { compile: true } if the shader will not compile or link, so
    // runBatched can bisect and isolate an unexpectedly unsupported probe; with fma excluded (below) the usual case
    // compiles in a single shot.
    const batch = (probes) => {
      const n = probes.length;
      canvas.width = n;
      canvas.height = 1;
      const fragment = compileShader(gl, gl.FRAGMENT_SHADER, buildFragmentShader(probes));
      const program = gl.createProgram();
      gl.attachShader(program, vertex);
      gl.attachShader(program, fragment);
      gl.linkProgram(program);
      if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
        const log = gl.getProgramInfoLog(program).trim();
        gl.deleteShader(fragment);
        gl.deleteProgram(program);
        const rejected = new Error(`link failed: ${log}`);
        rejected.compile = true;
        throw rejected;
      }
      gl.useProgram(program);
      gl.uniform1fv(gl.getUniformLocation(program, "probeIn"), inputs);
      gl.viewport(0, 0, n, 1);
      gl.clearColor(0, 0, 0, 0);
      gl.clear(gl.COLOR_BUFFER_BIT);
      gl.drawArrays(gl.TRIANGLES, 0, 3);
      const glErr = gl.getError();
      if (glErr !== gl.NO_ERROR) {
        gl.deleteShader(fragment);
        gl.deleteProgram(program);
        throw new Error(`gl error 0x${glErr.toString(16)} after draw`);
      }
      const px = new Uint8Array(n * 4);
      gl.readPixels(0, 0, n, 1, gl.RGBA, gl.UNSIGNED_BYTE, px);
      gl.deleteShader(fragment);
      gl.deleteProgram(program);
      const bits = [];
      for (let i = 0; i < n; i++) {
        bits.push((px[i * 4] | (px[i * 4 + 1] << 8) | (px[i * 4 + 2] << 16) | (px[i * 4 + 3] << 24)) >>> 0);
      }
      return bits;
    };

    // Only the probes GLSL ES 3.00 can express go through the shader; a probe with no glsl form (fma) is refused
    // without a compile attempt, so the registry still carries all sixteen keys with fma as ????????.
    const glProbes = data.probes.filter((p) => p.glsl);
    const widen = (answered) => {
      const byName = new Map(glProbes.map((p, i) => [p.name, answered[i]]));
      return data.probes.map((p) => p.glsl ? byName.get(p.name) : { refused: "no GLSL ES 3.00 builtin (WebGPU only)" });
    };
    const modes = await runBothModes(glProbes, batch, widen);
    for (const [mode] of MODES) {
      const { outcomes, elapsedMs } = modes[mode];
      outcomes.forEach((o, i) => {
        if (o.refused) debug.push(`${mode}: probe ${data.probes[i].name}: ${o.refused}`);
      });
      const count = outcomes.filter((o) => "bits" in o).length;
      debug.push(`${mode}: answered ${count} of ${outcomes.length} in ${elapsedMs} ms`);
    }
    return { modes, renderer: `${vendor} / ${renderer}`, debug };
  } catch (e) {
    debug.push(`fatal: ${e.message || e}`);
    return { error: `WebGL 2 run threw: ${e.message || e}`, debug };
  }
}

// --- WebGL 1 --------------------------------------------------------------------------------------------------

// WebGL 1 is GLSL ES 1.00, which the WebGL 2 path's two tricks do not have: there is no floatBitsToUint to pack a
// result into four bytes, and no gl_VertexID to make a triangle without a vertex buffer. The triangle is drawn from
// an attribute, and the result is read back one of two ways (see the read-back path choice below): preferably from a
// 32-bit float colour buffer, with no bit work in the shader; or, where that is refused or unavailable, by packing
// the bits into four bytes in the shader and reading them as RGBA8, which needs no extension. It needs highp float
// in the fragment shader, which WebGL 1 does not guarantee; where that is missing the interface is reported
// unavailable. tanh has no GLSL ES 1.00 builtin, so that one probe is refused here, as fma is on every WebGL path.
//
// This interface exists for browsers that offer WebGL 1 but not WebGL 2 or WebGPU. The Tor Browser at its default
// level is the case of interest: our runs found it still offers WebGL 1 while withholding the other two, and it
// allows the float texture and the render but refuses the float read-back, so the RGBA8 path is what tests whether a
// numeric result can be got out of it at all. A signature that differs from one visit to the next (see the visit
// history) is the read-back being randomised instead; a stable one is a numeric path its defence leaves open.
const VERTEX_SHADER_V1 = `attribute vec2 aPos;
void main() {
    gl_Position = vec4(aPos, 0.0, 1.0);
}
`;

// The exact IEEE-754 single-precision bits of a value, packed into four bytes with the red channel least
// significant, so a UNSIGNED_BYTE read-back recovers the same 32 bits the float read-back would. This is used only
// where a float read-back is refused (the Tor Browser) or unavailable, because it does its own arithmetic -- but
// only exact operations (powers of two, floor, mod on integer-valued floats), so for a value the GPU can represent
// it is a lossless re-encoding of the bits, which runWebGL1 verifies against the float path where both are allowed.
const PACK_FLOAT_BITS = `
vec4 packFloatBits(float value) {
    if (value == 0.0) {
        return vec4(0.0);
    }
    float sign = 0.0;
    float v = value;
    if (v < 0.0) {
        sign = 1.0;
        v = -v;
    }
    float exponent = floor(log2(v));
    float biased;
    float mantissaBits;
    if (exponent < -126.0) {
        biased = 0.0;
        mantissaBits = floor(v / exp2(-126.0) * exp2(23.0) + 0.5);
    } else {
        float m = v / exp2(exponent);
        if (m < 1.0) {
            exponent -= 1.0;
            m = v / exp2(exponent);
        }
        if (m >= 2.0) {
            exponent += 1.0;
            m = v / exp2(exponent);
        }
        biased = exponent + 127.0;
        mantissaBits = floor((m - 1.0) * exp2(23.0) + 0.5);
        if (mantissaBits >= exp2(23.0)) {
            mantissaBits = 0.0;
            biased += 1.0;
        }
    }
    float byte3 = sign * 128.0 + floor(biased / 2.0);
    float byte2 = mod(biased, 2.0) * 128.0 + floor(mantissaBits / exp2(16.0));
    float rem = mod(mantissaBits, exp2(16.0));
    float byte1 = floor(rem / exp2(8.0));
    float byte0 = mod(rem, exp2(8.0));
    return vec4(byte0, byte1, byte2, byte3) / 255.0;
}
`;

function buildFragmentShaderV1(probes, pack) {
  const branches = probes.map((p, i) => `    if (idx == ${i}) { value = ${p.glsl}; }`).join("\n");
  const output = pack ? "gl_FragColor = packFloatBits(value);" : "gl_FragColor = vec4(value, 0.0, 0.0, 0.0);";
  return `precision highp float;
precision mediump int;
uniform float probeIn[8];
${pack ? PACK_FLOAT_BITS : ""}
void main() {
    float ulpBase = probeIn[0];
    float negatedSquare = probeIn[1];
    float big = probeIn[2];
    float one = probeIn[3];
    float angle = probeIn[4];
    float three = probeIn[5];
    float tiny = probeIn[6];
    float oneHalf = probeIn[7];

    float sum = big;
    for (int turn = 0; turn < 4; turn++) {
        sum = sum + one;
    }

    int idx = int(gl_FragCoord.x);
    float value = 0.0;
${branches}
    ${output}
}
`;
}

async function runWebGL1(data, inputs) {
  const debug = [];
  // The failing call's name, so that when a browser refuses one (the Tor Browser throws "Operation is not supported"
  // from somewhere on this path) the reported reason says which operation it was, not just that something threw. It
  // is updated before each call that can throw, and read in the catch; the GL error after the float calls is checked
  // too, since a driver can reject a float texture or a float read-back by setting an error rather than throwing.
  let at = "getContext";
  let named = "";
  const glName = (gl, code) => {
    for (
      const key of [
        "NO_ERROR",
        "INVALID_ENUM",
        "INVALID_VALUE",
        "INVALID_OPERATION",
        "INVALID_FRAMEBUFFER_OPERATION",
        "OUT_OF_MEMORY",
        "CONTEXT_LOST_WEBGL",
      ]
    ) {
      if (gl[key] === code) return key;
    }
    return `0x${code.toString(16)}`;
  };
  try {
    const canvas = document.createElement("canvas");
    canvas.width = 1;
    canvas.height = 1;
    const opts = { antialias: false, premultipliedAlpha: false, preserveDrawingBuffer: true };
    const gl = canvas.getContext("webgl", opts) || canvas.getContext("experimental-webgl", opts);
    debug.push(`getContext("webgl") -> ${gl ? "ok" : "null"}`);
    if (!gl) return { error: "WebGL 1 is unavailable (getContext returned null, so WebGL is turned off)", debug };
    gl.disable(gl.DITHER);
    gl.disable(gl.BLEND);

    at = "getExtension(WEBGL_debug_renderer_info) / getParameter(renderer)";
    const dbg = gl.getExtension("WEBGL_debug_renderer_info");
    const renderer = dbg ? gl.getParameter(dbg.UNMASKED_RENDERER_WEBGL) : gl.getParameter(gl.RENDERER);
    const vendor = dbg ? gl.getParameter(dbg.UNMASKED_VENDOR_WEBGL) : gl.getParameter(gl.VENDOR);
    named = `${vendor} / ${renderer}`;
    debug.push(`WEBGL_debug_renderer_info -> ${dbg ? "present" : "absent (renderer masked)"}`);
    debug.push(`GLSL version: ${gl.getParameter(gl.SHADING_LANGUAGE_VERSION)}`);

    // A 32-bit result needs highp float in the fragment shader; WebGL 1 does not guarantee it, and in mediump the
    // bits would be meaningless. Stop and say so rather than record a half-precision signature as if it were one.
    at = "getShaderPrecisionFormat(FRAGMENT_SHADER, HIGH_FLOAT)";
    const prec = gl.getShaderPrecisionFormat(gl.FRAGMENT_SHADER, gl.HIGH_FLOAT);
    debug.push(
      `fragment highp float: ${prec ? `precision ${prec.precision}, rangeMax ${prec.rangeMax}` : "unsupported"}`,
    );
    if (!prec || prec.precision < 23) {
      return {
        error: "WebGL 1 has no highp float in the fragment shader, so a 32-bit result cannot be computed",
        debug,
        renderer: named,
      };
    }

    at = "compile vertex shader";
    let vertex;
    try {
      vertex = compileShader(gl, gl.VERTEX_SHADER, VERTEX_SHADER_V1);
      debug.push("vertex shader: compiled");
    } catch (e) {
      debug.push(`vertex shader FAILED: ${e.message || e}`);
      return { error: "vertex shader failed to compile", debug, renderer: named };
    }
    at = "create vertex buffer";
    const verts = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, verts);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 3, -1, -1, 3]), gl.STATIC_DRAW);
    const tex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    const fbo = gl.createFramebuffer();

    // The result is preferably read back as a float, with no bit work in the shader, so the value is exactly what the
    // GPU computed. That needs OES_texture_float for the texture, a complete float framebuffer, and a float read-back
    // -- and the Tor Browser allows the texture and the render but refuses the read with "Operation is not supported".
    // Where any of the three is missing, fall back to packing the 32 bits into four bytes in the shader and reading
    // them as RGBA8, which every WebGL 1 supports and needs no extension. The two read paths give the same bits (see
    // packFloatBits), so the signature is the same either way; which path ran is recorded in the debug log.
    at = "choose the read-back path";
    const floatTex = gl.getExtension("OES_texture_float");
    const floatBuf = gl.getExtension("WEBGL_color_buffer_float");
    debug.push(
      `OES_texture_float -> ${floatTex ? "present" : "absent"}; WEBGL_color_buffer_float -> ${
        floatBuf ? "present" : "absent"
      }`,
    );
    let useFloat = false;
    if (floatTex) {
      gl.bindTexture(gl.TEXTURE_2D, tex);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 1, 1, 0, gl.RGBA, gl.FLOAT, null);
      const texErr = gl.getError();
      gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0);
      const complete = gl.checkFramebufferStatus(gl.FRAMEBUFFER) === gl.FRAMEBUFFER_COMPLETE;
      let readErr = "not tried";
      if (texErr === gl.NO_ERROR && complete) {
        try {
          gl.readPixels(0, 0, 1, 1, gl.RGBA, gl.FLOAT, new Float32Array(4));
          readErr = glName(gl, gl.getError());
        } catch (e) {
          readErr = `threw: ${e.message || e}`;
        }
      }
      useFloat = texErr === gl.NO_ERROR && complete && readErr === "NO_ERROR";
      debug.push(
        `float read-back: texImage ${glName(gl, texErr)}, framebuffer ${complete ? "complete" : "incomplete"}, ` +
          `read ${readErr}`,
      );
    }
    const readType = useFloat ? gl.FLOAT : gl.UNSIGNED_BYTE;
    debug.push(`read-back path: ${useFloat ? "float texture (exact)" : "RGBA8 packed bits (no float extension)"}`);

    // Compile and run one combined fragment shader for a batch of probes, rendering to an n-wide texture and reading
    // the n pixels back. Throws { compile: true } if the shader will not compile or link, so runBatched can bisect
    // and isolate a probe GLSL ES 1.00 cannot express (tanh), keeping the rest in one shader.
    const batch = (probes) => {
      const n = probes.length;
      at = "compile/link fragment shader";
      const fragment = compileShader(gl, gl.FRAGMENT_SHADER, buildFragmentShaderV1(probes, !useFloat));
      const program = gl.createProgram();
      gl.attachShader(program, vertex);
      gl.attachShader(program, fragment);
      gl.linkProgram(program);
      if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
        const log = gl.getProgramInfoLog(program).trim();
        gl.deleteShader(fragment);
        gl.deleteProgram(program);
        const rejected = new Error(`link failed: ${log}`);
        rejected.compile = true;
        throw rejected;
      }
      at = "set up the render target";
      gl.bindTexture(gl.TEXTURE_2D, tex);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, n, 1, 0, gl.RGBA, readType, null);
      gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0);
      gl.bindBuffer(gl.ARRAY_BUFFER, verts);
      const loc = gl.getAttribLocation(program, "aPos");
      gl.enableVertexAttribArray(loc);
      gl.vertexAttribPointer(loc, 2, gl.FLOAT, false, 0, 0);
      gl.useProgram(program);
      gl.uniform1fv(gl.getUniformLocation(program, "probeIn"), inputs);
      gl.viewport(0, 0, n, 1);
      gl.clearColor(0, 0, 0, 0);
      gl.clear(gl.COLOR_BUFFER_BIT);
      at = "drawArrays";
      gl.drawArrays(gl.TRIANGLES, 0, 3);
      at = useFloat ? "readPixels(RGBA, FLOAT)" : "readPixels(RGBA, UNSIGNED_BYTE)";
      const out = useFloat ? new Float32Array(n * 4) : new Uint8Array(n * 4);
      gl.readPixels(0, 0, n, 1, gl.RGBA, readType, out);
      const glErr = gl.getError();
      gl.deleteShader(fragment);
      gl.deleteProgram(program);
      if (glErr !== gl.NO_ERROR) {
        throw new Error(`read-back gave ${glName(gl, glErr)}; the result could not be read`);
      }
      const bits = [];
      if (useFloat) {
        const words = new Uint32Array(out.buffer);
        for (let i = 0; i < n; i++) bits.push(words[i * 4] >>> 0);
      } else {
        for (let i = 0; i < n; i++) {
          bits.push((out[i * 4] | (out[i * 4 + 1] << 8) | (out[i * 4 + 2] << 16) | (out[i * 4 + 3] << 24)) >>> 0);
        }
      }
      return bits;
    };

    // Only the probes with a GLSL form go through the shader; fma has none and is refused without a compile attempt,
    // so the registry still carries all sixteen keys. tanh has a GLSL form but no GLSL ES 1.00 builtin, so it fails
    // to compile and runBatched records it refused, leaving fourteen answers on this interface.
    const glProbes = data.probes.filter((p) => p.glsl);
    const widen = (answered) => {
      const byName = new Map(glProbes.map((p, i) => [p.name, answered[i]]));
      return data.probes.map((p) => p.glsl ? byName.get(p.name) : { refused: "no GLSL builtin (WebGPU only)" });
    };
    const modes = await runBothModes(glProbes, batch, widen);
    for (const [mode] of MODES) {
      const { outcomes, elapsedMs } = modes[mode];
      outcomes.forEach((o, i) => {
        if (o.refused) debug.push(`${mode}: probe ${data.probes[i].name}: ${o.refused}`);
      });
      const count = outcomes.filter((o) => "bits" in o).length;
      debug.push(`${mode}: answered ${count} of ${outcomes.length} in ${elapsedMs} ms`);
    }
    return { modes, renderer: named, debug };
  } catch (e) {
    debug.push(`fatal at ${at}: ${e.message || e}`);
    return { error: `WebGL 1 failed at ${at}: ${e.message || e}`, debug, renderer: named || undefined };
  }
}

// --- canvas ---------------------------------------------------------------------------------------------------

// A classic 2-D canvas fingerprint, added so its signature and its timing sit beside the numeric WebGPU and
// WebGL 2 ones for comparison. It is a different signal -- it reads how the browser and OS rasterise text and
// shapes, not how the GPU rounds arithmetic -- but it is the fingerprint the numeric probe is meant to improve on,
// so measuring both on one device is the point. A hard read failure is reported as blocked; a browser that injects
// per-read noise (Brave, Tor, Firefox's resistFingerprinting) is caught by rendering twice and comparing.
//
// Two canvas fingerprints are taken, because a defence can apply to one way of making and reading a canvas and not
// to another: this page's own scene, read with getImageData from a canvas the page made, and the BrowserLeaks
// canvas test, read as a PNG from a canvas made inside a sandboxed frame. A third check draws known values and
// reads them back through every combination of those choices, to show which of them a browser alters.

// A 128-bit hash of the pixels as four lowercase hex words, enough entropy to tell devices apart and in the same
// hex-word shape the registry stores. Four independent multiply-xor lanes, folded together periodically.
function hashPixels(bytes) {
  let h0 = 0x811c9dc5, h1 = 0x1b873593, h2 = 0x9e3779b1, h3 = 0x85ebca77;
  for (let i = 0; i < bytes.length; i++) {
    const b = bytes[i];
    h0 = Math.imul(h0 ^ b, 16777619) >>> 0;
    h1 = Math.imul(h1 ^ b, 2246822519) >>> 0;
    h2 = Math.imul(h2 ^ b, 3266489917) >>> 0;
    h3 = Math.imul(h3 ^ b, 668265263) >>> 0;
    if ((i & 63) === 63) {
      h2 = (h2 ^ (h0 >>> 13)) >>> 0;
      h3 = (h3 ^ (h1 >>> 7)) >>> 0;
    }
  }
  return [h0, h1, h2, h3].map((h) => hex8(h >>> 0));
}

// A fixed scene: a blended two-colour gradient under two overlapping translucent circles, then serif and italic
// text with an emoji. The mix of text shaping, anti-aliasing and compositing is what varies between stacks.
function drawCanvasScene(ctx, w, h) {
  ctx.clearRect(0, 0, w, h);
  const grad = ctx.createLinearGradient(0, 0, w, h);
  grad.addColorStop(0, "#f09");
  grad.addColorStop(0.5, "#0cf");
  grad.addColorStop(1, "#fc0");
  ctx.fillStyle = grad;
  ctx.fillRect(0, 0, w, h);
  ctx.globalCompositeOperation = "multiply";
  for (const [cx, colour] of [[0.35, "#e33"], [0.6, "#39f"]]) {
    ctx.fillStyle = colour;
    ctx.beginPath();
    ctx.arc(w * cx, h * 0.5, h * 0.4, 0, Math.PI * 2);
    ctx.fill();
  }
  ctx.globalCompositeOperation = "source-over";
  ctx.textBaseline = "alphabetic";
  ctx.fillStyle = "#222";
  ctx.font = "600 22px 'Times New Roman', Georgia, serif";
  ctx.fillText("GPU fp ✨ 1234567890", 8, h * 0.55);
  ctx.strokeStyle = "rgba(0,0,0,0.5)";
  ctx.font = "italic 16px system-ui, sans-serif";
  ctx.strokeText("the quick brown fox", 10, h * 0.85);
}

// MD5 of a byte array as 32 lowercase hex digits. BrowserLeaks publishes its canvas signature as an MD5, and Web
// Crypto does not offer that digest, so it is computed here to give a value that can be compared with theirs.
const MD5_SHIFTS = [7, 12, 17, 22, 5, 9, 14, 20, 4, 11, 16, 23, 6, 10, 15, 21];

function md5(bytes) {
  const sines = Array.from({ length: 64 }, (_, i) => Math.floor(Math.abs(Math.sin(i + 1)) * 2 ** 32) >>> 0);
  const padded = new Uint8Array((((bytes.length + 8) >> 6) + 1) << 6);
  padded.set(bytes);
  padded[bytes.length] = 0x80;
  const view = new DataView(padded.buffer);
  view.setUint32(padded.length - 8, (bytes.length << 3) >>> 0, true);
  view.setUint32(padded.length - 4, Math.floor(bytes.length / 2 ** 29), true);
  const state = [0x67452301, 0xefcdab89, 0x98badcfe, 0x10325476];
  for (let block = 0; block < padded.length; block += 64) {
    let [a, b, c, d] = state;
    for (let i = 0; i < 64; i++) {
      const round = i >> 4;
      const mixed = [(b & c) | (~b & d), (d & b) | (~d & c), b ^ c ^ d, c ^ (b | ~d)][round];
      const word = [i, 5 * i + 1, 3 * i + 5, 7 * i][round] & 15;
      const sum = (a + mixed + sines[i] + view.getUint32(block + word * 4, true)) >>> 0;
      const shift = MD5_SHIFTS[round * 4 + (i & 3)];
      a = d;
      d = c;
      c = b;
      b = (b + ((sum << shift) | (sum >>> (32 - shift)))) >>> 0;
    }
    [a, b, c, d].forEach((value, i) => {
      state[i] = (state[i] + value) >>> 0;
    });
  }
  const digest = new DataView(new ArrayBuffer(16));
  state.forEach((value, i) => digest.setUint32(i * 4, value, true));
  return [0, 4, 8, 12].map((at) => hex8(digest.getUint32(at))).join("");
}

// The bytes of the file a data URL holds.
function dataUrlBytes(url) {
  return Uint8Array.from(atob(url.split(",")[1]), (ch) => ch.charCodeAt(0));
}

// What one PNG scanline filter predicts for a byte from the bytes to its left, above it, and above and to the left.
function pngPredict(filter, left, up, upLeft) {
  if (filter === 1) return left;
  if (filter === 2) return up;
  if (filter === 3) return (left + up) >> 1;
  if (filter !== 4) return 0;
  const guess = left + up - upLeft;
  const [toLeft, toUp, toUpLeft] = [left, up, upLeft].map((value) => Math.abs(guess - value));
  if (toLeft <= toUp && toLeft <= toUpLeft) return left;
  return toUp <= toUpLeft ? up : upLeft;
}

// The pixels of a PNG as RGBA bytes, for the 8-bit truecolour files a canvas encodes. The file is decoded here, not
// drawn back onto a canvas, so that reading the answer does not go through the canvas read being checked.
async function decodePng(bytes) {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  let header = null;
  const data = [];
  for (let pos = 8; pos + 12 <= bytes.length;) {
    const length = view.getUint32(pos);
    const type = String.fromCharCode(...bytes.subarray(pos + 4, pos + 8));
    const body = bytes.subarray(pos + 8, pos + 8 + length);
    if (type === "IHDR") {
      header = { width: view.getUint32(pos + 8), height: view.getUint32(pos + 12), depth: body[8], colour: body[9] };
      header.interlaced = body[12] !== 0;
    }
    if (type === "IDAT") data.push(body);
    pos += 12 + length;
  }
  const channels = header ? { 2: 3, 6: 4 }[header.colour] : 0;
  if (!channels || header.depth !== 8 || header.interlaced) throw new Error("the PNG is not 8-bit truecolour");
  const inflated = new Blob(data).stream().pipeThrough(new DecompressionStream("deflate"));
  const raw = new Uint8Array(await new Response(inflated).arrayBuffer());
  const stride = header.width * channels;
  const pixels = new Uint8Array(header.width * header.height * 4).fill(255);
  let above = new Uint8Array(stride);
  for (let y = 0; y < header.height; y++) {
    const filter = raw[y * (stride + 1)];
    const line = raw.slice(y * (stride + 1) + 1, (y + 1) * (stride + 1));
    for (let i = 0; i < stride; i++) {
      const left = i >= channels ? line[i - channels] : 0;
      const upLeft = i >= channels ? above[i - channels] : 0;
      line[i] = (line[i] + pngPredict(filter, left, above[i], upLeft)) & 255;
    }
    for (let x = 0; x < header.width; x++) {
      pixels.set(line.subarray(x * channels, (x + 1) * channels), (y * header.width + x) * 4);
    }
    above = line;
  }
  return pixels;
}

// A frame like the one BrowserLeaks makes its canvas in: a sandboxed, same-origin about:blank document. It is given
// no size instead of being hidden with display:none, so the document inside still has a layout to draw text with.
function sandboxFrame() {
  const frame = document.createElement("iframe");
  frame.setAttribute("sandbox", "allow-same-origin");
  frame.setAttribute("aria-hidden", "true");
  frame.style.cssText = "position:absolute;width:0;height:0;border:0";
  document.body.append(frame);
  return frame;
}

// A scene whose every pixel is known in advance: blocks of flat opaque colour, aligned to the pixel grid, which
// every renderer draws exactly. Returns the values a faithful read gives back.
const KNOWN_SIZE = 64;
const KNOWN_BLOCK = 8;

function drawKnownScene(ctx) {
  const expected = new Uint8ClampedArray(KNOWN_SIZE * KNOWN_SIZE * 4);
  for (let by = 0; by < KNOWN_SIZE / KNOWN_BLOCK; by++) {
    for (let bx = 0; bx < KNOWN_SIZE / KNOWN_BLOCK; bx++) {
      const colour = [(bx * 37 + by * 11 + 20) & 255, (bx * 59 + by * 23 + 70) & 255, (bx * 17 + by * 101 + 130) & 255];
      ctx.fillStyle = `rgb(${colour.join(",")})`;
      ctx.fillRect(bx * KNOWN_BLOCK, by * KNOWN_BLOCK, KNOWN_BLOCK, KNOWN_BLOCK);
      for (let y = by * KNOWN_BLOCK; y < (by + 1) * KNOWN_BLOCK; y++) {
        for (let x = bx * KNOWN_BLOCK; x < (bx + 1) * KNOWN_BLOCK; x++) {
          expected.set([...colour, 255], (y * KNOWN_SIZE + x) * 4);
        }
      }
    }
  }
  return expected;
}

// How many of the values read back differ from the ones drawn. Any that do were changed on the way out, so this
// shows a browser adding noise to a canvas read within a single visit, with no unprotected browser to compare
// against. The pattern is a hash of where and how the values differ, to tell one noise from another.
function alteredValues(read, expected) {
  if (read.length !== expected.length) return { error: `read ${read.length} values, not ${expected.length}` };
  const difference = new Uint8Array(read.length);
  let changed = 0;
  for (let i = 0; i < read.length; i++) {
    difference[i] = read[i] ^ expected[i];
    if (difference[i]) changed += 1;
  }
  return { changed, total: read.length, pattern: changed ? hashPixels(difference).join("-") : "" };
}

// The known scene read back through every way this page makes and reads a canvas: made by the page or inside a
// sandboxed frame, with a default context or one marked willReadFrequently, and read as pixels or as a PNG. A
// defence that covers only some of these shows up as noise on those paths and none on the others.
async function probeReadPaths() {
  const cells = [];
  for (const home of ["page", "sandboxed frame"]) {
    for (const context of ["willReadFrequently", "default"]) {
      const frame = home === "page" ? null : sandboxFrame();
      const cell = (read, outcome) =>
        cells.push({ home, context, read, key: `${home}|${context}|${read}`, ...outcome });
      try {
        const canvas = (frame ? frame.contentDocument : document).createElement("canvas");
        canvas.width = KNOWN_SIZE;
        canvas.height = KNOWN_SIZE;
        const ctx = canvas.getContext("2d", context === "default" ? undefined : { willReadFrequently: true });
        const expected = drawKnownScene(ctx);
        const reads = [
          ["getImageData", () => ctx.getImageData(0, 0, KNOWN_SIZE, KNOWN_SIZE).data],
          ["toDataURL", () => decodePng(dataUrlBytes(canvas.toDataURL("image/png")))],
        ];
        for (const [read, take] of reads) {
          try {
            cell(read, alteredValues(await take(), expected));
          } catch (e) {
            cell(read, { error: String(e.message || e) });
          }
        }
      } catch (e) {
        cell("any", { error: String(e.message || e) });
      } finally {
        if (frame) frame.remove();
      }
    }
  }
  return cells;
}

// The drawing the BrowserLeaks canvas test makes, call for call, including the canvas being sized after its context
// is taken.
function drawBrowserleaksScene(canvas, ctx) {
  const text = "BrowserLeaks,com <canvas> 1.0";
  canvas.setAttribute("width", 220);
  canvas.setAttribute("height", 30);
  ctx.textBaseline = "top";
  ctx.font = "14px 'Arial'";
  ctx.textBaseline = "alphabetic";
  ctx.fillStyle = "#f60";
  ctx.fillRect(125, 1, 62, 20);
  ctx.fillStyle = "#069";
  ctx.fillText(text, 2, 15);
  ctx.fillStyle = "rgba(102, 204, 0, 0.7)";
  ctx.fillText(text, 4, 17);
}

// The BrowserLeaks signature of a fresh canvas made by the given document: the MD5 of the PNG that toDataURL
// returns. Read twice from the one canvas, to show noise that changes between reads of the same pixels.
function browserleaksSignature(doc) {
  const canvas = doc.createElement("canvas");
  const ctx = canvas.getContext("2d");
  if (!ctx) throw new Error("getContext('2d') returned null");
  drawBrowserleaksScene(canvas, ctx);
  const t0 = performance.now();
  const hash = md5(dataUrlBytes(canvas.toDataURL("image/png")));
  const elapsedMs = Math.round((performance.now() - t0) * 10) / 10;
  return { hash, elapsedMs, again: md5(dataUrlBytes(canvas.toDataURL("image/png"))) };
}

// This page's own canvas fingerprint: its scene on a canvas the page made, read with getImageData.
function runCanvasScene(debug) {
  const w = 300, h = 70;
  const canvas = document.createElement("canvas");
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  if (!ctx) return { error: "2-D canvas is unavailable (getContext('2d') returned null)" };
  const once = () => {
    drawCanvasScene(ctx, w, h);
    return hashPixels(ctx.getImageData(0, 0, w, h).data);
  };
  const t0 = performance.now();
  const first = once();
  const elapsedMs = Math.round((performance.now() - t0) * 10) / 10;
  const second = once();
  const stable = first.every((x, i) => x === second[i]);
  debug.push(`scene read 1: ${first.join("-")}`);
  debug.push(`scene read 2: ${second.join("-")}`);
  const results = { lane0: first[0], lane1: first[1], lane2: first[2], lane3: first[3] };
  return { results, elapsedMs, stable, hash: first.join("-"), shown: first.join("-") };
}

// The BrowserLeaks canvas test, made the way that site makes it: a canvas created by a sandboxed frame's document,
// read with toDataURL. It is taken in two separate frames, as two loads of that site would, and once more on a
// canvas the page made, to show whether the frame changes the answer.
function runBrowserleaks(debug) {
  const inFrame = () => {
    const frame = sandboxFrame();
    try {
      return browserleaksSignature(frame.contentDocument);
    } finally {
      frame.remove();
    }
  };
  const first = inFrame();
  const second = inFrame();
  const page = browserleaksSignature(document);
  debug.push(`BrowserLeaks test, frame 1: ${first.hash} then ${first.again} on a second read of that canvas`);
  debug.push(`BrowserLeaks test, frame 2: ${second.hash} then ${second.again} on a second read of that canvas`);
  debug.push(`BrowserLeaks test, canvas made by the page: ${page.hash}`);
  const stable = [first.again, second.hash, second.again].every((hash) => hash === first.hash);
  const results = { md5: first.hash };
  const pageDiffers = page.hash !== first.hash;
  return {
    results,
    elapsedMs: first.elapsedMs,
    stable,
    hash: first.hash,
    shown: first.hash.toUpperCase(),
    pageDiffers,
  };
}

// Each canvas fingerprint with the name the registry files it under, the read path of the known-value check that
// it uses, and the words that describe it on the page.
const CANVAS_PROBES = [
  {
    iface: "canvas",
    title: "This page's scene, read with getImageData",
    short: "scene",
    suffix: "canvas 2D",
    path: "page|willReadFrequently|getImageData",
    run: runCanvasScene,
  },
  {
    iface: "canvas-browserleaks",
    title: "BrowserLeaks canvas test, read with toDataURL from a sandboxed frame",
    short: "BrowserLeaks test",
    suffix: "canvas 2D, BrowserLeaks test",
    path: "sandboxed frame|default|toDataURL",
    run: runBrowserleaks,
  },
];

async function runCanvas() {
  const debug = [];
  const probes = CANVAS_PROBES.map((probe) => {
    try {
      return { ...probe, ...probe.run(debug) };
    } catch (e) {
      debug.push(`${probe.short} failed: ${e.message || e}`);
      return { ...probe, error: `2-D canvas read threw (${e.message || e})` };
    }
  });
  const noise = await probeReadPaths();
  for (const cell of noise) {
    const outcome = cell.error ? `not checked (${cell.error})` : `${cell.changed} of ${cell.total} values altered`;
    debug.push(`known values, ${cell.home}, ${cell.context} context, ${cell.read}: ${outcome} ${cell.pattern || ""}`);
  }
  return { probes, noise, debug };
}

// --- signature ------------------------------------------------------------------------------------------------

function words(outcomes) {
  return outcomes.map((o) => ("bits" in o ? hex8(o.bits) : "????????"));
}

function completed(outcomes) {
  return outcomes.filter((o) => "bits" in o).length;
}

function deviations(outcomes, reference) {
  return outcomes.filter((o, i) => "bits" in o && o.bits !== reference[i]).length;
}

function fingerprint(outcomes) {
  return words(outcomes).join("-");
}

// A result keyed by probe name, e.g. { multiply_add: "33800000", fma: "????????", ... }. This is what the registry
// stores and matches on, so a probe can be added or removed without shifting every other value.
function resultsOf(data, outcomes) {
  const results = {};
  data.probes.forEach((p, i) => {
    results[p.name] = "bits" in outcomes[i] ? hex8(outcomes[i].bits) : "????????";
  });
  return results;
}

function sameResults(a, b) {
  if (!a || !b) return false;
  const keys = Object.keys(a);
  return keys.length === Object.keys(b).length && keys.every((k) => a[k] === b[k]);
}

function classOf(iface, results, registry) {
  const match = registry.find((r) => r.interface === iface && sameResults(r.results, results));
  return match ? match.label : UNRECOGNISED;
}

function traits(data, outcomes) {
  const byName = new Map(data.probes.map((p, i) => [p.name, outcomes[i]]));
  return Object
    .entries(data.choices)
    .map(([name, legal]) => {
      const o = byName.get(name);
      let chosen = "failed";
      if (o && "bits" in o) {
        const hit = legal.find(([bits]) => parseInt(bits, 16) >>> 0 === o.bits);
        chosen = hit ? hit[1] : "other";
      }
      return `${name}=${chosen}`;
    })
    .join(" ");
}

// --- rendering ------------------------------------------------------------------------------------------------

function el(tag, attrs, children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (k === "class") node.className = v;
    else if (k === "text") node.textContent = v;
    else node.setAttribute(k, v);
  }
  for (const child of children || []) node.append(child);
  return node;
}

// True when a local server.py is plausibly reachable at this origin (localhost, a private LAN IP, or a nip.io
// host). Public static hosts -- github.io, vercel.app, pages.dev and the like -- are not, and have no /register,
// so the submit UI (and the known-device timing contribution) is never offered there, only the copyable line.
function looksLocalServer() {
  const h = location.hostname;
  return h === "localhost" || h === "127.0.0.1" || h === "::1" ||
    /\.local$/i.test(h) || /\.nip\.io$/i.test(h) ||
    /^10\./.test(h) || /^192\.168\./.test(h) || /^172\.(1[6-9]|2\d|3[01])\./.test(h);
}

// Tracks the latest status of each interface so one banner can show that all three were tested and whether each is
// a new device to submit; the result cards are far enough apart that users were submitting only the first.
const STATUS = { webgpu: null, webgl2: null, webgl1: null, canvas: null };

function updateStatusSummary() {
  const node = document.getElementById("status-summary");
  if (!node) return;
  const parts = [];
  let anyNew = false;
  let anyKnown = false;
  const named = [["webgpu", "WebGPU"], ["webgl2", "WebGL 2"], ["webgl1", "WebGL 1"], ["canvas", "Canvas 2D"]];
  for (const [iface, name] of named) {
    const st = STATUS[iface];
    let state;
    if (!st) {
      state = "...";
    } else if (st.blocked) {
      state = "blocked";
    } else if (st.error) {
      state = "unavailable";
    } else if (st.present) {
      state = "recognised";
      anyKnown = true;
    } else {
      state = "NEW -- please submit";
      anyNew = true;
    }
    if (st && st.differs) state += " (split and combined differ)";
    if (st && st.note) state += ` (${st.note})`;
    parts.push(`${name}: ${state}`);
  }
  node.textContent = (WINDOW === "private" ? "PRIVATE WINDOW    |    " : "") + parts.join("    |    ");
  node.className = "banner" + (anyNew ? " new" : anyKnown ? " known" : "");
}

// Shared registry-submission UI, used for a new numeric device, a new canvas signature, and a blocked interface.
// On a local server it shows a label field and a submit button; elsewhere, only the copyable JSON line. makeEntry
// builds the payload from the edited label; rerun re-runs every interface after a successful add (a new signature
// then shows as recognised). Kept in one place so the three callers do not each carry a copy of the fetch wiring.
function entryPanel({ guessed, makeEntry: bare, intro, elsewhereIntro, buttonText, rerun }) {
  // Every entry says which kind of window it came from, so a private window's results get entries of their own.
  const makeEntry = (label) => ({ ...bare(label), window: WINDOW });
  const children = [];
  let currentLabel = () => guessed;
  if (looksLocalServer()) {
    const labelInput = el("input", { class: "labelin", type: "text", value: guessed });
    labelInput.value = guessed;
    currentLabel = () => labelInput.value;
    const submit = el("button", { class: "submit", text: buttonText });
    const status = el("span", { class: "submitstatus" });
    submit.addEventListener("click", async () => {
      submit.disabled = true;
      status.textContent = "submitting...";
      try {
        const entry = makeEntry(labelInput.value);
        const res = await fetch("/register", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(entry),
        });
        const reply = await res.json().catch(() => ({}));
        if (res.ok && reply.ok) {
          if (entry.results) JUST_ADDED.add(signatureKey(entry.interface, entry.results));
          status.textContent = rerun ? "added -- re-checking..." : (reply.message || "recorded");
          await reloadRegistry();
          if (rerun) await runAll();
        } else {
          status.textContent = reply.message || `not added (HTTP ${res.status})`;
          submit.disabled = false;
        }
      } catch (e) {
        status.textContent = `submit failed (is this served by server.py?): ${e.message || e}`;
        submit.disabled = false;
      }
    });
    children.push(el("div", { class: "k", text: intro }));
    children.push(el("div", { class: "submitrow" }, [labelInput, submit, status]));
    children.push(el("div", { class: "k", text: "or copy this line to paste into the submit form elsewhere:" }));
  } else {
    children.push(el("div", { class: "k", text: elsewhereIntro }));
  }
  const pre = el("pre", { class: "code", text: JSON.stringify(makeEntry(guessed)) });
  const copy = el("button", { class: "copy", text: "Copy line" });
  copy.addEventListener("click", async () => {
    try {
      await navigator.clipboard.writeText(JSON.stringify(makeEntry(currentLabel())));
      copy.textContent = "Copied";
    } catch {
      copy.textContent = "select and copy";
    }
  });
  children.push(pre, copy);
  return el("div", { class: "addentry" }, children);
}

// Signatures added from this page since it was last run by hand. A new signature is submitted with its timing, and
// the page then re-runs to show it as recognised; that re-run's timing nearly always falls outside the one value
// just recorded, so offering it would ask for a second submission straight after the first.
const JUST_ADDED = new Set();

function signatureKey(iface, results) {
  return `${iface}|${Object.values(results).join("-")}`;
}

// The timing block for a signature already in the registry: how this run compares with the recorded range and, on
// a local server and only when it would widen the range, a button to submit it.
function timingPanel(iface, label, elapsedMs, results, match) {
  const run = RUN;
  const lo = match ? match[`${run}_ms_min`] : null;
  const hi = match ? match[`${run}_ms_max`] : null;
  if (JUST_ADDED.has(signatureKey(iface, results))) {
    const text = `this ${run} run: ${elapsedMs} ms (signature added just now, with its timing)`;
    return el("div", { class: "timing" }, [el("span", { class: "k", text })]);
  }
  let status;
  if (lo == null || hi == null) {
    status = `this ${run} run: ${elapsedMs} ms (no ${run}-run timing recorded for this signature yet)`;
  } else if (elapsedMs < lo) {
    status = `this ${run} run: ${elapsedMs} ms, a new fastest (known ${run}-run range ${lo}-${hi} ms)`;
  } else if (elapsedMs > hi) {
    status = `this ${run} run: ${elapsedMs} ms, a new slowest (known ${run}-run range ${lo}-${hi} ms)`;
  } else {
    status = `this ${run} run: ${elapsedMs} ms, within the known ${run}-run range ${lo}-${hi} ms`;
  }
  const wouldWiden = lo == null || hi == null || elapsedMs < lo || elapsedMs > hi;
  const block = el("div", { class: "timing" }, [el("span", { class: "k", text: status })]);
  if (looksLocalServer() && wouldWiden) {
    const btn = el("button", { class: "copy", text: "Submit this timing" });
    const note = el("span", { class: "submitstatus" });
    btn.addEventListener("click", async () => {
      btn.disabled = true;
      note.textContent = " submitting...";
      try {
        const res = await fetch("/register", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ interface: iface, label, window: WINDOW, elapsed_ms: elapsedMs, run, results }),
        });
        const reply = await res.json().catch(() => ({}));
        note.textContent = ` ${reply.message || (reply.ok ? "submitted" : "not submitted")}`;
        await reloadRegistry();
      } catch (e) {
        note.textContent = ` submit failed: ${e.message || e}`;
        btn.disabled = false;
      }
    });
    block.append(el("span", { text: " " }), btn, note);
  }
  return block;
}

// The registry row that already records this browser withholding an interface for this reason, if there is one. A
// blocked row has no signature to match on, so it is matched on its label naming this browser's family and major
// version and this operating system; a label that goes on to name the device still matches. The reason has to
// match as well, because browsers built on Firefox report themselves as Firefox: a row saying that one of them
// blocks an interface in one way must not stand in for another that blocks it in a different way.
function knownBlocked(iface, reason) {
  const browser = ENV.browser || "";
  const cut = browser.lastIndexOf(" ");
  if (cut < 0 || !ENV.os || !DATA) return null;
  const family = `${browser.slice(0, cut)} ${browser.slice(cut + 1).split(".")[0]}`;
  const names = (label) => {
    const at = label.indexOf(family);
    return at >= 0 && !/[0-9]/.test(label.charAt(at + family.length)) && label.includes(ENV.os);
  };
  const same = (r) => r.interface === iface && r.blocked && String(r.reason) === reason && names(String(r.label));
  return (DATA.registry || []).find(same) || null;
}

// Offer to record that this browser blocks an interface, so the registry can show which browsers do, unless the
// registry already says so. A blocked entry carries a browser/OS label and a reason, and no signature.
function blockedOffer(iface, reason) {
  const trimmed = String(reason).slice(0, 160);
  const known = knownBlocked(iface, trimmed);
  if (known) return el("div", { class: "meta ok", text: `Already recorded in the registry: ${known.label}` });
  const guessed = guessBrowserLabel();
  const makeEntry = (label) => ({ interface: iface, label: (label || guessed).trim(), blocked: true, reason: trimmed });
  return entryPanel({
    guessed,
    makeEntry,
    intro: `This browser blocks ${iface}. Record which browser blocks it:`,
    elsewhereIntro: `This browser blocks ${iface}. Copy this line and submit it on a machine running server.py:`,
    buttonText: "Record as blocked",
    rerun: false,
  });
}

// A label for an entry with no GPU renderer string to go on (a blocked interface, or a canvas signature): the
// browser and operating system from navigator.*, which is all that identifies the stack in those cases.
function guessBrowserLabel() {
  const where = (ENV.os + (ENV.osVersion ? ` ${ENV.osVersion}` : "")).trim();
  const who = ENV.browser || "unknown browser";
  return where ? `${who} on ${where}` : who;
}

function render(container, iface, result, data, reference) {
  container.textContent = "";
  const showDebug = () => {
    if (result.debug && result.debug.length) {
      container.append(el("details", { class: "debug" }, [
        el("summary", { text: "debug log" }),
        el("pre", { class: "code", text: result.debug.join("\n") }),
      ]));
    }
  };
  if (result.error) {
    container.append(el("p", { class: "error", text: `${iface}: ${result.error}` }));
    if (result.renderer) container.append(el("div", { class: "meta", text: `reported GPU: ${result.renderer}` }));
    container.append(blockedOffer(iface, result.error));
    showDebug();
    STATUS[iface] = { error: true };
    for (const [mode] of MODES) LAST[`${iface}-${mode}`] = "unavailable";
    updateStatusSummary();
    return;
  }
  const { modes, renderer } = result;
  for (const [mode] of MODES) LAST[`${iface}-${mode}`] = fingerprint(modes[mode].outcomes);

  // Say first whether the two modes agree: a difference means the shader around an expression changes its result
  // on this device, which is itself a distinguishing feature.
  const word = (o) => ("bits" in o ? hex8(o.bits) : "????????");
  const differing = data
    .probes
    .map((p, i) => [p.name, word(modes.split.outcomes[i]), word(modes.combined.outcomes[i])])
    .filter(([, split, combined]) => split !== combined);
  if (differing.length) {
    const list = differing.map(([name, split, combined]) => `${name} (split ${split}, combined ${combined})`);
    const text = `Split and combined signatures differ on ${differing.length} of ${data.probes.length} ` +
      `expressions, so the shader around an expression changes its result here: ${list.join("; ")}`;
    container.append(el("p", { class: "banner new", text }));
  } else {
    container.append(el("p", { class: "meta ok", text: "Split and combined signatures are identical here." }));
  }

  // A signature the registry has under one mode only is new under the other, even when the two modes agree. Say so
  // and offer the known mode's label, so that a device the registry already has does not read as a new device.
  const known = Object.fromEntries(MODES.map(([mode]) => {
    return [mode, classOf(`${iface}-${mode}`, resultsOf(data, modes[mode].outcomes), data.registry)];
  }));
  const states = MODES.map(([mode, title]) => {
    const other = mode === "split" ? "combined" : "split";
    const twin = !differing.length && known[other] !== UNRECOGNISED ? { mode: other, label: known[other] } : null;
    return renderMode(container, `${iface}-${mode}`, title, modes[mode], renderer, data, reference, twin);
  });
  showDebug();
  STATUS[iface] = { present: states.every((state) => state.present), differs: differing.length > 0 };
  updateStatusSummary();
}

// Where the registry already has a signature that it lacks here: under the other kind of window, or under the other
// mode. Either makes a familiar device read as familiar instead of as new, and supplies its label.
function knownElsewhere(iface, results, twin) {
  const kin = inOtherWindow(iface, results);
  if (kin) {
    const there = WINDOW === "private" ? "a normal window" : "a private window";
    return {
      label: kin.label,
      headline: `not recorded for a ${WINDOW} window yet -- same signature as in ${there}`,
      intro: `The registry has this signature from ${there} only. Record it for a ${WINDOW} window too:`,
    };
  }
  if (twin) {
    return {
      label: twin.label,
      headline: `not recorded in this mode yet -- same signature as in ${twin.mode} mode`,
      intro: `The registry has this signature in ${twin.mode} mode only. Record it for this mode too:`,
    };
  }
  return null;
}

// One mode's block within an interface's card: whether its signature is in the registry under that mode's interface
// name, the per-probe detail, and the offer to submit a new signature or a new timing.
function renderMode(container, iface, title, run, renderer, data, reference, twin) {
  const { outcomes, elapsedMs } = run;
  const results = resultsOf(data, outcomes);
  const label = classOf(iface, results, data.registry);
  const present = label !== UNRECOGNISED;
  const elsewhere = present ? null : knownElsewhere(iface, results, twin);
  const absent = elsewhere ? elsewhere.headline : "not present -- new device";

  // Headline, always visible: the mode and whether this signature is already in the registry.
  container.append(el("div", { class: "headline" }, [
    el("span", { class: "hl-iface", text: `${iface} (${title}):` }),
    el("span", { class: present ? "hl-match" : "hl-absent", text: present ? label : absent }),
  ]));

  // The per-probe detail folds away for a device already in the registry and opens automatically for a new one.
  const detail = el("details", { class: "result-details" }, [el("summary", { text: "probe detail" })]);
  if (!present) detail.open = true;

  detail.append(el("div", { class: "summary" }, [
    el("div", {}, [el("span", { class: "k", text: "reported GPU " }), el("span", { class: "v", text: renderer })]),
    el("div", {}, [
      el("span", { class: "k", text: "answered " }),
      el("span", { class: "v", text: `${completed(outcomes)} of ${outcomes.length}` }),
    ]),
    el("div", {}, [
      el("span", { class: "k", text: "elapsed " }),
      el("span", { class: "v", text: elapsedMs != null ? `${elapsedMs} ms` : "n/a" }),
    ]),
    el("div", {}, [
      el("span", { class: "k", text: "deviations " }),
      el("span", { class: "v", text: String(deviations(outcomes, reference)) }),
    ]),
    el("div", {}, [
      el("span", { class: "k", text: "traits " }),
      el("span", { class: "v", text: traits(data, outcomes) }),
    ]),
    el("div", { class: "fp" }, [
      el("span", { class: "k", text: "fingerprint " }),
      el("span", { class: "v mono", text: fingerprint(outcomes) }),
    ]),
  ]));

  const rows = [el("tr", {}, [
    el("th", { text: "probe" }),
    el("th", { text: "gpu" }),
    el("th", { text: "correct" }),
    el("th", { text: "ulps" }),
    el("th", { text: "choice" }),
  ])];
  const choiceNames = new Set(Object.keys(data.choices));
  data.probes.forEach((probe, i) => {
    const o = outcomes[i];
    const correct = reference[i];
    let gpu, ulps, cls;
    if ("bits" in o) {
      gpu = hex8(o.bits);
      cls = o.bits === correct ? "ok" : "dev";
      if (choiceNames.has(probe.name)) {
        // For the choice probes a bit-pattern difference is not a ulp distance; the choice column says what differs.
        ulps = o.bits === correct ? "." : "";
      } else {
        const d = (o.bits | 0) - (correct | 0);
        ulps = o.bits === correct ? "." : (d > 0 ? `+${d}` : String(d));
      }
    } else {
      gpu = "refused";
      ulps = "";
      cls = "refused";
    }
    let choice = "";
    if (choiceNames.has(probe.name) && "bits" in o) {
      const legal = data.choices[probe.name].find(([b]) => parseInt(b, 16) >>> 0 === o.bits);
      choice = legal ? legal[1] : "other";
    }
    rows.push(el("tr", { class: cls }, [
      el("td", { text: probe.name }),
      el("td", { class: "mono", text: gpu }),
      el("td", { class: "mono", text: hex8(correct) }),
      el("td", { class: "mono", text: ulps }),
      el("td", { text: choice }),
    ]));
  });
  detail.append(el("table", {}, rows));
  container.append(detail);

  if (!present) {
    const guessed = elsewhere ? elsewhere.label : guessLabel(renderer);
    const run = RUN;
    const makeEntry = (lbl) => ({
      interface: iface,
      label: (lbl || guessed).trim(),
      elapsed_ms: elapsedMs,
      run,
      results,
    });
    container.append(entryPanel({
      guessed,
      makeEntry,
      intro: elsewhere ? elsewhere.intro : "New device. Label it and submit it to the registry:",
      elsewhereIntro: "New device. Copy this line and paste it into the submit form on a machine running server.py:",
      buttonText: "Submit to registry",
      rerun: true,
    }));
  }

  if (present && elapsedMs != null) {
    const match = data.registry.find((r) => r.interface === iface && sameResults(r.results, results));
    container.append(timingPanel(iface, label, elapsedMs, results, match));
  }

  return { present, label };
}

// The canvas card. Unlike render(), there is no per-expression table or reference: a 2-D canvas fingerprint is one
// hash. A hard read failure shows as blocked; two differing reads show as randomised (both offer to be recorded);
// otherwise the hash is matched against the registry like the numeric signatures.
function renderCanvas(container, result, data) {
  container.textContent = "";
  const showDebug = () => {
    if (result.debug && result.debug.length) {
      container.append(el("details", { class: "debug" }, [
        el("summary", { text: "debug log" }),
        el("pre", { class: "code", text: result.debug.join("\n") }),
      ]));
    }
  };
  renderReadPaths(container, result.noise);
  const states = result.probes.map((probe) => renderCanvasProbe(container, probe, result.noise, data));
  showDebug();
  // One summary for the two fingerprints: new if either is, blocked only if both are, and a note naming the one
  // that is randomised when the other is not.
  const randomised = result.probes.filter((_, i) => states[i] === "blocked").map((probe) => probe.short);
  const blocked = randomised.length === states.length;
  const note = !blocked && randomised.length ? `${randomised.join(", ")} randomised` : "";
  STATUS.canvas = { blocked, present: !states.includes("new"), note };
  updateStatusSummary();
}

// The known-value check, shown before the fingerprints: it says which ways of reading a canvas this browser alters,
// and so what each hash below is worth. The visit history says whether an alteration is the same in every session.
function renderReadPaths(container, cells) {
  const altered = cells.filter((cell) => cell.changed);
  const unchecked = cells.filter((cell) => cell.error);
  LAST["canvas-noise"] = altered.length ? altered.map((cell) => `${cell.key} ${cell.pattern}`).join("; ") : "none";
  const checked = cells.length - unchecked.length;
  const text = altered.length
    ? `Noise check: this browser altered known values on ${altered.length} of ${checked} ways of reading a ` +
      "canvas, so it adds noise to those reads."
    : `Noise check: known values came back unaltered through all ${checked} ways of reading a canvas that could ` +
      "be checked, so this browser adds no noise to canvas reads on this page.";
  container.append(el("p", { class: altered.length ? "banner new" : "meta ok", text }));
  const rows = cells.map((cell) => {
    let outcome = "unaltered";
    if (cell.error) outcome = `not checked (${cell.error})`;
    else if (cell.changed) outcome = `${cell.changed} of ${cell.total} values altered (pattern ${cell.pattern})`;
    const fields = [cell.home, cell.context, cell.read, outcome];
    return el(
      "tr",
      { class: cell.changed ? "dev" : cell.error ? "refused" : "" },
      fields.map((f) => el("td", { text: f })),
    );
  });
  const heads = ["canvas made by", "context", "read with", "known values"];
  const table = el("table", {}, [el("tr", {}, heads.map((head) => el("th", { text: head }))), ...rows]);
  container.append(el("details", { class: "result-details", open: "" }, [
    el("summary", { text: "known values read back through each way of making and reading a canvas" }),
    table,
  ]));
}

// One canvas fingerprint's block in the card: whether it is stable within this visit and across visits, whether
// the registry has it, and the offer to submit a new signature, a new timing, or the fact that it is randomised.
// Returns "blocked" when the browser defeats this fingerprint, "present" when the registry has it, else "new".
function renderCanvasProbe(container, probe, noise, data) {
  const { iface, title, error, results, elapsedMs, stable, hash, shown } = probe;
  container.append(el("h3", { text: title }));
  const headline = (cls, text) => {
    const parts = [el("span", { class: "hl-iface", text: `${iface}:` }), el("span", { class: cls, text })];
    container.append(el("div", { class: "headline" }, parts));
  };
  if (error) {
    container.append(el("p", { class: "error", text: `${iface}: ${error}` }));
    container.append(blockedOffer(iface, error));
    LAST[iface] = "unavailable";
    return "blocked";
  }
  if (probe.pageDiffers) {
    const text = "The same drawing on a canvas made by the page gave a different signature, so where the canvas " +
      "is made changes what this browser returns.";
    container.append(el("p", { class: "banner new", text }));
  }
  if (!stable) {
    LAST[iface] = "differs on every read";
    headline("hl-absent", "randomised -- canvas fingerprinting is being defeated");
    const text = `elapsed ${elapsedMs} ms; two reads of the one scene differed, so the browser injects per-read noise.`;
    container.append(el("div", { class: "timing", text }));
    container.append(blockedOffer(iface, "randomised: two reads of the same scene differed (per-read noise)"));
    return "blocked";
  }
  // Two matching reads only rule out noise that changes on every read. Noise keyed to the session survives them,
  // so the hash is also checked against the ones this browser returned on earlier visits.
  LAST[iface] = hash;
  const earlier = earlierHashes(iface, hash);
  if (earlier.length) {
    headline("hl-absent", "randomised between visits -- not a stable fingerprint");
    const text = `elapsed ${elapsedMs} ms; this visit's hash ${shown} differs from ${earlier.length} earlier ` +
      `one${earlier.length === 1 ? "" : "s"} in this browser. Noise keyed to the session does this; so can a ` +
      "browser or font update, in which case forget the earlier visits below.";
    container.append(el("div", { class: "timing", text }));
    const reason = "randomised: hash differs between visits in one browser (noise keyed to the session)";
    container.append(blockedOffer(iface, reason));
    return "blocked";
  }
  const label = classOf(iface, results, data.registry);
  const present = label !== UNRECOGNISED;
  const elsewhere = present ? null : knownElsewhere(iface, results, null);
  headline(
    present ? "hl-match" : "hl-absent",
    present ? label : elsewhere ? elsewhere.headline : "not present -- new signature",
  );
  const detail = el("details", { class: "result-details" }, [el("summary", { text: "hash detail" })]);
  if (!present) detail.open = true;
  detail.append(el("div", { class: "summary" }, [
    el("div", {}, [el("span", { class: "k", text: "elapsed " }), el("span", { class: "v", text: `${elapsedMs} ms` })]),
    el("div", {}, [
      el("span", { class: "k", text: "stable " }),
      el("span", { class: "v", text: "yes (the reads in this visit matched)" }),
    ]),
    el("div", { class: "fp" }, [
      el("span", { class: "k", text: "hash " }),
      el("span", { class: "v mono", text: shown }),
    ]),
  ]));
  container.append(detail);
  if (present) {
    const match = data.registry.find((r) => r.interface === iface && sameResults(r.results, results));
    container.append(timingPanel(iface, label, elapsedMs, results, match));
    return "present";
  }
  const noised = noise.some((cell) => cell.key === probe.path && cell.changed) ? " (noise added to reads)" : "";
  const guessed = elsewhere ? elsewhere.label : `${guessBrowserLabel()} -- ${probe.suffix}${noised}`;
  const run = RUN;
  const makeEntry = (lbl) => ({
    interface: iface,
    label: (lbl || guessed).trim(),
    elapsed_ms: elapsedMs,
    run,
    results,
  });
  container.append(entryPanel({
    guessed,
    makeEntry,
    intro: elsewhere ? elsewhere.intro : `New ${iface} signature. Label it and submit it to the registry:`,
    elsewhereIntro: `New ${iface} signature. Copy this line and submit it on a machine running server.py:`,
    buttonText: "Submit to registry",
    rerun: true,
  }));
  return "new";
}

// --- labelling an unknown device ------------------------------------------------------------------------------

// A best-guess label for a device not yet in the registry, assembled from Web APIs: the GPU and its backend from
// the renderer string, and the operating system and browser (Brave included) from navigator.*. It is written
// into the registry entry so the entry is self-describing without the user having to name the device by hand.

let ENV = { os: "", osVersion: "", browser: "" };

async function gatherEnv() {
  const env = { os: "", osVersion: "", browser: "" };
  const ua = navigator.userAgent || "";

  // Browser family and version from the user-agent string. detect-browser separates Edge, Opera and Samsung
  // Internet from Chrome, which is what the UA-CH brands list fails to do reliably, and it also names Safari and
  // Firefox, so no hand-rolled regex fallback is needed.
  const info = detect(ua);
  if (info && info.type === "browser") {
    env.browser = info.version ? `${prettyBrowser(info.name)} ${info.version}` : prettyBrowser(info.name);
    env.os = normaliseOs(info.os || "");
  }

  // Brave masks its user agent as Chrome, so no parser can see it; the browser answers directly.
  let isBrave = false;
  try {
    if (navigator.brave && navigator.brave.isBrave) isBrave = await navigator.brave.isBrave();
  } catch { /* ignore */ }
  if (isBrave) {
    const version = (env.browser.match(/[\d.]+$/) || [""])[0];
    env.browser = version ? `Brave ${version}` : "Brave";
  }

  // UA-CH supplies what the frozen user-agent string cannot: the precise OS version (Windows 10 versus 11), and
  // its platform string is preferred for the OS family. navigator.platform is avoided because WebKit freezes it
  // to "MacIntel" even on Apple Silicon.
  const uaData = navigator.userAgentData;
  if (uaData) {
    if (uaData.platform) env.os = uaData.platform;
    try {
      const hi = await uaData.getHighEntropyValues(["platformVersion"]);
      if (hi.platformVersion) env.osVersion = hi.platformVersion;
    } catch { /* high-entropy values may be withheld */ }
  }

  if (!env.browser) env.browser = "unknown browser";
  if (!env.os) env.os = (navigator.platform || "").trim();
  return env;
}

// Map detect-browser's family identifiers to the names used in registry labels.
function prettyBrowser(name) {
  const named = {
    "edge-chromium": "Edge",
    "edge": "Edge",
    "edge-ios": "Edge",
    "chrome": "Chrome",
    "crios": "Chrome",
    "chromium-webview": "Chrome WebView",
    "firefox": "Firefox",
    "fxios": "Firefox",
    "safari": "Safari",
    "ios": "Safari",
    "opera": "Opera",
    "opera-mini": "Opera Mini",
    "samsung": "Samsung Internet",
  };
  if (named[name]) return named[name];
  return name ? name.replace(/(^|[\s-])(\w)/g, (_, sep, ch) => `${sep}${ch.toUpperCase()}`) : "unknown browser";
}

// Reduce detect-browser's OS strings to the family; the UA-CH platform string overrides this when present.
function normaliseOs(os) {
  if (/^Windows/i.test(os)) return "Windows";
  if (/^Mac OS/i.test(os)) return "macOS";
  if (/^Android/i.test(os)) return "Android";
  if (/^Chrome OS/i.test(os)) return "Chrome OS";
  if (/^iOS/i.test(os)) return "iOS";
  if (/^Linux/i.test(os)) return "Linux";
  return os;
}

// Reduce a verbose renderer string to the chip and its backend, e.g. an ANGLE wrapper to "Apple M5 Pro through Metal".
function cleanRenderer(renderer) {
  let s = renderer.replace(/^Google Inc\.\s*\([^)]*\)\s*\/\s*/i, "").trim();
  const backend = /Metal/i.test(s)
    ? "Metal"
    : /Direct3D ?11|D3D11/i.test(s)
    ? "Direct3D 11"
    : /Vulkan/i.test(s)
    ? "Vulkan"
    : /OpenGL ?ES/i.test(s)
    ? "OpenGL ES"
    : /OpenGL/i.test(s)
    ? "OpenGL"
    : "";

  // Pull out the renderer field. ANGLE reports "ANGLE (vendor, renderer, version)", and the renderer itself can
  // contain parentheses -- "Intel(R) Arc(TM) Graphics (0x7D55)", "Vulkan 1.4.312 (NVIDIA GB10 (0x2E12))" -- so the
  // field is taken as the whole middle comma-group, never matched up to the first ")" (which truncated it before).
  let field = s;
  const angle = s.match(/^ANGLE \((.*)\)\s*$/i);
  if (angle) {
    const parts = angle[1].split(/,\s+/);
    field = parts.length >= 3 ? parts.slice(1, -1).join(", ") : parts[parts.length - 1];
  }
  const marked = field.match(/Renderer:\s*(.+)/i);
  if (marked) field = marked[1];

  // Some backends wrap the device as "Vulkan 1.4.312 (Device (0xID))"; there the device is the first parenthesised
  // group, so lift it out when the field opens with a backend word rather than the device name.
  let chip = field;
  if (/^\s*(?:Vulkan|OpenGL|Direct3D|D3D|Metal)\b/i.test(field)) {
    const inner = field.match(/\(([^()]*(?:\([^()]*\)[^()]*)*)\)/);
    if (inner) chip = inner[1];
  }

  // Drop hex device ids and vendor marks, leftover backend and version tokens, and D3D shader-model suffixes.
  chip = chip
    .replace(/\(\s*0x[0-9a-f]+\s*\)/gi, " ")
    .replace(/\((?:R|TM|C)\)/gi, " ")
    .replace(/\b(?:Direct3D\d*|D3D\d*|Vulkan|OpenGL(?: ES)?|Metal)\b/gi, " ")
    .replace(/\b\d+(?:\.\d+)+\b/g, " ")
    .replace(/\b[vp]s_\d[\w.]*/gi, " ")
    .replace(/[()]/g, " ")
    .trim();

  // Collapse repeated words, e.g. Safari returning "apple apple apple apple" for the four adapter fields.
  const name = [...new Set((chip || field).split(/\s+/).filter(Boolean))].join(" ");
  return backend && name ? `${name} through ${backend}` : name;
}

function guessLabel(renderer) {
  const where = (ENV.os + (ENV.osVersion ? ` ${ENV.osVersion}` : "")).trim();
  const on = where ? ` on ${where}` : "";
  const who = ENV.browser ? ` (${ENV.browser})` : "";
  return `${cleanRenderer(renderer)}${on}${who}`;
}

// --- wiring ---------------------------------------------------------------------------------------------------

let DATA = null;
let INPUTS = null;
let REFERENCE = null;

async function fetchRegistry() {
  const res = await fetch(`./registry.yaml?t=${Date.now()}`);
  return jsyaml.load(await res.text()) || [];
}

// --- window kind ----------------------------------------------------------------------------------------------

// Whether this is a private (incognito) window. Some browsers change what they return there -- adding noise to
// canvas reads, for one -- so a private window's results are matched against, and submitted to, registry entries
// of their own. No browser says outright that a window is private. detectincognitojs infers it; it can be wrong on
// a browser newer than it knows and some content blockers stop it loading, so the tick box on the page has the
// last word.
let WINDOW = "normal";

async function detectPrivateWindow() {
  try {
    const lib = await import("https://cdn.jsdelivr.net/npm/detectincognitojs@1.9.0/+esm");
    const detect = lib.detectIncognito || (lib.default && (lib.default.detectIncognito || lib.default));
    return Boolean((await detect()).isPrivate);
  } catch {
    return null;
  }
}

// Narrow the registry to the entries for this kind of window; registryAll keeps every entry.
function applyWindow() {
  DATA.registry = DATA.registryAll.filter((r) => (r.window || "normal") === WINDOW);
}

// The entry that holds this signature for the other kind of window, if there is one.
function inOtherWindow(iface, results) {
  const other = (r) => r.interface === iface && (r.window || "normal") !== WINDOW;
  return DATA.registryAll.find((r) => other(r) && sameResults(r.results, results)) || null;
}

async function setUpWindowKind() {
  const box = document.getElementById("private-window");
  const note = document.getElementById("private-note");
  const detected = await detectPrivateWindow();
  WINDOW = detected ? "private" : "normal";
  box.checked = WINDOW === "private";
  note.textContent = detected === null
    ? "(could not be detected in this browser; tick the box if this is a private window)"
    : `(detected as a ${WINDOW} window; correct the box if that is wrong)`;
  applyWindow();
  box.addEventListener("change", () => {
    WINDOW = box.checked ? "private" : "normal";
    applyWindow();
    JUST_ADDED.clear();
    return runAll();
  });
}

// --- history in this browser ----------------------------------------------------------------------------------

// What each visit to the page returned, kept in this browser's own storage. A canvas defence that adds noise keyed
// to the session gives the same hash on every read within a visit and a different one in the next session, which
// two reads in one visit cannot show; comparing visits can. A private window's storage is its own and is discarded
// when the window closes, so there the history starts empty every time.
const HISTORY_KEY = "gpu-fingerprint-history";
const HISTORY_ORDER = [
  "webgpu-split",
  "webgpu-combined",
  "webgl2-split",
  "webgl2-combined",
  "webgl1-split",
  "webgl1-combined",
  "canvas",
  "canvas-browserleaks",
  "canvas-noise",
];
const LAST = {};

function readHistory() {
  try {
    const visits = JSON.parse(localStorage.getItem(HISTORY_KEY) || "[]");
    return Array.isArray(visits) ? visits : [];
  } catch {
    return null;
  }
}

// An identifier that lasts as long as this tab's session, to tell a reload from a visit in a new session.
function tabSession() {
  try {
    let id = sessionStorage.getItem(HISTORY_KEY);
    if (!id) {
      id = Math.random().toString(36).slice(2, 10);
      sessionStorage.setItem(HISTORY_KEY, id);
    }
    return id;
  } catch {
    return "";
  }
}

// The hashes that earlier visits returned for one canvas fingerprint and this visit did not. Any at all means the
// hash is not stable across visits in this browser.
function earlierHashes(key, hash) {
  const seen = (readHistory() || []).map((visit) => visit.seen && visit.seen[key]);
  return [...new Set(seen.filter((h) => h && /^[0-9a-f-]+$/.test(h) && h !== hash))];
}

// Add this visit to the history and say, for each result, whether it has been the same on every visit.
function recordVisit() {
  const node = document.getElementById("history");
  const visits = readHistory();
  if (!node) return;
  node.textContent = "";
  if (visits === null) {
    node.textContent = "This browser gives the page no storage, so this visit cannot be compared with earlier ones.";
    return;
  }
  visits.push({ at: new Date().toISOString(), tab: tabSession(), seen: { ...LAST } });
  const kept = visits.slice(-50);
  try {
    localStorage.setItem(HISTORY_KEY, JSON.stringify(kept));
  } catch { /* storage full or refused; the summary below still covers this visit */ }
  if (kept.length === 1) {
    node.textContent = "First visit recorded in this browser. Come back after restarting the browser to see " +
      "whether each result is the same in a new session.";
    return;
  }
  const tabs = new Set(kept.map((visit) => visit.tab)).size;
  const parts = HISTORY_ORDER.map((key) => {
    const values = new Set(kept.map((visit) => visit.seen && visit.seen[key]).filter(Boolean));
    return `${key}: ${values.size > 1 ? `${values.size} different results` : "unchanged"}`;
  });
  const changed = parts.some((part) => part.endsWith("results"));
  const since = kept[0].at.slice(0, 10);
  const forget = el("button", { class: "copy", text: "Forget earlier visits" });
  forget.addEventListener("click", () => {
    try {
      localStorage.removeItem(HISTORY_KEY);
    } catch { /* nothing to forget */ }
    node.textContent = "Earlier visits forgotten.";
  });
  node.className = changed ? "meta error" : "meta";
  node.append(
    el("span", { text: `This browser has run the page ${kept.length} times since ${since}, in ${tabs} tab ` }),
    el("span", { text: `session${tabs === 1 ? "" : "s"}. ${parts.join("; ")}. ` }),
    forget,
  );
}

async function load() {
  const res = await fetch("./probes.yaml");
  DATA = jsyaml.load(await res.text());
  DATA.registryAll = await fetchRegistry();
  applyWindow();
  INPUTS = inputsFrom(DATA);
  REFERENCE = DATA.reference.map((h) => parseInt(h, 16) >>> 0);

  // Check the JavaScript recreation of the reference against the data file, bit for bit.
  const recomputed = computeReference(INPUTS);
  const mismatch = recomputed.filter((bits, i) => bits !== REFERENCE[i]);
  const note = document.getElementById("refcheck");
  if (mismatch.length === 0) {
    note.textContent =
      "reference check: the 16 correctly rounded answers recomputed in JavaScript match the data file.";
    note.className = "ok";
  } else {
    note.textContent =
      `reference check: ${mismatch.length} of 16 recomputed answers differ from the data file (see console).`;
    note.className = "error";
    console.warn("reference mismatch", recomputed.map(hex8), REFERENCE.map(hex8));
  }

  ENV = await gatherEnv();
  const where = (ENV.os + (ENV.osVersion ? ` ${ENV.osVersion}` : "")).trim();
  document.getElementById("env").textContent =
    `browser: ${ENV.browser || "unknown"} | os: ${where || "unknown"} | secure context: ${window.isSecureContext} ` +
    `| WebGPU present: ${!!navigator.gpu}`;

  // The bulk paste-and-register form is served by server.py at /submit, so only link to it on a local origin.
  const link = document.getElementById("locallink");
  if (link && looksLocalServer()) {
    const anchor = el("a", { href: "/submit", text: "Bulk submit: paste copied registry lines" });
    anchor.style.color = "inherit";
    link.append(anchor);
  }
}

async function reloadRegistry() {
  DATA.registryAll = await fetchRegistry();
  applyWindow();
}

async function onWebGPU() {
  const out = document.getElementById("webgpu-out");
  out.textContent = "running...";
  try {
    render(out, "webgpu", await runWebGPU(DATA, INPUTS), DATA, REFERENCE);
  } catch (e) {
    out.textContent = "";
    out.append(el("p", { class: "error", text: `webgpu crashed: ${e.message || e}` }));
    console.error(e);
  }
}

async function onWebGL2() {
  const out = document.getElementById("webgl2-out");
  out.textContent = "running...";
  try {
    render(out, "webgl2", await runWebGL2(DATA, INPUTS), DATA, REFERENCE);
  } catch (e) {
    out.textContent = "";
    out.append(el("p", { class: "error", text: `webgl2 crashed: ${e.message || e}` }));
    console.error(e);
  }
}

async function onWebGL1() {
  const out = document.getElementById("webgl1-out");
  out.textContent = "running...";
  try {
    render(out, "webgl1", await runWebGL1(DATA, INPUTS), DATA, REFERENCE);
  } catch (e) {
    out.textContent = "";
    out.append(el("p", { class: "error", text: `webgl1 crashed: ${e.message || e}` }));
    console.error(e);
  }
}

async function onCanvas() {
  const out = document.getElementById("canvas-out");
  out.textContent = "running...";
  try {
    renderCanvas(out, await runCanvas(), DATA);
  } catch (e) {
    out.textContent = "";
    out.append(el("p", { class: "error", text: `canvas crashed: ${e.message || e}` }));
    console.error(e);
  }
}

// Whether the run being shown is the first since the page loaded or a repeat. A browser keeps compiled shaders, so
// a repeat is far quicker than a first run, and the registry keeps the two timings apart. A first run here is not
// necessarily a cold one: the browser's shader cache can outlive the page.
let RUN = "first";
let RUNS = 0;

async function runAll() {
  RUNS += 1;
  RUN = RUNS === 1 ? "first" : "repeat";
  await onWebGL2();
  await onWebGL1();
  await onWebGPU();
  await onCanvas();
  if (RUNS === 1) recordVisit();
}

window.addEventListener("DOMContentLoaded", async () => {
  await load();
  await setUpWindowKind();
  document.getElementById("run-again").addEventListener("click", () => {
    JUST_ADDED.clear();
    return runAll();
  });
  await runAll();
});
