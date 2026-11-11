# Numeric GPU fingerprint

A pure HTML / JavaScript / YAML recreation of the `et-gpu-probe` technique (from `edge-toolkit/core`,
`libs/gpu-probe`). It puts a fixed set of floating-point expressions to the device's GPU, through WebGPU compute
shaders and through WebGL 2 and WebGL 1 fragment shaders, reads back the raw 32-bit results, and compares them with
the correctly rounded answer. The sequence of results is a signature of the GPU, driver and interface.

The point of the technique: the signature is read from the GPU's **numerical output**, not from any string the
browser reports (so name-masking defences such as a frozen user-agent or a blanked WebGPU adapter do not affect
it), and the WebGPU path is **not a canvas read** (so canvas defences such as a read prompt or per-pixel noise do
not apply to it).

## Files

- `index.html` -- the page and its controls.
- `probe.js` -- builds the shaders, runs the probes through both interfaces, reads back the bits, and
  computes the signature (fingerprint, deviations, traits, and the matched device class). No build step; its one
  runtime dependency, js-yaml, is loaded from a CDN (in `index.html`) to parse the YAML data files.
- `probes.yaml` -- the fixed probe data: the eight inputs (as exact f32 bit patterns), the sixteen probe
  expressions in WGSL and GLSL, the correctly rounded reference answers, and the choice table.
- `registry.yaml` -- the registry of measured devices, a YAML block list of entries. Each entry has an `interface`,
  a `label`, a `results` map keyed by probe name (so fields can be added without shifting values), and optional
  timings: `first_ms_min` / `first_ms_max` for the first run after the page loads and `repeat_ms_min` /
  `repeat_ms_max` for later runs on the same page, each the fastest and slowest seen. The two are kept apart
  because a browser keeps compiled shaders, so a repeat is far quicker; a first run is still not necessarily a
  cold one, since that cache can outlive the page. A new signature is appended as a block; a repeat submission of
  a known one is folded in, widening the range for its kind of run rather than adding a duplicate.
- `server.py` -- a standard-library server that serves these files, a paste-and-register form at `/submit`, and a
  POST `/register` endpoint that appends new signatures to `registry.yaml`. Runs over HTTP, or HTTPS with a cert.
- `analyse_registry.py` -- derives the figures reported for the technique from `registry.yaml` and `probes.yaml`.
- `mise.toml` -- tasks that run the server (`cert`, `serve`, `serve-lan`; both serve tasks use HTTPS) and the
  analysis (`analyse`).

## Running

The page fetches `probes.yaml` and loads JavaScript, so it must be served, not opened from a `file://` path. Both
serve tasks use HTTPS and generate a self-signed certificate on first run (through the `cert` task), because WebGPU
needs a secure context and browsers treat any non-localhost HTTP origin as insecure.

For this machine:

```bash
mise run serve
```

then open `https://localhost:8443/` and accept the self-signed certificate once. Both interfaces work.

To test from other devices (a phone, another laptop), or to check whether a browser behaves differently off
localhost (localhost is exempt from some defences, such as Brave's farbling), serve on the LAN:

```bash
mise run serve-lan
```

then open `https://<this-machine-ip>:8443/` and accept the certificate once in each browser. The cert also covers
the matching `nip.io` hostname (for example `https://10-130-52-183.nip.io:8443/`), a real registrable domain that
resolves to the LAN IP -- useful for testing defences that treat a bare IP differently from a domain.

## Submitting a new device

When a device is not in the registry, its result shows a suggested label, guessed from the GPU string, OS and
browser. Served by `server.py`, the result also shows a **Submit to registry** button that POSTs the signature to
`/register`, which validates it and appends a block to `registry.yaml`; the page then re-checks and shows the
device as recognised. The entry is always also shown as a compact JSON line to copy.

On a public static host (GitHub Pages, a Vercel preview) there is no `/register`, so the submit button is hidden
and only the copyable line is shown. To register such a result, copy the line and paste it into the `/submit` form
that a running `server.py` serves. That is how a device seen only in, say, Tor Browser on a phone can still reach
the registry.

## Adding a fingerprinting mechanism

A new mechanism is added in `probe.js` alone; `server.py` and `analyse_registry.py` take it as it comes.

The page chooses the interface name a result is filed under and the names and values of its results. The server
accepts any interface name, and any results whose names and values are plain tokens (letters, digits and a few
punctuation marks, with no spaces or quotes), so a result can be a hex word, a longer hash or a number. An entry
has a fixed shape -- `interface`, `label`, `window`, `results`, the timings, and `blocked` with a `reason` -- so
anything else a mechanism needs to record goes in `results`.

The analysis reads the mechanisms from the registry. An interface filed by mode (`<name>-split`,
`<name>-combined`) whose results are answers to the probes in `probes.yaml` gets the full numeric report; a probe
that some of its signatures do not answer is left out with a note. Every other interface is counted and its
signatures listed. A signature the script has no column name for is reported under a name of the form `?1`.

## Analysing the registry

```bash
mise run analyse
```

prints, for each numeric interface, how far each signature is from the correctly rounded answer, how many distinct
values each probe took, the number of probes on which each pair of signatures differs, and the smallest sets of
probes that tell every signature apart; then the probes on which the two interfaces disagree on each GPU measured
through both, a count for every interface in the registry, the signatures of the interfaces that are not numeric,
and the interfaces browsers were recorded as withholding.
`python3 analyse_registry.py --latex` adds the rows of the bit-pattern and distance tables. The script names each
signature by a short column name looked up from its registry label, so a signature added to the registry needs a
line in the script's `COLUMNS` table to get a name of its own.

## Reading the result

WebGPU, WebGL 2 and WebGL 1 are each run in two modes: **split**, with every expression compiled in a shader of its
own, and **combined**, with all of them compiled in one shader. A compiler can optimise an expression differently
when it sits beside others, so the two modes can return different bits on one device, and each mode's signature is filed
under its own registry interface name (`webgl2-split`, `webgl2-combined`, `webgl1-split`, `webgl1-combined`,
`webgpu-split`, `webgpu-combined`). The card opens with a banner when the two modes differ, naming the expressions
and both values.

WebGL 1 is the oldest path and the most restricted. It has no `floatBitsToUint`, so the result is rendered to a
32-bit float colour buffer and read straight back as a float (needing the `OES_texture_float` and
`WEBGL_color_buffer_float` extensions and highp float in the fragment shader; where any is missing the interface is
reported unavailable). It has no `tanh` builtin either, so that one probe is refused, leaving fourteen answers. It is
run because some browsers offer WebGL 1 while withholding WebGL 2 and WebGPU -- the Tor Browser at its default level
is the case of interest, where the question is whether a numeric result survives the read-back its fingerprinting
defence randomises. A signature that differs from one visit to the next (see the visit history) is that randomisation
at work.

Each mode shows a headline -- the matched device, or "not present" for a new one -- and a collapsible detail (open
automatically for a new device) with the reported GPU, how many probes were answered, how many deviated from the
correctly rounded value, the per-choice traits, the full fingerprint, and a per-probe table. A device already in
the registry has its detail folded away.

The page also recomputes the sixteen reference answers in JavaScript and checks them against `probes.yaml`, so a
drift between this port and the data is reported rather than hidden.

## Private windows

Some browsers change what they return in a private (incognito) window; Firefox, for one, adds noise to canvas reads
there. So the page works out whether the window is private and matches and submits its results separately: a
registry entry from a private window carries `window: private` and is never folded into a normal window's entry.
When a private window returns a signature the registry already has from a normal window, the card says so.

No browser reports that a window is private. The page infers it with the `detectincognitojs` library, loaded from a
CDN. The inference can be wrong on a browser newer than the library knows, and some content blockers stop the
library loading, so the **This is a private window** tick box at the top of the page has the last word. The status
line begins with PRIVATE WINDOW when results are being treated that way.

## Visit history

Reads of a canvas within one visit show only noise that changes on every read. A browser that keys its noise to
the session returns the same hash on every read within a visit and a different one in the next session. To see
that, the page keeps what each visit returned in the browser's own storage and says, for each result, whether it
has been the same on every visit. A canvas hash that differs from an earlier visit's is shown as randomised between
visits and offered as such, not as a signature. A browser or font update can also change the hash; **Forget earlier
visits** clears the history when that is the cause. A private window's storage is discarded when it closes, so its
history starts empty every time.

## Canvas fingerprints

A canvas defence can cover one way of making and reading a canvas and not another, so the canvas card takes two
fingerprints and files each under its own registry interface name:

- `canvas` -- the page's own scene (a gradient, blended circles, serif and italic text), drawn on a canvas the page
  made and read with `getImageData`.
- `canvas-browserleaks` -- the canvas test of browserleaks.com, made the way that site makes it: its text drawing on
  a 220 by 30 canvas created by the document of a sandboxed `about:blank` frame, read with `toDataURL`, and hashed
  with MD5 over the PNG. The card shows the MD5 in capitals, and it is the value browserleaks.com shows as its
  signature in the same browser, so the two can be compared directly.

Each is read more than once within the visit and compared with earlier visits, and each is offered to the registry
as a signature when it is stable and as randomised when it is not.

## Canvas noise check

The canvas card opens with a noise check. The page draws blocks of flat opaque colour, aligned to the pixel grid,
which every renderer draws exactly, and reads them back. Any value that differs from the colour drawn was altered
by the browser on the way out, so this shows noise being added to canvas reads within a single visit, with no
unprotected browser to compare against.

The check is run through every combination of three choices: the canvas made by the page or by a sandboxed frame,
its context left at the default or marked `willReadFrequently`, and the read made with `getImageData` or with
`toDataURL` (whose PNG the page decodes itself, so that reading the answer is not another canvas read). The table
gives, for each of the eight, the number of altered values and a hash of the pattern, which shows which ways of
reading a canvas a browser's defence covers. The visit history says whether the patterns are the same from one
visit to the next.
