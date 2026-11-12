# Original research

This directory holds the original research produced in this repository, as distinct from the reproductions of
others' published work under `reproductions/`.

## Contents

- [`gpu-fingerprint/`](gpu-fingerprint/) -- numerically fingerprinting GPUs in the browser: a fixed set of
  floating-point expressions is put to the device through WebGPU and WebGL 2, the raw bits are read back and
  compared with the correctly rounded answer, and the sequence of results is a signature of the GPU, driver and
  interface. Ships a device-signature registry and a local collection server. See its own `README.md`.
