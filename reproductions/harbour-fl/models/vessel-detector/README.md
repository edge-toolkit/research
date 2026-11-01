# Camera vessel detector

This directory supports standalone inspection of a separately trained YOLOv8 ONNX detector. It can preview selected
vessel crops, but it is independent of Harbour-FL's pretrained classifier, which uses full-image cached embeddings.
Detector weights are optional and do not participate in federated training or aggregation.

## Prepare the model

1. Train and evaluate the detector in your separate detector-training project.
2. Export the trained YOLOv8 weights in that environment. For example, with the Ultralytics CLI:

   ```bash
   yolo export model=best.pt format=onnx imgsz=640 batch=1 dynamic=False half=False nms=False opset=17
   ```

3. Copy the exported file into this directory as `model.onnx`.
4. Copy `detector.example.json` to `detector.json` and set `num_classes` and `class_ids` to match your detector.

Keep held-out evaluation frames separate from detector training when measuring end-to-end generalisation.

`class_ids` selects detector classes representing vessels. A one-class vessel detector uses `[0]`. These indices are
independent of the vessel-type labels used by the FL classifier. The supported export has one FP32 RGB input shaped
`[1, 3, input_size, input_size]` and raw predictions shaped `[1, 4 + num_classes, anchors]`.

Use trusted model artifacts. Optional `sha256` in `detector.json` pins the expected file digest. Weight files are
Git-ignored; distribute them through your chosen model-artifact storage. Model exports and configuration are immutable
for a utility process: start a new process after changing them.

## Enable and validate

Install the optional runtime and run the utility from the project directory:

```bash
.venv/bin/python -m pip install -e '.[detection]'
HARBOUR_FL_DETECTOR_ENABLED=true .venv/bin/python -m harbour_fl.vision check
HARBOUR_FL_DETECTOR_ENABLED=true .venv/bin/python -m harbour_fl.vision preview --image /path/to/camera-image.jpg
```

The preview writes `crop.png` and `detection.json` under ignored `results/detection-preview/`. Keep detection disabled
when preparing features or running the classification application. These tools do not alter prepared features, the
classification model or saved training checkpoints. `HARBOUR_FL_DETECTOR_CONFIG` can select a different configuration.

## Crop behaviour

| Setting        | Behaviour                                                                                      |
| -------------- | ---------------------------------------------------------------------------------------------- |
| `confidence`   | Minimum score for a selected vessel class.                                                     |
| `iou`          | Overlap threshold for suppressing duplicate vessel detections.                                 |
| `margin`       | Extra context around the box, as a fraction of its width and height; clipped to the image.     |
| `selection`    | `largest` selects the largest qualifying vessel; `confidence` selects the highest-scoring one. |
| `no_detection` | `full-image` retains the frame, `zero` masks the image branch, or `error` stops processing.    |
| `cache`        | Reuse detected boxes for identical image pixels, model weights and preprocessing settings.     |

Defaults are in `detector.example.json`. Multiple detections produce one selected crop per supplied image.
The default missing-detection policy retains the full image; `zero` produces a blank preview, and `error` rejects
the image. These policies apply to the standalone utility and do not change classification training metrics.

Cached JSON records contain boxes and scores. Source images are unchanged. Native cache files live in
`results/detector-cache/`; containers use `/tmp/harbour-fl-detector-cache`. `HARBOUR_FL_DETECTOR_CACHE` can override the
runtime cache location. An unwritable cache falls back to uncached inference with a warning.

For timing comparisons, use a fresh cache directory for cold-cache runs or pre-warm both workloads for warm-cache runs.

Preview outputs record the detector identity and settings for inspection. Harbour-FL's image-noise benchmarks use
features extracted from noisy full images, independently of this detector. Evaluate detector robustness with noisy raw
frames in the separate detector project.

The runtime uses the [ONNX Runtime Python API](https://onnxruntime.ai/docs/api/python/api_summary.html).
See the [Ultralytics export guide](https://docs.ultralytics.com/modes/export/) for detector-export options.
