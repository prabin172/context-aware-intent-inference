# Context-Aware Intent Inference

This repository contains the code pipeline for context-aware human intent inference using wearable motion sensing and first-person RGB-D visual context.

The project is organized around four main modules:

1. **Gesture Recognition (GR)**  
   Uses processed Xsens joint-angle data and ANVIL gesture annotations to predict early gesture intent from motion windows.

2. **2D Mapping / Proximity Extraction**  
   Uses already-synchronized RGB-D PNG frames, YOLO-based object detection, depth values, and Xsens segment position/orientation data to estimate human-object distances over time.

3. **Bayesian Fusion (BF)**  
   Combines gesture-recognition confidence scores with object proximity information using manually defined Bayesian conditional probability tables.

4. **End-to-End Transformer (E2E)**  
   Uses synchronized RGB-D frames and Xsens motion data to directly predict intent with a multimodal Transformer-based model.

## Current Pipeline Status

This repository is being reconstructed around the Ro-Man accepted context-aware intent inference pipeline.

The current active cleanup focus is the **2D Mapping / Proximity Extraction** module.

## Local Data Layout

Large data files are kept locally inside `data/` but are not pushed to GitHub.

Expected local structure:

```text
data/
├── Synced-color-depthPNG/
│   ├── sub2b/
│   ├── sub3/
│   ├── sub4-001/
│   └── ...
├── extracted_JointAngles/
│   ├── sub2b_ja.csv
│   ├── sub3_ja.csv
│   └── ...
├── xsens_segments/
│   ├── sub3_Segment Position.csv
│   ├── sub3_Segment Orientation - Euler.csv
│   └── ...
├── trials/
│   ├── sub3_trials.csv
│   └── ...
└── models/
    └── yolov8-ex-finetuned.pt
```

The repository assumes RGB-D PNGs are already synchronized. Raw Realsense `.bin` files, camera alignment code, and camera timestamp correction are outside the scope of this repository.

## 2D Mapping / Proximity Extraction

The 2D Mapping module uses:

```text
Synced RGB-D PNGs
YOLO object detector
Xsens pelvis position
Xsens pelvis orientation
ANVIL action annotations
Xsens/action annotation offsets
```

The final intended output for each session is:

```text
results/proximity/<session_id>/proximity.csv
```

This file will later be used by Bayesian Fusion.

## Step 1: YOLO + Depth Extraction

The script:

```text
src/proximity_mapping/processRawData.py
```

runs YOLO object detection on RGB frames and extracts object depth from the corresponding depth PNG using K-means clustering inside the YOLO bounding box.

Single-session example:

```bash
python src/proximity_mapping/processRawData.py \
  --rgbd_dir data/Synced-color-depthPNG/sub3 \
  --yolo_model data/models/yolov8-ex-finetuned.pt \
  --output_csv results/proximity/sub3/yolo_detections.csv \
  --conf 0.6 \
  --n_clusters 3
```

Batch script for all available sessions:

```bash
scripts/run_yolo_depth_all_sessions.sh
```

Output:

```text
results/proximity/<session_id>/yolo_detections.csv
```

At the current checkpoint, YOLO/depth detections have been generated for all 15 available session folders.

## Step 2: Xsens Pelvis Segment Files

For proximity mapping, only pelvis position and pelvis orientation are needed.

Each session should have:

```text
data/xsens_segments/<session_id>_Segment Position.csv
data/xsens_segments/<session_id>_Segment Orientation - Euler.csv
```

Each file should contain:

```text
Time (in ms)
Pelvis x
Pelvis y
Pelvis z
```

Current status:

```text
Position files: available for all 15 sessions
Orientation files: available for 14/15 sessions
Missing: sub2b_Segment Orientation - Euler.csv
```

The missing `sub2b` orientation file needs to be exported from Xsens and placed in:

```text
data/xsens_segments/sub2b_Segment Orientation - Euler.csv
```

## Step 3: Trial CSV Generation

Trial CSVs are generated from averaged ANVIL annotations and the Xsens/action annotation offset file.

The script:

```text
scripts/create_trials_from_anvil.py
```

uses:

```text
context-a-if/annotations/Averaged_Annotations/
context-a-if/GroundTruth-Annotations-Offset.csv
data/extracted_JointAngles/
```

For each annotated action, the trial window is defined as:

```text
2.5 seconds before gesture onset
1.0 second after gesture onset
```

Run:

```bash
python scripts/create_trials_from_anvil.py
```

Output:

```text
data/trials/<session_id>_trials.csv
```

Each generated trial CSV contains:

```text
Trial Number
Gesture
Object
Action Start Time (ANVIL s)
Action End Time (ANVIL s)
Offset (s)
Action Start Time (Xsens)
Action End Time (Xsens)
Start Time (Xsens)
End Time (Xsens)
```

## Next Steps

The remaining 2D Mapping scripts still need to be parameterized:

```text
src/proximity_mapping/extractObjectPositions.py
src/proximity_mapping/mapper.py
src/proximity_mapping/normalizeTime.py
```

After that, the full proximity pipeline will be:

```text
yolo_detections.csv
→ object_positions_by_trial.csv
→ mapped_distances_by_trial.csv
→ proximity.csv
```

A complete one-command shell script for the full proximity pipeline will be added after these scripts are cleaned and tested.
