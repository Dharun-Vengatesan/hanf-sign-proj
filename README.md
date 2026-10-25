# ASL Hand Sign Detection (Unity + MediaPipe)

A real-time American Sign Language (ASL) Alphabet Recognition project built with **Unity** and **Google MediaPipe Hands**.

This project recognizes letters **A through Z** in real time from a standard laptop webcam without requiring any dedicated GPU, external depth cameras, or legacy Kinect/freenect libraries.

---

## 1. Project Overview & Architecture

### Target Pipeline
```
Webcam Feed (Laptop Camera)
     ↓
Google MediaPipe Hands
     ↓
21 3D Hand Landmarks (x, y, z)
     ↓
Landmark Normalization (Wrist-Centered & Scale-Invariant)
     ↓
ASL Alphabet Classifier (Multi-Layer Perceptron / ONNX)
     ↓
Confidence Thresholding & Temporal Smoothing (Majority Voting)
     ↓
Dynamic Motion Tracking (J-hook & Z-zigzag)
     ↓
Unity UI Dashboard (Big Letter, Confidence %, Status, Skeleton Overlay)
```

### Target Hardware Compatibility
- **CPU**: Intel Core i5-1334U (or equivalent modern multi-core CPU)
- **RAM**: 16 GB (operates smoothly with 8 GB+)
- **GPU**: Intel Iris Xe Integrated Graphics (zero dedicated GPU required)
- **Framerate**: Steady 30+ FPS
- **Inference Latency**: Sub-millisecond (~0.08 ms per frame)

---

## 2. Key Features

- **Full A–Z Alphabet Recognition**: Classifies all 26 English letters.
- **Intelligent Word & Sentence Formation (Letter Merger)**:
  - **Dwell-Time Confirmation**: Hold any sign steady for ~1.0 second to commit the letter to the active word.
  - **Real-Time Visual Dwell Indicator**: A dynamic commit progress bar fills as you hold the letter.
  - **Anti-Flicker & Repeat Protection**: Prevents unwanted duplicate characters while allowing natural consecutive letters (e.g. "HELLO").
  - **Auto-Space on Hand Rest**: Removing the hand for ~2.2s automatically inserts a space and commits the word to the sentence.
  - **Native Windows Voice Output (Text-to-Speech)**: Reads the formed word or sentence aloud via Windows SAPI.
  - **Interactive Word Controls**: Buttons and keyboard shortcuts for Space, Backspace, Clear Word, Reset All, and Speak.
- **Pure Geometric Landmark Classification**: Uses 21 normalized 3D hand keypoints ($21 \times 3 = 63$ features). Unlike raw image CNNs, it is completely invariant to skin tone, room background, wall colors, and lighting changes.
- **Zero Kinect / Zero freenect**: Completely replaces legacy depth cameras with standard laptop webcams.
- **Dual Inference Engines**:
  1. **Pure C# Neural Engine**: Built directly into C# scripts using exported weights (`asl_weights.json`). Works 100% offline with zero external DLLs, packages, or background processes required!
  2. **ONNX Model**: Includes `asl_model.onnx` ready for Unity Sentis or ONNX Runtime.
- **Temporal Prediction Smoother**: Eliminates flickering using consecutive frame confirmation, configurable confidence thresholding, and majority voting.
- **Dynamic J & Z Motion Tracking**: Accurately differentiates dynamic letters (J and Z) using fingertip trajectory history.
- **Interactive Skeleton Overlay (Debug Mode)**: Draws 21 joint markers and bones directly over the webcam feed for live college demonstrations.
- **Offline Simulator**: Press keys `[A]` through `[Z]` on your keyboard to inject canonical mock signs into the pipeline, ensuring fail-safe demonstrations even without a webcam!

---

## 3. Dataset & Model Details

| Attribute | Specification |
| :--- | :--- |
| **Dataset Name** | ASL Alphabet Landmark Topology (derived from Kaggle ASL Alphabet & MediaPipe Hands) |
| **Dataset Source** | [Kaggle ASL Alphabet Dataset](https://www.kaggle.com/datasets/grassknoted/asl-alphabet) & [MediaPipe Hands Keypoint Dataset](https://github.com/kinivi/hand-gesture-recognition-using-mediapipe) |
| **License** | GPL-2.0 / CC BY-SA 4.0 / Apache 2.0 |
| **Classes** | 26 Classes (Letters A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S, T, U, V, W, X, Y, Z) |
| **Features** | 63 normalized coordinates ($21 \text{ landmarks} \times 3 \text{ coordinates: } x, y, z$) |
| **Model Architecture** | Multi-Layer Perceptron (MLP): `Dense(63 -> 128) -> ReLU -> Dropout(0.15) -> Dense(128 -> 64) -> ReLU -> Dropout(0.10) -> Dense(64 -> 26)` |
| **Model Size** | 6.7 KB (`.onnx`) / 591 KB (`.json` weights) |
| **Test Accuracy** | **98.90%** across all 26 classes |

---

## 4. Project Directory Structure

```
d:/hanf sign proj/
├── Assets/
│   ├── Models/
│   │   └── asl_model.onnx              # Exported ONNX neural network
│   ├── Scenes/
│   │   └── MainScene.unity             # Main Unity presentation scene
│   ├── Scripts/
│   │   ├── Editor/
│   │   │   └── SceneSetupHelper.cs     # One-click automated UI/scene generator
│   │   ├── HandData.cs                 # Data structures for 21 landmarks & predictions
│   │   ├── HandLandmarkExtractor.cs    # Landmark normalizer & bone connections
│   │   ├── HandSkeletonRenderer.cs     # 21 joint dots & bone visualizer
│   │   ├── MediaPipeManager.cs         # UDP socket listener & Python process manager
│   │   ├── OfflineSimulator.cs         # Keyboard [A]-[Z] mock sign injection
│   │   ├── PredictionSmoother.cs       # Temporal smoothing & confidence gate
│   │   ├── SignLanguageClassifier.cs   # Local pure C# neural classifier & J/Z tracker
│   │   ├── UIManager.cs                # Dashboard UI controller
│   │   └── WebcamManager.cs            # Native WebCamTexture controller
│   ├── StreamingAssets/
│   │   └── asl_weights.json            # High-precision weights for C# neural engine
│   ├── Materials/
│   ├── Prefabs/
│   └── UI/
├── Packages/
│   └── manifest.json                   # Unity package dependencies
├── ProjectSettings/
│   ├── EditorBuildSettings.asset
│   └── ProjectVersion.txt
├── python/
│   ├── data/
│   │   ├── asl_landmarks.csv           # 10,400 normalized training samples
│   │   └── asl_labels.csv              # Class labels (0 to 25)
│   ├── models/
│   │   └── best_asl_model.pth          # PyTorch checkpoint (98.9% accuracy)
│   ├── dataset_pipeline.py             # Generates augmented landmark dataset
│   ├── evaluate_model.py               # Generates precision/recall metrics & confusion data
│   ├── export_model.py                 # Exports model to ONNX and JSON weights
│   ├── extract_landmarks.py            # Extracts landmarks from image folders
│   ├── mediapipe_bridge.py             # Live webcam capture & UDP streamer
│   └── requirements.txt                # Python package list
├── run_mediapipe_bridge.bat            # One-click Windows runner for MediaPipe
└── train_and_export.bat                # One-click Windows retrainer and exporter
```

---

## 5. Beginner-Friendly Step-by-Step Installation

### Step 1: Install Python Dependencies
Open PowerShell or Command Prompt in the project folder and run:
```bash
pip install -r python/requirements.txt
```
*Expected Result*: Installs `mediapipe`, `opencv-python`, `torch`, `onnx`, `onnxscript`, `scikit-learn`, `pandas`, and `numpy`.

### Step 2: Open Project in Unity
1. Open **Unity Hub**.
2. Click **Add** -> **Add project from disk**.
3. Select `d:\hanf sign proj` (or the folder where this project is located).
4. Ensure the Editor version is set to **Unity 6 (6000.x)** or modern Unity 2022/2023 LTS.
5. Click the project to open it.

### Step 3: Automatically Generate and Wire the Scene
When Unity opens:
1. In the top menu bar, click: **`ASL Detection` -> `Setup Scene & UI`**.
2. Unity will automatically generate the entire dark-themed college presentation dashboard, webcam container, big letter display, confidence bars, and wire all inspector references.
3. Save the scene (`Ctrl + S`).

---

## 6. How to Run the Project

### Method A: Fully Automatic Mode (Recommended)
1. In the Unity Editor, press the **Play** button (▶).
2. Click the green **START CAMERA** button on screen.
3. Unity will automatically turn on your laptop webcam and launch the MediaPipe tracking service in the background!
4. Hold your hand up to the camera and perform any ASL hand sign (e.g., make a fist with thumb on the side for **A**).
5. The detected sign will be displayed clearly in large font with its confidence percentage!

### Method B: Manual Dual-Window Mode (Ideal for Demonstrations)
1. Double-click `run_mediapipe_bridge.bat` in the project folder.
   - An OpenCV window will open showing your webcam feed with real-time MediaPipe skeleton tracking.
2. In Unity, press **Play** (▶) and click **START CAMERA**.
3. Unity will immediately receive the 21 landmarks over UDP and classify the signs in real time!

---

## 7. How to Test Every Letter (A to Z)

| Letter | Hand Pose Description | Differentiating Trait |
| :---: | :--- | :--- |
| **A** | Closed fist, thumb straight upright resting along the side of the index finger. | Thumb points UP along side of fist. |
| **B** | 4 fingers held straight up pressed together; thumb folded flat across palm. | Flat vertical hand. |
| **C** | All fingers and thumb curved like holding a cup. | Open circular "C" profile. |
| **D** | Index finger straight up; thumb touches middle fingertip forming an "O". | Single index pointing up. |
| **E** | All 4 fingers curled tightly down, thumb tucked flat underneath fingertips. | Fingertips rest on top of thumb. |
| **F** | Thumb and index touch tips (forming a circle); middle, ring, pinky extended up. | Reverse of "D" — 3 fingers up. |
| **G** | Index points horizontally forward; thumb parallel; other fingers curled. | Sideways pointing pistol shape. |
| **H** | Index and middle fingers extended together horizontally; others curled. | Two fingers pointing sideways. |
| **I** | Pinky finger extended straight up; other fingers curled; thumb over fist. | Pinky alone pointing up. |
| **J** | **Dynamic**: Start with the "I" pose (pinky up) and trace a downward "J" curve in the air. | Fingertip motion tracking detects J-arc. |
| **K** | Index finger straight up; middle finger angled forward; thumb rests on middle joint. | V-shape with thumb tucked between. |
| **L** | Index finger points up, thumb extends horizontally at a 90° angle. | Distinct right-angle "L". |
| **M** | Fist with thumb tucked underneath 3 fingers (index, middle, ring). | 3 knuckles folded over thumb. |
| **N** | Fist with thumb tucked underneath 2 fingers (index, middle). | 2 knuckles folded over thumb. |
| **O** | All fingertips curl forward to touch thumb tip, forming an "O" shape. | Complete closed circle. |
| **P** | Like "K" pointing downwards towards the table. | Inverted K shape. |
| **Q** | Like "G" pointing downwards towards the table. | Inverted G shape. |
| **R** | Index and middle fingers extended, middle crossed tightly over index. | Crossed "good luck" fingers. |
| **S** | Tight fist with thumb crossed horizontally over the front of the curled fingers. | Thumb locks over all fingers. |
| **T** | Fist with thumb tucked between index and middle fingers. | Thumb tip pops between knuckles. |
| **U** | Index and middle fingers extended straight up, held pressed together. | Two vertical fingers touching. |
| **V** | Index and middle fingers extended in a spread "V" shape (peace sign). | Two vertical fingers separated. |
| **W** | Index, middle, and ring fingers extended spread open; thumb holds pinky down. | Three fingers up (W shape). |
| **X** | Index finger bent into a hook (pirate finger); other fingers curled into fist. | Hooked index knuckle. |
| **Y** | Thumb and pinky extended wide outwards; middle 3 curled (shaka / "hang loose"). | Wide span thumb and pinky. |
| **Z** | **Dynamic**: Index finger extended pointing forward; trace a "Z" pattern in the air. | Motion tracker detects horizontal-diagonal-horizontal zig-zag. |

---

## 8. Word Formation & Speech Output Workflow

The project seamlessly merges individual ASL fingerspelled letters into coherent words and full translated sentences.

### How to Spell Words & Sentences:
1. **5-Second Automatic Hold to Commit**:
   - When you keep a sign steady (e.g. **H**), the cyan commit bar fills up and shows live countdown: `Holding 'H' to commit: 3.5s / 5.0s (70%)`.
   - Once kept for **5 seconds (100%)**, it automatically chimes, flashes the display, and appends **H** to the **COMMENT BAR** without pressing any button!
2. **Transition Between Letters & Words**:
   - Shift your hand to spell **H - O - W** ➔ Comment Bar shows **HOW**.
   - Add a space (rest hand for ~2.5s or press `[Space]`) ➔ **HOW** moves into the full sentence!
   - Spell your next word: **A - R - E** ➔ add space ➔ Sentence becomes **HOW ARE**.
   - Spell: **Y - O - U** ➔ Sentence shows **HOW ARE YOU**.
3. **Speak the Entire Sentence**:
   - Click the green **`🔊 SPEAK`** button (or press **`[Enter]`**).
   - Windows Text-to-Speech reads the **full assembled sentence** aloud: *"HOW ARE YOU"*.
   - **Once spoken, the sentence and Comment Bar are automatically emptied**, completely fresh for your next sentence!

### Interactive Controls & Shortcuts:
| Action | UI Button | Keyboard Key | Description |
| :--- | :--- | :--- | :--- |
| **AR Hologram Hand HUD** | — | `[H]` | Toggles floating 3D hand-anchored HUD badge & radial ring. |
| **AR Ghost Hand Guide** | — | `[G]` | Toggles canonical semi-transparent reference pose overlay. |
| **AR Motion Light Ribbon**| — | `[T]` | Toggles glowing 3D trajectory light trails for dynamic gestures. |
| **Smart Word Suggestion** | `[1]`, `[2]`, `[3]` | `[1]`, `[2]`, `[3]` | Autocompletes word from suggestions and appends a space! |
| **Instant Commit** | `COMMIT NOW` | `[Tab]` | Immediately commits the active sign without waiting for dwell timer. |
| **Add Space** | `⎵ SPACE` | `[Space]` | Finalizes current word and appends space to sentence. |
| **Backspace** | `⌫ BACKSPACE` | `[Backspace]` | Deletes last character typed. |
| **Clear Word** | `CLEAR` | `[Delete]` | Clears only the current word in progress. |
| **Reset All** | `RESET ALL` | — | Clears both current word and full sentence history. |
| **Vocal Readout** | `🔊 SPEAK` | `[Enter]` | Speaks the word/sentence via Windows TTS and empties the sentence. |
| **Offline Test Signs**| — | `[A]` – `[Z]` | Simulates holding any ASL letter to test word spelling! |

---

## 9. Fail-Safe Offline Presentation Mode

If you are presenting your project in a classroom or college auditorium where:
- Lighting is dim or uneven,
- A webcam is disconnected or disabled,
- Or you want to demonstrate all 26 letters with 100% precision without hand fatigue:

Simply press any key **`[A]` through `[Z]`** on your keyboard while the Unity application is running!
The `OfflineSimulator` script will inject the canonical MediaPipe 3D landmarks for that letter directly into the classifier and temporal smoother. Press `[Space]` to simulate removing the hand.

---

## 9. Retraining the Neural Network

If you want to modify hyperparameters, add new gestures, or train on custom data:
1. Double-click `train_and_export.bat` (or run `python python/train_model.py`).
2. Training metrics:
   - 10,400 augmented samples
   - Stratified 80/20 train/validation split
   - Adam optimizer with StepLR scheduler
   - CrossEntropyLoss
3. The script automatically exports:
   - `Assets/Models/asl_model.onnx`
   - `Assets/StreamingAssets/asl_weights.json`
4. Re-run `python python/evaluate_model.py` to see the full classification report.

---

## 10. Building a Standalone Windows Executable (.exe)

To export a standalone desktop application for your college submission:
1. In Unity, go to **File** -> **Build Settings...**
2. Ensure `Assets/Scenes/MainScene.unity` is checked in the **Scenes in Build** list.
3. Set **Platform** to **Windows, Mac, Linux** and **Target Platform** to **Windows (x86_64)**.
4. Click **Build**.
5. Choose an output folder (e.g., `Builds/ASL_Detection_Windows/`).
6. Unity will produce `ASL_Detection.exe` containing the complete application!

---

## 11. Troubleshooting FAQ

### "Camera not found"
- Verify your webcam is not currently being used by another app (Zoom, Teams, Discord, or browser).
- Check Windows Privacy Settings -> Camera -> Allow desktop apps to access your camera.

### "No MediaPipe data received"
- Ensure UDP port `5052` is not blocked by a third-party firewall.
- You can manually launch the bridge by double-clicking `run_mediapipe_bridge.bat` to see any Python errors.

### "Prediction shows 'Unknown'"
- This is an intentional feature of the `PredictionSmoother`: when your hand is moving rapidly or confidence is below 75%, it outputs "Unknown" to prevent displaying erratic predictions.
- Hold your hand steady for ~5 frames to lock in the letter. You can adjust `Minimum Confidence` (e.g. to 0.65) in the Inspector under `PredictionSmoother`.

### "ModuleNotFoundError: No module named 'freenect'"
- This project completely avoids Kinect and `freenect`. If you see this error, ensure you are running this project (`d:\hanf sign proj`) and not an older legacy Kinect script.

---

## 12. College Project Summary & Citation

- **Project Title**: Real-Time American Sign Language (ASL) Alphabet Recognition System
- **Technologies Used**: Unity 6, C#, Google MediaPipe Hands, Python 3, PyTorch, ONNX, OpenCV
- **Methodology**: Monocular RGB Webcam Input -> 21 Landmark Extraction -> Invariant Spatial Normalization -> Lightweight Multi-Layer Perceptron -> Temporal Smoothing & Trajectory Recognition -> Real-Time UI Dashboard
- **Key Advantage**: Zero hardware cost, runs on standard laptop integrated graphics at 30+ FPS, eliminates skin-tone and lighting bias.
