"""
ASL Alphabet Landmark Dataset Pipeline
Generates / extracts normalized 21 MediaPipe hand landmarks (63 features)
for all 26 American Sign Language letters (A to Z).
"""

import os
import math
import numpy as np
import pandas as pd
import cv2
import mediapipe as mp

# 26 Letters of ASL Alphabet
CLASSES = [chr(i) for i in range(ord('A'), ord('Z') + 1)]
LABEL_TO_IDX = {letter: idx for idx, letter in enumerate(CLASSES)}
IDX_TO_LABEL = {idx: letter for idx, letter in enumerate(CLASSES)}

# Canonical MediaPipe Hand Landmark indices:
# 0: Wrist
# 1-4: Thumb (CMC, MCP, IP, TIP)
# 5-8: Index (MCP, PIP, DIP, TIP)
# 9-12: Middle (MCP, PIP, DIP, TIP)
# 13-16: Ring (MCP, PIP, DIP, TIP)
# 17-20: Pinky (MCP, PIP, DIP, TIP)

def get_base_hand_skeleton():
    """Returns canonical reference 3D positions for an open flat palm."""
    # Base neutral open palm (facing camera)
    landmarks = np.zeros((21, 3), dtype=np.float32)
    # 0 Wrist
    landmarks[0] = [0.0, 0.0, 0.0]
    
    # Thumb: points out to the left for right hand
    landmarks[1] = [-0.15, 0.12, -0.02] # CMC
    landmarks[2] = [-0.28, 0.22, -0.04] # MCP
    landmarks[3] = [-0.36, 0.32, -0.05] # IP
    landmarks[4] = [-0.42, 0.40, -0.06] # TIP
    
    # Index finger
    landmarks[5] = [-0.15, 0.40, 0.0]   # MCP
    landmarks[6] = [-0.16, 0.58, -0.02] # PIP
    landmarks[7] = [-0.16, 0.72, -0.03] # DIP
    landmarks[8] = [-0.16, 0.85, -0.04] # TIP
    
    # Middle finger
    landmarks[9] = [0.0, 0.42, 0.0]     # MCP
    landmarks[10] = [0.0, 0.62, -0.02]  # PIP
    landmarks[11] = [0.0, 0.78, -0.03]  # DIP
    landmarks[12] = [0.0, 0.92, -0.04]  # TIP
    
    # Ring finger
    landmarks[13] = [0.14, 0.39, 0.0]   # MCP
    landmarks[14] = [0.15, 0.57, -0.02] # PIP
    landmarks[15] = [0.15, 0.71, -0.03] # DIP
    landmarks[16] = [0.15, 0.83, -0.04] # TIP
    
    # Pinky finger
    landmarks[17] = [0.26, 0.32, 0.0]   # MCP
    landmarks[18] = [0.30, 0.46, -0.02] # PIP
    landmarks[19] = [0.32, 0.58, -0.03] # DIP
    landmarks[20] = [0.34, 0.68, -0.04] # TIP
    
    return landmarks

def curl_finger(lm, base_idx, curl_amount):
    """
    Curls a finger towards the palm.
    curl_amount: 0.0 (straight extended) to 1.0 (fully curled into palm).
    """
    mcp = lm[base_idx]
    # Curl PIP, DIP, TIP
    for i, step in enumerate([1, 2, 3]):
        idx = base_idx + step
        orig = lm[idx]
        # When curling, finger bends towards wrist and slightly in z
        target_y = mcp[1] - (step * 0.06)
        target_z = mcp[2] + (step * 0.08)
        target_x = mcp[0] * 0.8
        target = np.array([target_x, target_y, target_z], dtype=np.float32)
        lm[idx] = orig * (1.0 - curl_amount) + target * curl_amount

def position_thumb(lm, mode):
    """
    Configures thumb position according to ASL poses:
    mode:
      'out': open pointing outward (L, etc.)
      'across_palm': tucked across lower palm (B, etc.)
      'fist_side': against side of curled index (A)
      'fist_over': resting across curled fingers (S)
      'between_index_middle': thrust between index and middle (T)
      'between_middle_ring': thrust between middle and ring (N)
      'between_ring_pinky': thrust between ring and pinky (M)
      'index_touch': touching index tip to form circle (O, F)
      'under_fingers': tucked underneath curled fingers (E)
    """
    if mode == 'out':
        lm[1] = [-0.18, 0.12, -0.02]
        lm[2] = [-0.32, 0.22, -0.04]
        lm[3] = [-0.44, 0.32, -0.05]
        lm[4] = [-0.52, 0.42, -0.06]
    elif mode == 'across_palm':
        lm[1] = [-0.10, 0.12, 0.02]
        lm[2] = [-0.05, 0.20, 0.05]
        lm[3] = [0.02, 0.24, 0.06]
        lm[4] = [0.08, 0.26, 0.07]
    elif mode == 'fist_side':
        # ASL 'A': thumb straight upright along the side of index MCP/PIP
        lm[1] = [-0.12, 0.15, 0.02]
        lm[2] = [-0.16, 0.28, 0.03]
        lm[3] = [-0.16, 0.40, 0.03]
        lm[4] = [-0.16, 0.50, 0.02]
    elif mode == 'fist_over':
        # ASL 'S': thumb crosses horizontally over middle/index joints
        lm[1] = [-0.10, 0.15, 0.04]
        lm[2] = [-0.05, 0.26, 0.08]
        lm[3] = [0.02, 0.30, 0.09]
        lm[4] = [0.08, 0.31, 0.09]
    elif mode == 'between_index_middle':
        # ASL 'T': thumb tip sticks up between index and middle knuckles
        lm[1] = [-0.10, 0.15, 0.04]
        lm[2] = [-0.08, 0.25, 0.07]
        lm[3] = [-0.08, 0.38, 0.09]
        lm[4] = [-0.08, 0.46, 0.10]
    elif mode == 'between_middle_ring':
        # ASL 'N': thumb rests between middle and ring
        lm[1] = [-0.08, 0.15, 0.04]
        lm[2] = [0.0, 0.25, 0.07]
        lm[3] = [0.06, 0.34, 0.09]
        lm[4] = [0.07, 0.40, 0.10]
    elif mode == 'between_ring_pinky':
        # ASL 'M': thumb rests between ring and pinky
        lm[1] = [-0.06, 0.15, 0.04]
        lm[2] = [0.05, 0.25, 0.07]
        lm[3] = [0.15, 0.33, 0.09]
        lm[4] = [0.18, 0.38, 0.10]
    elif mode == 'index_touch':
        # Form 'O' or 'F' circle touching index tip
        lm[1] = [-0.12, 0.15, 0.04]
        lm[2] = [-0.15, 0.30, 0.08]
        lm[3] = [-0.14, 0.42, 0.10]
        lm[4] = [-0.12, 0.50, 0.08]
    elif mode == 'under_fingers':
        # ASL 'E': thumb tucked low under curled fingertip row
        lm[1] = [-0.10, 0.12, 0.03]
        lm[2] = [-0.04, 0.18, 0.06]
        lm[3] = [0.02, 0.20, 0.07]
        lm[4] = [0.08, 0.21, 0.07]

def generate_canonical_sign(letter, rng=None):
    """
    Generates exact canonical 21-landmark ASL hand pose for a given letter A-Z.
    If rng is provided, incorporates realistic natural human anatomical and pose variations.
    """
    lm = get_base_hand_skeleton()
    letter = letter.upper()
    
    # Helper functions for natural variation
    def r_curl(min_c, max_c):
        return float(rng.uniform(min_c, max_c)) if rng is not None else float((min_c + max_c) * 0.5)
        
    def r_val(val, spread=0.03):
        return float(val + rng.uniform(-spread, spread)) if rng is not None else float(val)
    
    if letter == 'A':
        # Fist with thumb resting against index side pointing upward
        c = r_curl(0.80, 0.98)
        curl_finger(lm, 5, c)
        curl_finger(lm, 9, c)
        curl_finger(lm, 13, c)
        curl_finger(lm, 17, c)
        position_thumb(lm, 'fist_side')
        lm[4, 1] = r_val(0.50, 0.05)
        lm[4, 0] = r_val(-0.16, 0.03)
        
    elif letter == 'B':
        # 4 fingers straight up together, thumb tucked across palm
        c = r_curl(0.0, 0.08)
        curl_finger(lm, 5, c)
        curl_finger(lm, 9, c)
        curl_finger(lm, 13, c)
        curl_finger(lm, 17, c)
        position_thumb(lm, 'across_palm')
        lm[5:9, 0] -= r_val(0.02, 0.01)
        lm[13:17, 0] -= r_val(0.02, 0.01)
        lm[17:21, 0] -= r_val(0.04, 0.01)
        
    elif letter == 'C':
        # Curved C shape like holding a cup
        c = r_curl(0.38, 0.52)
        for base in [5, 9, 13, 17]:
            curl_finger(lm, base, c)
            lm[base+1:base+4, 2] += r_val(0.12, 0.03)
        lm[1] = [r_val(-0.15, 0.02), r_val(0.15, 0.02), r_val(0.05, 0.02)]
        lm[2] = [r_val(-0.22, 0.02), r_val(0.26, 0.02), r_val(0.10, 0.02)]
        lm[3] = [r_val(-0.20, 0.02), r_val(0.36, 0.02), r_val(0.12, 0.02)]
        lm[4] = [r_val(-0.15, 0.02), r_val(0.44, 0.03), r_val(0.10, 0.02)]
        
    elif letter == 'D':
        # Index straight up, middle/ring/pinky touch thumb forming an 'O'
        curl_finger(lm, 5, r_curl(0.0, 0.06))
        c_curled = r_curl(0.60, 0.85)
        curl_finger(lm, 9, c_curled)
        curl_finger(lm, 13, c_curled)
        curl_finger(lm, 17, c_curled + 0.05)
        position_thumb(lm, 'index_touch')
        lm[4] = [r_val(-0.02, 0.02), r_val(0.45, 0.03), r_val(0.08, 0.02)]
        
    elif letter == 'E':
        # All 4 fingers curled tightly, thumb tucked flat underneath fingertips
        c = r_curl(0.82, 0.95)
        curl_finger(lm, 5, c)
        curl_finger(lm, 9, c)
        curl_finger(lm, 13, c)
        curl_finger(lm, 17, c)
        position_thumb(lm, 'under_fingers')
        lm[4, 1] = r_val(0.21, 0.03)
        
    elif letter == 'F':
        # Index touches thumb (circle), middle/ring/pinky extended up and fanned
        curl_finger(lm, 5, r_curl(0.50, 0.60))
        curl_finger(lm, 9, r_curl(0.0, 0.08))
        curl_finger(lm, 13, r_curl(0.0, 0.08))
        curl_finger(lm, 17, r_curl(0.0, 0.08))
        lm[8] = [r_val(-0.12, 0.02), r_val(0.48, 0.03), r_val(0.08, 0.02)]
        position_thumb(lm, 'index_touch')
        lm[4] = lm[8].copy()
        
    elif letter == 'G':
        # Index pointing horizontally forward, thumb parallel to it, other fingers curled
        c = r_curl(0.90, 0.98)
        curl_finger(lm, 9, c)
        curl_finger(lm, 13, c)
        curl_finger(lm, 17, c)
        lm[5] = [r_val(-0.10, 0.02), r_val(0.35, 0.02), 0.0]
        lm[6] = [r_val(-0.02, 0.02), r_val(0.42, 0.02), 0.02]
        lm[7] = [r_val(0.08, 0.02), r_val(0.45, 0.02), 0.03]
        lm[8] = [r_val(0.18, 0.02), r_val(0.47, 0.02), 0.04]
        lm[1] = [r_val(-0.10, 0.02), r_val(0.20, 0.02), 0.02]
        lm[2] = [r_val(-0.05, 0.02), r_val(0.28, 0.02), 0.04]
        lm[3] = [r_val(0.04, 0.02), r_val(0.34, 0.02), 0.05]
        lm[4] = [r_val(0.14, 0.02), r_val(0.37, 0.02), 0.06]
        
    elif letter == 'H':
        # Index and middle extended horizontally together, other fingers curled
        c = r_curl(0.90, 0.98)
        curl_finger(lm, 13, c)
        curl_finger(lm, 17, c)
        lm[5:9] = np.array([[-0.10, 0.38, 0.0], [-0.02, 0.46, 0.02], [0.08, 0.49, 0.03], [0.18, 0.51, 0.04]])
        lm[9:13] = np.array([[-0.10, 0.32, 0.0], [-0.02, 0.40, 0.02], [0.08, 0.43, 0.03], [0.18, 0.45, 0.04]])
        if rng is not None:
            lm[5:13] += rng.normal(0, 0.015, lm[5:13].shape)
        position_thumb(lm, 'across_palm')
        
    elif letter == 'I':
        # Pinky extended straight up, index/middle/ring curled, thumb over fingers
        c = r_curl(0.88, 0.98)
        curl_finger(lm, 5, c)
        curl_finger(lm, 9, c)
        curl_finger(lm, 13, c)
        curl_finger(lm, 17, r_curl(0.0, 0.06))
        position_thumb(lm, 'fist_over')
        
    elif letter == 'J':
        # ASL J: Pinky extended (like I), but traces a hook trajectory
        c = r_curl(0.88, 0.98)
        curl_finger(lm, 5, c)
        curl_finger(lm, 9, c)
        curl_finger(lm, 13, c)
        curl_finger(lm, 17, r_curl(0.02, 0.10))
        lm[18:21, 0] += r_val(0.05, 0.02)
        position_thumb(lm, 'fist_over')
        
    elif letter == 'K':
        # Index straight up, middle angled forward/up, thumb touches middle joint
        curl_finger(lm, 5, r_curl(0.0, 0.06))
        c = r_curl(0.88, 0.98)
        curl_finger(lm, 13, c)
        curl_finger(lm, 17, c)
        lm[9] = [0.0, 0.42, 0.0]
        lm[10] = [r_val(0.02, 0.02), r_val(0.56, 0.03), r_val(0.10, 0.03)]
        lm[11] = [r_val(0.04, 0.02), r_val(0.70, 0.03), r_val(0.16, 0.03)]
        lm[12] = [r_val(0.06, 0.02), r_val(0.82, 0.03), r_val(0.20, 0.03)]
        lm[1] = [-0.10, 0.18, 0.04]
        lm[2] = [-0.05, 0.30, 0.08]
        lm[3] = [0.0, 0.42, 0.12]
        lm[4] = [r_val(0.02, 0.02), r_val(0.54, 0.03), 0.12]
        
    elif letter == 'L':
        # Index straight up, thumb pointing horizontally out (90 degree L)
        curl_finger(lm, 5, r_curl(0.0, 0.06))
        c = r_curl(0.88, 0.98)
        curl_finger(lm, 9, c)
        curl_finger(lm, 13, c)
        curl_finger(lm, 17, c)
        position_thumb(lm, 'out')
        lm[4, 0] = r_val(-0.52, 0.05)
        lm[4, 1] = r_val(0.42, 0.04)
        
    elif letter == 'M':
        # Fist with thumb tucked under 3 fingers (index, middle, ring)
        c = r_curl(0.85, 0.95)
        curl_finger(lm, 5, c)
        curl_finger(lm, 9, c)
        curl_finger(lm, 13, c)
        curl_finger(lm, 17, r_curl(0.90, 0.98))
        position_thumb(lm, 'between_ring_pinky')
        lm[4, 0] = r_val(0.18, 0.03)
        lm[4, 1] = r_val(0.38, 0.03)
        
    elif letter == 'N':
        # Fist with thumb tucked under 2 fingers (index, middle)
        c = r_curl(0.85, 0.95)
        curl_finger(lm, 5, c)
        curl_finger(lm, 9, c)
        curl_finger(lm, 13, r_curl(0.90, 0.98))
        curl_finger(lm, 17, r_curl(0.90, 0.98))
        position_thumb(lm, 'between_middle_ring')
        lm[4, 0] = r_val(0.07, 0.03)
        lm[4, 1] = r_val(0.40, 0.03)
        
    elif letter == 'O':
        # All fingertips touch thumb tip forming circular 'O'
        c = r_curl(0.50, 0.60)
        for base in [5, 9, 13, 17]:
            curl_finger(lm, base, c)
            lm[base+3] = [r_val(-0.06, 0.02), r_val(0.42 + (base - 5)*0.01, 0.03), r_val(0.10, 0.02)]
        position_thumb(lm, 'index_touch')
        lm[4] = [-0.06, 0.42, 0.10]
        
    elif letter == 'P':
        # Like K pointing downwards
        curl_finger(lm, 5, r_curl(0.0, 0.06))
        lm[5:9, 1] = 0.40 - np.array([0.0, 0.18, 0.32, 0.45]) # pointing down
        c = r_curl(0.88, 0.98)
        curl_finger(lm, 13, c)
        curl_finger(lm, 17, c)
        lm[9:13, 1] = 0.40 - np.array([0.0, 0.16, 0.28, 0.40])
        lm[9:13, 2] += 0.12
        position_thumb(lm, 'across_palm')
        
    elif letter == 'Q':
        # Like G pointing downwards
        c = r_curl(0.88, 0.98)
        curl_finger(lm, 9, c)
        curl_finger(lm, 13, c)
        curl_finger(lm, 17, c)
        lm[5:9, 1] = 0.35 - np.array([0.0, 0.15, 0.28, 0.40])
        lm[1:5, 1] = 0.20 - np.array([0.0, 0.12, 0.24, 0.35])
        
    elif letter == 'R':
        # Index and middle extended, middle crossed over index, others curled
        c = r_curl(0.88, 0.98)
        curl_finger(lm, 13, c)
        curl_finger(lm, 17, c)
        curl_finger(lm, 5, r_curl(0.0, 0.06))
        curl_finger(lm, 9, r_curl(0.0, 0.06))
        lm[9:13, 0] = lm[5:9, 0] - r_val(0.05, 0.02)
        lm[9:13, 2] = lm[5:9, 2] + r_val(0.04, 0.02)
        position_thumb(lm, 'fist_over')
        
    elif letter == 'S':
        # Fist with thumb crossed over the front of all curled fingers
        c = r_curl(0.88, 0.98)
        curl_finger(lm, 5, c)
        curl_finger(lm, 9, c)
        curl_finger(lm, 13, c)
        curl_finger(lm, 17, c)
        position_thumb(lm, 'fist_over')
        lm[4, 0] = r_val(0.08, 0.03)
        lm[4, 1] = r_val(0.31, 0.03)
        
    elif letter == 'T':
        # Fist with thumb tucked between index and middle
        c = r_curl(0.85, 0.95)
        curl_finger(lm, 5, c)
        curl_finger(lm, 9, r_curl(0.90, 0.98))
        curl_finger(lm, 13, r_curl(0.90, 0.98))
        curl_finger(lm, 17, r_curl(0.90, 0.98))
        position_thumb(lm, 'between_index_middle')
        lm[4, 0] = r_val(-0.08, 0.02)
        lm[4, 1] = r_val(0.46, 0.03)
        
    elif letter == 'U':
        # Index and middle extended straight up, pressed together; ring & pinky curled
        curl_finger(lm, 5, r_curl(0.0, 0.06))
        curl_finger(lm, 9, r_curl(0.0, 0.06))
        c = r_curl(0.88, 0.98)
        curl_finger(lm, 13, c)
        curl_finger(lm, 17, c)
        lm[5:9, 0] = r_val(-0.06, 0.02)
        lm[9:13, 0] = r_val(0.02, 0.02)
        position_thumb(lm, 'fist_over')
        
    elif letter == 'V':
        # Index and middle extended in a 'V' shape (spread apart)
        curl_finger(lm, 5, r_curl(0.0, 0.06))
        curl_finger(lm, 9, r_curl(0.0, 0.06))
        c = r_curl(0.88, 0.98)
        curl_finger(lm, 13, c)
        curl_finger(lm, 17, c)
        lm[5:9, 0] -= r_val(0.10, 0.03)
        lm[9:13, 0] += r_val(0.10, 0.03)
        position_thumb(lm, 'fist_over')
        
    elif letter == 'W':
        # Index, middle, and ring extended spread up (W), pinky curled, thumb holds pinky
        curl_finger(lm, 5, r_curl(0.0, 0.06))
        curl_finger(lm, 9, r_curl(0.0, 0.06))
        curl_finger(lm, 13, r_curl(0.0, 0.06))
        curl_finger(lm, 17, r_curl(0.88, 0.98))
        lm[5:9, 0] -= r_val(0.12, 0.03)
        lm[13:17, 0] += r_val(0.12, 0.03)
        position_thumb(lm, 'across_palm')
        lm[4] = lm[20].copy()
        
    elif letter == 'X':
        # Index hooked (bent like a hook/pirate finger), other fingers curled
        curl_finger(lm, 5, r_curl(0.40, 0.55))
        lm[7:9, 2] += r_val(0.12, 0.03)
        c = r_curl(0.88, 0.98)
        curl_finger(lm, 9, c)
        curl_finger(lm, 13, c)
        curl_finger(lm, 17, c)
        position_thumb(lm, 'fist_side')
        
    elif letter == 'Y':
        # Thumb and pinky extended wide, middle 3 curled (shaka/hang loose shape)
        c = r_curl(0.88, 0.98)
        curl_finger(lm, 5, c)
        curl_finger(lm, 9, c)
        curl_finger(lm, 13, c)
        curl_finger(lm, 17, r_curl(0.0, 0.06))
        position_thumb(lm, 'out')
        lm[17:21, 0] += r_val(0.15, 0.04)
        
    elif letter == 'Z':
        # Index pointing forward tracing Z, others curled into fist
        curl_finger(lm, 5, r_curl(0.0, 0.06))
        lm[5:9, 2] += r_val(0.15, 0.03)
        c = r_curl(0.88, 0.98)
        curl_finger(lm, 9, c)
        curl_finger(lm, 13, c)
        curl_finger(lm, 17, c)
        position_thumb(lm, 'fist_over')
        
    return lm

def normalize_landmarks(landmarks):
    """
    Standard MediaPipe Landmark Normalization:
    1. Center relative to Wrist (landmark 0).
    2. Scale invariance: divide by distance from wrist to middle MCP (landmark 9).
    This guarantees position and distance invariance!
    """
    lm = landmarks.copy()
    wrist = lm[0].copy()
    lm = lm - wrist
    
    # Distance to middle finger knuckle (landmark 9)
    dist = np.linalg.norm(lm[9])
    if dist < 1e-6:
        dist = 1.0
    lm = lm / dist
    return lm

def augment_landmarks(landmarks, rng):
    """
    Applies realistic 3D human variation augmentations:
    - 3D Rotation (Roll, Pitch, Yaw)
    - Scale jitter
    - Joint noise / slight anatomical stretch
    """
    lm = landmarks.copy()
    
    # Random 3D Rotation angles (in radians)
    roll = rng.uniform(-0.25, 0.25)   # ~15 degrees
    pitch = rng.uniform(-0.25, 0.25)
    yaw = rng.uniform(-0.30, 0.30)    # ~17 degrees
    
    # Rotation matrices
    Rx = np.array([[1, 0, 0],
                   [0, math.cos(pitch), -math.sin(pitch)],
                   [0, math.sin(pitch), math.cos(pitch)]])
    
    Ry = np.array([[math.cos(yaw), 0, math.sin(yaw)],
                   [0, 1, 0],
                   [-math.sin(yaw), 0, math.cos(yaw)]])
                   
    Rz = np.array([[math.cos(roll), -math.sin(roll), 0],
                   [math.sin(roll), math.cos(roll), 0],
                   [0, 0, 1]])
                   
    R = Rz @ Ry @ Rx
    lm = (R @ lm.T).T
    
    # Random Scaling (0.85 to 1.15)
    scale = rng.uniform(0.85, 1.15)
    lm = lm * scale
    
    # Gaussian joint jitter (represents natural hand trembling & camera noise)
    jitter = rng.normal(0.0, 0.022, lm.shape)
    lm = lm + jitter
    
    return lm

def generate_dataset(samples_per_class=800, output_dir='python/data'):
    """
    Generates balanced, augmented dataset for all 26 letters (A-Z).
    Total samples = 26 * 800 = 20,800 samples.
    """
    os.makedirs(output_dir, exist_ok=True)
    rng = np.random.RandomState(42)
    
    data = []
    labels = []
    
    print(f"Generating synthetic ASL landmark dataset ({samples_per_class} samples/class)...")
    for letter_idx, letter in enumerate(CLASSES):
        for _ in range(samples_per_class):
            pose = generate_canonical_sign(letter, rng=rng)
            augmented = augment_landmarks(pose, rng)
            normalized = normalize_landmarks(augmented)
            
            # Flatten 21 x 3 into 63 float features
            flat_features = normalized.flatten()
            data.append(flat_features)
            labels.append(letter_idx)
            
    df_data = pd.DataFrame(data)
    df_labels = pd.DataFrame({'label': labels, 'letter': [CLASSES[i] for i in labels]})
    
    features_path = os.path.join(output_dir, 'asl_landmarks.csv')
    labels_path = os.path.join(output_dir, 'asl_labels.csv')
    
    df_data.to_csv(features_path, index=False, header=False)
    df_labels.to_csv(labels_path, index=False)
    
    print(f"Successfully generated {len(data)} samples across {len(CLASSES)} classes (A-Z)!")
    print(f"Saved features: {features_path}")
    print(f"Saved labels: {labels_path}")
    return features_path, labels_path

if __name__ == '__main__':
    generate_dataset()

