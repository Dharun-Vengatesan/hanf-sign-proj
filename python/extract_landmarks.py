"""
MediaPipe Landmark Extractor from Image Directory
Extracts 21 3D hand landmarks from any folder of ASL images
and appends them to a CSV dataset.
"""

import os
import glob
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd

def extract_from_directory(image_root_dir, output_csv_path='python/data/extracted_landmarks.csv'):
    """
    Scans a folder structured as:
      image_root_dir/
         A/
            img1.jpg, img2.jpg, ...
         B/
            ...
         Z/
    Processes each image through MediaPipe Hands, extracts 21 landmarks,
    normalizes relative to wrist, and saves to CSV.
    """
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.5
    )
    
    rows = []
    labels = []
    
    classes = [chr(i) for i in range(ord('A'), ord('Z') + 1)]
    
    print(f"Scanning {image_root_dir} for ASL letters A-Z...")
    for letter_idx, letter in enumerate(classes):
        letter_dir = os.path.join(image_root_dir, letter)
        if not os.path.exists(letter_dir):
            continue
            
        image_files = glob.glob(os.path.join(letter_dir, "*.*"))
        print(f"Processing letter '{letter}': {len(image_files)} images found.")
        
        for img_path in image_files:
            img = cv2.imread(img_path)
            if img is None:
                continue
                
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)
            
            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                
                # Extract 21 x 3 coordinates
                raw_coords = np.zeros((21, 3), dtype=np.float32)
                for idx, lm in enumerate(hand_landmarks.landmark):
                    raw_coords[idx] = [lm.x, lm.y, lm.z]
                    
                # Normalize relative to wrist (point 0)
                wrist = raw_coords[0].copy()
                centered = raw_coords - wrist
                scale = np.linalg.norm(centered[9]) # dist to middle knuckle
                if scale < 1e-6:
                    scale = 1.0
                norm_coords = centered / scale
                
                flat = norm_coords.flatten()
                rows.append(flat)
                labels.append(letter_idx)
                
    hands.close()
    
    if len(rows) > 0:
        df = pd.DataFrame(rows)
        df.insert(0, 'label', labels)
        os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
        df.to_csv(output_csv_path, index=False)
        print(f"Extraction complete! Saved {len(rows)} samples to {output_csv_path}")
    else:
        print("No hand landmarks extracted. Ensure valid image folder path.")

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        extract_from_directory(sys.argv[1])
    else:
        print("Usage: python extract_landmarks.py <path_to_asl_alphabet_images>")
