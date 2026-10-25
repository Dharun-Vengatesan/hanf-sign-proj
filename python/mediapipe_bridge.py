"""
MediaPipe Hands Real-Time UDP Bridge
Captures webcam, extracts 21 3D hand landmarks via Google MediaPipe Hands,
encodes the video frame as JPEG, and streams both to Unity via local UDP.
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['GLOG_minloglevel'] = '3'
import warnings
warnings.filterwarnings('ignore')

import sys
import time
import json
import base64
import socket
import argparse
import cv2
import mediapipe as mp
import numpy as np

def run_bridge(camera_id=0, target_ip="127.0.0.1", target_port=5052, show_preview=False, send_video=True):
    print("==================================================")
    print("   ASL MediaPipe Hands Real-Time UDP Bridge       ")
    print("==================================================")
    print(f"Target Unity UDP: {target_ip}:{target_port}")
    print(f"Camera ID: {camera_id}")
    print(f"Send Video Frames: {send_video}")
    print("Initializing socket...")
    
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    
    # Initialize MediaPipe Hands
    try:
        mp_hands = mp.solutions.hands
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles
        
        hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=1,
            min_detection_confidence=0.60,
            min_tracking_confidence=0.55
        )
    except AttributeError:
        err_msg = (
            f"CRITICAL: 'mediapipe.solutions' was not found in this Python environment ({sys.executable}).\n"
            "This happens if MediaPipe >= 1.0 is installed where 'solutions' was removed.\n"
            "Please install mediapipe==0.10.14 under Python 3.11:\n"
            "  pip install mediapipe==0.10.14"
        )
        print(f"\n[ERROR] {err_msg}\n", file=sys.stderr)
        try:
            payload = json.dumps({"hand_detected": False, "status": "Python Error: Install mediapipe==0.10.14", "error": err_msg}).encode('utf-8')
            sock.sendto(payload, (target_ip, target_port))
        except Exception:
            pass
        sys.exit(1)
    
    print(f"Opening webcam {camera_id}...")
    cap = None
    
    # Try different backends on Windows
    backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY] if sys.platform == 'win32' else [cv2.CAP_ANY]
    for backend in backends:
        c = cv2.VideoCapture(camera_id, backend)
        if c.isOpened():
            ret, test_frame = c.read()
            if ret and test_frame is not None:
                cap = c
                print(f"Successfully opened camera with backend: {backend}")
                break
            else:
                c.release()
                
    if cap is None or not cap.isOpened():
        # Try camera index 1
        print("Camera 0 failed. Trying camera 1...")
        cap = cv2.VideoCapture(1)
        
    if cap is None or not cap.isOpened():
        error_msg = f"Webcam unavailable. Make sure no other program is using the camera."
        print(f"FATAL: {error_msg}")
        payload = json.dumps({"hand_detected": False, "status": "Camera not found", "error": error_msg}).encode('utf-8')
        sock.sendto(payload, (target_ip, target_port))
        sys.exit(1)
        
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    prev_time = time.time()
    frame_count = 0
    fps = 30.0
    
    print("\nMediaPipe Hands Bridge is RUNNING! Streaming video + landmarks to Unity...")
    if show_preview:
        print("Press 'q' in the preview window to exit.")
        
    try:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                time.sleep(0.01)
                continue
                
            frame_count += 1
            cur_time = time.time()
            if cur_time - prev_time >= 1.0:
                fps = frame_count / (cur_time - prev_time)
                frame_count = 0
                prev_time = cur_time
                
            # Flip horizontally for natural mirror selfie view
            frame = cv2.flip(frame, 1)
            
            # MediaPipe expects RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)
            
            # Compress video frame to JPEG for Unity display
            b64_image = ""
            if send_video:
                # Downscale slightly to 480x360 to guarantee fast transmission (< 16 KB)
                small_frame = cv2.resize(frame, (480, 360), interpolation=cv2.INTER_AREA)
                ret, buf = cv2.imencode('.jpg', small_frame, [cv2.IMWRITE_JPEG_QUALITY, 55])
                if ret:
                    b64_image = base64.b64encode(buf).decode('ascii')
            
            payload = {}
            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                
                handedness = "Right"
                confidence = 0.95
                if results.multi_handedness:
                    handedness = results.multi_handedness[0].classification[0].label
                    confidence = float(results.multi_handedness[0].classification[0].score)
                    
                # 1. Screen coordinates (0.0 to 1.0)
                screen_landmarks = []
                raw_landmarks_flat = []
                raw_coords = np.zeros((21, 3), dtype=np.float32)
                for i, lm in enumerate(hand_landmarks.landmark):
                    screen_landmarks.append({"x": float(lm.x), "y": float(lm.y), "z": float(lm.z)})
                    raw_landmarks_flat.extend([float(lm.x), float(lm.y), float(lm.z)])
                    raw_coords[i] = [lm.x, lm.y, lm.z]
                    
                # 2. Invariant Normalization for ASL Classifier
                wrist = raw_coords[0].copy()
                centered = raw_coords - wrist
                
                # Coordinate alignment with dataset:
                # MediaPipe Y: 0.0 at top, 1.0 at bottom. Invert Y so Up is +Y.
                centered[:, 1] = -centered[:, 1]
                # MediaPipe Z: negative towards camera. Invert Z so curling forward is +Z.
                centered[:, 2] = -centered[:, 2]
                
                # Handedness Invariance:
                # If Left Hand, mirror X so all signs match canonical Right Hand geometry
                if handedness == "Left":
                    centered[:, 0] = -centered[:, 0]
                
                scale = np.linalg.norm(centered[9]) # Distance to middle MCP
                if scale < 1e-6:
                    scale = 1.0
                normalized = centered / scale
                flat_features = normalized.flatten().tolist()
                
                payload = {
                    "hand_detected": True,
                    "status": "Hand Detected",
                    "landmarks": screen_landmarks,
                    "raw_landmarks": raw_landmarks_flat,
                    "features": flat_features,
                    "confidence": confidence,
                    "handedness": handedness,
                    "fps": round(fps, 1),
                    "timestamp": time.time(),
                    "image": b64_image
                }
                
                if show_preview:
                    mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style()
                    )
            else:
                payload = {
                    "hand_detected": False,
                    "status": "No Hand Detected",
                    "landmarks": [],
                    "raw_landmarks": [],
                    "features": [],
                    "confidence": 0.0,
                    "fps": round(fps, 1),
                    "timestamp": time.time(),
                    "image": b64_image
                }
                
            # Send UDP packet to Unity
            data_bytes = json.dumps(payload).encode('utf-8')
            sock.sendto(data_bytes, (target_ip, target_port))
            
            if show_preview:
                cv2.putText(frame, f"FPS: {fps:.1f} | Status: {payload['status']}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imshow("MediaPipe Hands - ASL Preview", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
    except KeyboardInterrupt:
        print("\nStopping MediaPipe bridge...")
    finally:
        if cap is not None:
            cap.release()
        if show_preview:
            cv2.destroyAllWindows()
        hands.close()
        sock.close()
        print("MediaPipe bridge terminated safely.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="MediaPipe Hands UDP Bridge for Unity")
    parser.add_argument("--camera", type=int, default=0, help="Camera device index (default: 0)")
    parser.add_argument("--ip", type=str, default="127.0.0.1", help="Target IP for Unity (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=5052, help="Target UDP port for Unity (default: 5052)")
    parser.add_argument("--preview", action="store_true", help="Show OpenCV debug preview window")
    parser.add_argument("--no-video", action="store_true", help="Do not send video frames (landmarks only)")
    args = parser.parse_args()
    
    run_bridge(camera_id=args.camera, target_ip=args.ip, target_port=args.port, 
               show_preview=args.preview, send_video=not args.no_video)
