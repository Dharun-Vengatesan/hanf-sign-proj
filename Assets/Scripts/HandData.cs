using System;
using UnityEngine;

namespace ASLSignDetection
{
    /// <summary>
    /// Represents a single 3D landmark coordinate (x, y, z).
    /// Fully supported by Unity's JsonUtility serialization.
    /// </summary>
    [Serializable]
    public struct LandmarkPoint
    {
        public float x;
        public float y;
        public float z;

        public LandmarkPoint(float x, float y, float z)
        {
            this.x = x;
            this.y = y;
            this.z = z;
        }

        public Vector3 ToVector3() => new Vector3(x, y, z);
    }

    /// <summary>
    /// Represents the complete hand tracking payload from MediaPipe Hands.
    /// </summary>
    [Serializable]
    public class HandTrackingData
    {
        public bool hand_detected;
        public string status;
        public float confidence;
        public string handedness;
        public float fps;
        public double timestamp;

        // 21 Landmark points (struct array - 100% supported by Unity JsonUtility)
        public LandmarkPoint[] landmarks;

        // Flat raw landmarks: 63 floats (x0, y0, z0, ... x20, y20, z20)
        public float[] raw_landmarks;

        // 63 normalized coordinates for classification (x0,y0,z0, ... x20,y20,z20)
        public float[] features;

        // Optional base64-encoded JPEG camera frame from MediaPipe bridge
        public string image;

        /// <summary>
        /// Gets a specific landmark as a Vector3.
        /// </summary>
        public Vector3 GetLandmark(int index)
        {
            if (landmarks != null && index >= 0 && index < landmarks.Length)
            {
                return new Vector3(landmarks[index].x, landmarks[index].y, landmarks[index].z);
            }
            if (raw_landmarks != null && (index * 3 + 2) < raw_landmarks.Length)
            {
                return new Vector3(raw_landmarks[index * 3], raw_landmarks[index * 3 + 1], raw_landmarks[index * 3 + 2]);
            }
            return Vector3.zero;
        }

        /// <summary>
        /// Total number of detected landmarks (typically 21).
        /// </summary>
        public int LandmarkCount
        {
            get
            {
                if (landmarks != null && landmarks.Length > 0) return landmarks.Length;
                if (raw_landmarks != null && raw_landmarks.Length >= 63) return raw_landmarks.Length / 3;
                return 0;
            }
        }
    }

    /// <summary>
    /// Classification result containing predicted letter, confidence, and timestamp.
    /// </summary>
    public struct ClassificationResult
    {
        public string Letter;
        public float Confidence;
        public float[] Probabilities;
        public float InferenceTimeMs;
        public bool IsValid;

        public static ClassificationResult Unknown(float confidence = 0f)
        {
            return new ClassificationResult
            {
                Letter = "Unknown",
                Confidence = confidence,
                Probabilities = new float[26],
                InferenceTimeMs = 0f,
                IsValid = false
            };
        }
    }
}
