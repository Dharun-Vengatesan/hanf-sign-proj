using System;
using UnityEngine;

namespace ASLSignDetection
{
    /// <summary>
    /// Processes raw MediaPipe hand landmarks, extracts normalized feature vectors
    /// for neural classification, and provides bone connection topology for visualization.
    /// </summary>
    public class HandLandmarkExtractor : MonoBehaviour
    {
        // Standard 21 MediaPipe hand connections (pairs of joint indices)
        public static readonly (int, int)[] BONE_CONNECTIONS = new (int, int)[]
        {
            // Palm base
            (0, 1), (0, 5), (5, 9), (9, 13), (13, 17), (0, 17),
            // Thumb
            (1, 2), (2, 3), (3, 4),
            // Index finger
            (5, 6), (6, 7), (7, 8),
            // Middle finger
            (9, 10), (10, 11), (11, 12),
            // Ring finger
            (13, 14), (14, 15), (15, 16),
            // Pinky finger
            (17, 18), (18, 19), (19, 20)
        };

        [Header("Settings")]
        [Tooltip("Minimum knuckle distance threshold to prevent divide-by-zero.")]
        [SerializeField] private float minKnuckleDistance = 0.001f;

        [Header("Extracted Data")]
        [SerializeField] private bool hasValidHand = false;
        [SerializeField] private float handStability = 1.0f; // 1.0 = completely still, 0.0 = fast moving

        private Vector3[] previousLandmarks = new Vector3[21];
        private float[] cachedFeatures = new float[63];

        public bool HasValidHand => hasValidHand;
        public float HandStability => handStability;
        public float[] CachedFeatures => cachedFeatures;

        public event Action<float[], HandTrackingData> OnFeaturesExtracted;

        /// <summary>
        /// Extracts 63 normalized coordinates from raw MediaPipe hand tracking data.
        /// </summary>
        public float[] ExtractFeatures(HandTrackingData handData)
        {
            if (handData == null || !handData.hand_detected)
            {
                hasValidHand = false;
                return null;
            }

            // 1. If the python bridge already sent pre-normalized features (63 floats), use them directly!
            if (handData.features != null && handData.features.Length == 63)
            {
                Array.Copy(handData.features, cachedFeatures, 63);
                hasValidHand = true;
                CalculateStability(handData);
                OnFeaturesExtracted?.Invoke(cachedFeatures, handData);
                return cachedFeatures;
            }

            // 2. Otherwise compute normalization from raw 21 landmarks
            if (handData.LandmarkCount < 21)
            {
                hasValidHand = false;
                return null;
            }

            // Otherwise, compute normalization from raw 21 landmarks:
            // 1. Center relative to Wrist (landmark 0)
            Vector3 wrist = handData.GetLandmark(0);
            Vector3[] centered = new Vector3[21];
            bool isLeft = (handData.handedness == "Left");

            for (int i = 0; i < 21; i++)
            {
                Vector3 pt = handData.GetLandmark(i) - wrist;
                // Invert Y because MediaPipe Y increases downwards (0 at top, 1 at bottom)
                pt.y = -pt.y;
                // Invert Z because MediaPipe Z is negative towards camera
                pt.z = -pt.z;
                // Mirror X if Left hand so signs map to canonical Right Hand
                if (isLeft)
                {
                    pt.x = -pt.x;
                }
                centered[i] = pt;
            }

            // 2. Scale invariance: divide by distance from wrist to middle MCP (landmark 9)
            float scale = centered[9].magnitude;
            if (scale < minKnuckleDistance)
            {
                scale = 1.0f;
            }

            // 3. Populate 63-element feature array (x0, y0, z0, ... x20, y20, z20)
            for (int i = 0; i < 21; i++)
            {
                Vector3 normalizedPoint = centered[i] / scale;
                cachedFeatures[i * 3 + 0] = normalizedPoint.x;
                cachedFeatures[i * 3 + 1] = normalizedPoint.y;
                cachedFeatures[i * 3 + 2] = normalizedPoint.z;
            }

            hasValidHand = true;
            CalculateStability(handData);
            OnFeaturesExtracted?.Invoke(cachedFeatures, handData);
            return cachedFeatures;
        }

        private void CalculateStability(HandTrackingData handData)
        {
            float totalDelta = 0f;
            for (int i = 0; i < 21; i++)
            {
                Vector3 current = handData.GetLandmark(i);
                totalDelta += Vector3.Distance(current, previousLandmarks[i]);
                previousLandmarks[i] = current;
            }

            // Stability: 1.0 when velocity is zero, decays as hand moves
            float avgMovement = totalDelta / 21f;
            handStability = Mathf.Clamp01(1.0f - (avgMovement * 15f));
        }
    }
}
