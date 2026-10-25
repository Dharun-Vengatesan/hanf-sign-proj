using System;
using System.IO;
using System.Collections.Generic;
using UnityEngine;

namespace ASLSignDetection
{
    /// <summary>
    /// Lightweight high-speed neural network classifier for ASL A-Z recognition.
    /// Executes the trained MLP model locally in pure C# with sub-millisecond latency.
    /// Also handles temporal trajectory detection for dynamic letters 'J' and 'Z'.
    /// </summary>
    public class SignLanguageClassifier : MonoBehaviour
    {
        public static readonly string[] ALPHABET = new string[]
        {
            "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M",
            "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"
        };

        [Header("Model Weights")]
        [Tooltip("Relative path in StreamingAssets to the exported JSON model weights.")]
        [SerializeField] private string weightsFileName = "asl_weights.json";

        [Header("Classification Status")]
        [SerializeField] private bool isModelLoaded = false;
        [SerializeField] private string lastPredictedLetter = "None";
        [SerializeField] private float lastConfidence = 0.0f;
        [SerializeField] private float lastInferenceTimeMs = 0.0f;

        [Header("Dynamic J & Z Motion Tracking")]
        [Tooltip("Enable dynamic trajectory recognition for letters J and Z.")]
        [SerializeField] private bool enableDynamicLetters = true;

        [Tooltip("Sliding window size (frames) for tracking fingertip motion.")]
        [SerializeField] private int trajectoryWindowSize = 25;

        // Neural Network Layer Weights and Biases
        // Layer 1: [128, 63] + [128]
        private float[][] w1;
        private float[] b1;

        // Layer 2: [64, 128] + [64]
        private float[][] w2;
        private float[] b2;

        // Layer 3: [26, 64] + [26]
        private float[][] w3;
        private float[] b3;

        // Hidden layer activation buffers
        private float[] h1 = new float[128];
        private float[] h2 = new float[64];
        private float[] logits = new float[26];
        private float[] probs = new float[26];

        // Motion tracking for dynamic signs (pinky tip for J, index tip for Z)
        private Queue<Vector2> pinkyTipHistory = new Queue<Vector2>();
        private Queue<Vector2> indexTipHistory = new Queue<Vector2>();

        public bool IsModelLoaded => isModelLoaded;
        public string LastPredictedLetter => lastPredictedLetter;
        public float LastConfidence => lastConfidence;
        public float LastInferenceTimeMs => lastInferenceTimeMs;
        public float[] CurrentProbabilities => probs;

        public event Action<ClassificationResult> OnClassified;

        private void Awake()
        {
            LoadModelWeights();
        }

        /// <summary>
        /// Loads the trained neural network weights from StreamingAssets.
        /// </summary>
        public void LoadModelWeights()
        {
            try
            {
                string path = Path.Combine(Application.streamingAssetsPath, weightsFileName);
                if (!File.Exists(path))
                {
                    Debug.LogError($"[SignLanguageClassifier] Model weights file not found at: {path}");
                    isModelLoaded = false;
                    return;
                }

                string json = File.ReadAllText(path);
                ModelWeightsData data = JsonUtility.FromJson<ModelWeightsData>(json);

                if (data == null || data.layers == null)
                {
                    Debug.LogError("[SignLanguageClassifier] Failed to parse model weights JSON.");
                    return;
                }

                // Unpack Layer 1
                int hidden1 = 128;
                int inFeatures = 63;
                w1 = new float[hidden1][];
                b1 = new float[hidden1];
                for (int i = 0; i < hidden1; i++)
                {
                    w1[i] = new float[inFeatures];
                    for (int j = 0; j < inFeatures; j++)
                    {
                        w1[i][j] = data.layers.fc1.weight[i * inFeatures + j];
                    }
                    b1[i] = data.layers.fc1.bias[i];
                }

                // Unpack Layer 2
                int hidden2 = 64;
                w2 = new float[hidden2][];
                b2 = new float[hidden2];
                for (int i = 0; i < hidden2; i++)
                {
                    w2[i] = new float[hidden1];
                    for (int j = 0; j < hidden1; j++)
                    {
                        w2[i][j] = data.layers.fc2.weight[i * hidden1 + j];
                    }
                    b2[i] = data.layers.fc2.bias[i];
                }

                // Unpack Layer 3
                int numClasses = 26;
                w3 = new float[numClasses][];
                b3 = new float[numClasses];
                for (int i = 0; i < numClasses; i++)
                {
                    w3[i] = new float[hidden2];
                    for (int j = 0; j < hidden2; j++)
                    {
                        w3[i][j] = data.layers.fc3.weight[i * hidden2 + j];
                    }
                    b3[i] = data.layers.fc3.bias[i];
                }

                isModelLoaded = true;
                Debug.Log($"[SignLanguageClassifier] ASL model loaded successfully ({inFeatures} -> {hidden1} -> {hidden2} -> {numClasses}). 100% offline inference ready.");
            }
            catch (Exception ex)
            {
                Debug.LogError($"[SignLanguageClassifier] Error loading model weights: {ex.Message}");
                isModelLoaded = false;
            }
        }

        /// <summary>
        /// Classifies the 63 normalized landmark features and returns predicted letter with confidence.
        /// </summary>
        public ClassificationResult Classify(float[] features, HandTrackingData rawHandData = null)
        {
            if (!isModelLoaded || features == null || features.Length < 63)
            {
                return ClassificationResult.Unknown();
            }

            long startTime = System.Diagnostics.Stopwatch.GetTimestamp();

            // Layer 1: Dense(63 -> 128) + ReLU
            for (int i = 0; i < 128; i++)
            {
                float sum = b1[i];
                float[] row = w1[i];
                for (int j = 0; j < 63; j++)
                {
                    sum += row[j] * features[j];
                }
                h1[i] = sum > 0f ? sum : 0f; // ReLU
            }

            // Layer 2: Dense(128 -> 64) + ReLU
            for (int i = 0; i < 64; i++)
            {
                float sum = b2[i];
                float[] row = w2[i];
                for (int j = 0; j < 128; j++)
                {
                    sum += row[j] * h1[j];
                }
                h2[i] = sum > 0f ? sum : 0f; // ReLU
            }

            // Layer 3: Dense(64 -> 26) -> Logits
            float maxLogit = float.MinValue;
            for (int i = 0; i < 26; i++)
            {
                float sum = b3[i];
                float[] row = w3[i];
                for (int j = 0; j < 64; j++)
                {
                    sum += row[j] * h2[j];
                }
                logits[i] = sum;
                if (sum > maxLogit) maxLogit = sum;
            }

            // Softmax Normalization
            float sumExp = 0f;
            for (int i = 0; i < 26; i++)
            {
                probs[i] = Mathf.Exp(logits[i] - maxLogit);
                sumExp += probs[i];
            }

            int bestIndex = 0;
            float bestProb = 0f;
            for (int i = 0; i < 26; i++)
            {
                probs[i] /= sumExp;
                if (probs[i] > bestProb)
                {
                    bestProb = probs[i];
                    bestIndex = i;
                }
            }

            string predictedLetter = ALPHABET[bestIndex];

            // Dynamic Motion check for J and Z
            if (enableDynamicLetters && rawHandData != null && rawHandData.LandmarkCount >= 21)
            {
                predictedLetter = CheckDynamicGestures(predictedLetter, rawHandData, ref bestProb);
            }

            long endTime = System.Diagnostics.Stopwatch.GetTimestamp();
            lastInferenceTimeMs = (float)((endTime - startTime) * 1000.0 / System.Diagnostics.Stopwatch.Frequency);
            lastPredictedLetter = predictedLetter;
            lastConfidence = bestProb;

            ClassificationResult result = new ClassificationResult
            {
                Letter = predictedLetter,
                Confidence = bestProb,
                Probabilities = probs,
                InferenceTimeMs = lastInferenceTimeMs,
                IsValid = true
            };

            OnClassified?.Invoke(result);
            return result;
        }

        /// <summary>
        /// Detects dynamic trajectory motions for letters J (pinky arc) and Z (index zig-zag).
        /// </summary>
        private string CheckDynamicGestures(string staticLetter, HandTrackingData handData, ref float confidence)
        {
            Vector3 pinkyTip = handData.GetLandmark(20);
            Vector3 indexTip = handData.GetLandmark(8);

            pinkyTipHistory.Enqueue(new Vector2(pinkyTip.x, pinkyTip.y));
            if (pinkyTipHistory.Count > trajectoryWindowSize)
                pinkyTipHistory.Dequeue();

            indexTipHistory.Enqueue(new Vector2(indexTip.x, indexTip.y));
            if (indexTipHistory.Count > trajectoryWindowSize)
                indexTipHistory.Dequeue();

            // Check for 'J': Static base is 'I' or 'J', with a curved downward hook motion
            if (staticLetter == "I" || staticLetter == "J")
            {
                if (DetectJMotion(pinkyTipHistory))
                {
                    confidence = Mathf.Max(confidence, 0.95f);
                    return "J";
                }
            }

            // Check for 'Z': Static base is index pointing forward, with a zig-zag trajectory
            if (staticLetter == "Z" || staticLetter == "D" || staticLetter == "1")
            {
                if (DetectZMotion(indexTipHistory))
                {
                    confidence = Mathf.Max(confidence, 0.95f);
                    return "Z";
                }
            }

            return staticLetter;
        }

        private bool DetectJMotion(Queue<Vector2> history)
        {
            if (history.Count < 12) return false;
            Vector2[] points = history.ToArray();

            // J trajectory: initial vertical downward drop followed by a curve to the left/upward
            float startY = points[0].y;
            float minY = points[0].y;
            int minYIndex = 0;

            for (int i = 1; i < points.Length; i++)
            {
                if (points[i].y > minY) // In screen coords, Y increases downwards
                {
                    minY = points[i].y;
                    minYIndex = i;
                }
            }

            // Downward movement then upward scoop
            bool dropped = (minY - startY) > 0.08f;
            bool scooped = (minYIndex < points.Length - 3) && (minY - points[points.Length - 1].y > 0.03f);

            return dropped && scooped;
        }

        private bool DetectZMotion(Queue<Vector2> history)
        {
            if (history.Count < 15) return false;
            Vector2[] points = history.ToArray();

            // Z trajectory: horizontal right -> diagonal down-left -> horizontal right
            // Check sign changes in horizontal velocity
            int directionChanges = 0;
            float prevVx = 0f;

            for (int i = 1; i < points.Length; i++)
            {
                float vx = points[i].x - points[i - 1].x;
                if (Mathf.Abs(vx) > 0.015f)
                {
                    if (prevVx != 0f && Mathf.Sign(vx) != Mathf.Sign(prevVx))
                    {
                        directionChanges++;
                    }
                    prevVx = vx;
                }
            }

            return directionChanges >= 2;
        }

        // Serialization helper classes for parsing asl_weights.json
        [Serializable]
        private class ModelWeightsData
        {
            public string architecture;
            public int input_features;
            public string[] classes;
            public LayersData layers;
        }

        [Serializable]
        private class LayersData
        {
            public LayerWeight fc1;
            public LayerWeight fc2;
            public LayerWeight fc3;
        }

        [Serializable]
        private class LayerWeight
        {
            public float[] weight;
            public float[] bias;
        }
    }
}
