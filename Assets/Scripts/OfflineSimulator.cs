using System;
using System.Collections.Generic;
using UnityEngine;

namespace ASLSignDetection
{
    /// <summary>
    /// Offline Hand Landmark Simulator for college presentations and testing.
    /// Injects realistic canonical MediaPipe hand landmarks for any letter A-Z
    /// when the user presses keyboard keys [A] - [Z] or triggers simulation.
    /// Ensures 100% fail-safe demonstrations even if webcam is unavailable!
    /// </summary>
    public class OfflineSimulator : MonoBehaviour
    {
        [Header("Settings")]
        [Tooltip("Target MediaPipeManager to receive simulated landmark streams.")]
        [SerializeField] private MediaPipeManager mediaPipeManager;

        [Tooltip("Whether keyboard shortcuts [A-Z] are enabled to inject mock signs.")]
        [SerializeField] private bool enableKeyboardSimulation = true;

        [Tooltip("Add subtle natural hand jitter/shake to simulated landmarks.")]
        [SerializeField] private bool addRealisticJitter = true;

        private Dictionary<string, LandmarkPoint[]> canonicalPosesDict = new Dictionary<string, LandmarkPoint[]>();
        private string activeSimulatedLetter = "";

        private void Awake()
        {
            LoadCanonicalPoses();
        }

        private void LoadCanonicalPoses()
        {
            try
            {
                string path = System.IO.Path.Combine(Application.streamingAssetsPath, "asl_canonical_landmarks.json");
                if (System.IO.File.Exists(path))
                {
                    string json = System.IO.File.ReadAllText(path);
                    CanonicalWrapper wrapper = JsonUtility.FromJson<CanonicalWrapper>(json);
                    if (wrapper != null && wrapper.items != null)
                    {
                        foreach (var item in wrapper.items)
                        {
                            canonicalPosesDict[item.letter.ToUpper()] = item.landmarks;
                        }
                        Debug.Log($"[OfflineSimulator] Loaded {canonicalPosesDict.Count} canonical ASL signs from StreamingAssets.");
                    }
                }
            }
            catch (Exception ex)
            {
                Debug.LogWarning("[OfflineSimulator] Note loading canonical poses: " + ex.Message);
            }
        }

        [Serializable]
        private class CanonicalWrapper
        {
            public CanonicalItem[] items;
        }

        [Serializable]
        private class CanonicalItem
        {
            public string letter;
            public LandmarkPoint[] landmarks;
        }

        private void Update()
        {
            if (!enableKeyboardSimulation || mediaPipeManager == null) return;

            // Check A through Z keys
            for (char c = 'A'; c <= 'Z'; c++)
            {
                KeyCode key = (KeyCode)Enum.Parse(typeof(KeyCode), c.ToString());
                if (Input.GetKey(key))
                {
                    InjectLetter(c.ToString());
                    return;
                }
                if (Input.GetKeyUp(key))
                {
                    mediaPipeManager.FeedSimulatedData(new HandTrackingData
                    {
                        hand_detected = false,
                        status = "No Hand (Transition)",
                        landmarks = null,
                        features = null
                    });
                }
            }

            // Key '0' or Space to simulate 'No Hand'
            if (Input.GetKeyDown(KeyCode.Space) || Input.GetKeyDown(KeyCode.Alpha0))
            {
                mediaPipeManager.FeedSimulatedData(new HandTrackingData
                {
                    hand_detected = false,
                    status = "No Hand (Simulation)",
                    landmarks = null,
                    features = null
                });
            }
        }

        /// <summary>
        /// Injects canonical landmarks for a chosen letter into the detection pipeline.
        /// </summary>
        public void InjectLetter(string letter)
        {
            activeSimulatedLetter = letter.ToUpper();
            LandmarkPoint[] landmarks = GenerateMockLandmarks(activeSimulatedLetter);

            HandTrackingData simData = new HandTrackingData
            {
                hand_detected = true,
                status = $"Simulated '{activeSimulatedLetter}'",
                confidence = 0.99f,
                handedness = "Right",
                fps = 30f,
                timestamp = Time.time,
                landmarks = landmarks
            };

            mediaPipeManager.FeedSimulatedData(simData);
        }

        private LandmarkPoint[] GenerateMockLandmarks(string letter)
        {
            if (canonicalPosesDict.TryGetValue(letter, out LandmarkPoint[] baseLm) && baseLm != null && baseLm.Length == 21)
            {
                LandmarkPoint[] copy = new LandmarkPoint[21];
                for (int i = 0; i < 21; i++)
                {
                    float jitterX = addRealisticJitter ? UnityEngine.Random.Range(-0.003f, 0.003f) : 0f;
                    float jitterY = addRealisticJitter ? UnityEngine.Random.Range(-0.003f, 0.003f) : 0f;
                    copy[i] = new LandmarkPoint(baseLm[i].x + jitterX, baseLm[i].y + jitterY, baseLm[i].z);
                }
                return copy;
            }

            LandmarkPoint[] lm = new LandmarkPoint[21];
            // Center hand at screen coords (0.5, 0.5)
            float cx = 0.5f;
            float cy = 0.5f;
            float jitter = addRealisticJitter ? UnityEngine.Random.Range(-0.005f, 0.005f) : 0f;

            // 0 Wrist
            lm[0] = new LandmarkPoint(cx, cy + 0.25f, 0f);

            // Default curled positions
            for (int i = 1; i < 21; i++)
            {
                lm[i] = new LandmarkPoint(cx + ((i % 4) * 0.03f), cy + 0.10f - ((i / 4) * 0.04f) + jitter, 0.02f);
            }

            // Finger specific extension based on ASL letter
            switch (letter)
            {
                case "A":
                    // Fist, thumb upright on side
                    lm[4] = new LandmarkPoint(cx - 0.08f, cy + 0.08f, 0f);
                    break;

                case "B":
                    // 4 fingers extended straight up
                    for (int f = 0; f < 4; f++)
                    {
                        int baseIdx = 5 + (f * 4);
                        for (int s = 0; s < 4; s++)
                            lm[baseIdx + s] = new LandmarkPoint(cx - 0.06f + (f * 0.04f), cy + 0.10f - (s * 0.08f), 0f);
                    }
                    lm[4] = new LandmarkPoint(cx + 0.02f, cy + 0.12f, 0.04f); // thumb across
                    break;

                case "C":
                    // Curved C
                    for (int f = 0; f < 4; f++)
                    {
                        int tip = 8 + (f * 4);
                        lm[tip] = new LandmarkPoint(cx + 0.10f, cy - 0.05f + (f * 0.03f), 0.08f);
                    }
                    lm[4] = new LandmarkPoint(cx - 0.08f, cy + 0.10f, 0.08f);
                    break;

                case "D":
                    // Index straight up
                    for (int s = 0; s < 4; s++)
                        lm[5 + s] = new LandmarkPoint(cx, cy + 0.10f - (s * 0.08f), 0f);
                    lm[4] = new LandmarkPoint(cx + 0.03f, cy + 0.08f, 0.04f);
                    break;

                case "L":
                    // Index up, thumb horizontal
                    for (int s = 0; s < 4; s++)
                        lm[5 + s] = new LandmarkPoint(cx, cy + 0.10f - (s * 0.08f), 0f);
                    for (int s = 0; s < 4; s++)
                        lm[1 + s] = new LandmarkPoint(cx - (s * 0.06f), cy + 0.15f, 0f);
                    break;

                case "V":
                    // Index and middle in V
                    for (int s = 0; s < 4; s++)
                    {
                        lm[5 + s] = new LandmarkPoint(cx - 0.05f - (s * 0.02f), cy + 0.10f - (s * 0.08f), 0f);
                        lm[9 + s] = new LandmarkPoint(cx + 0.05f + (s * 0.02f), cy + 0.10f - (s * 0.08f), 0f);
                    }
                    break;

                case "Y":
                    // Thumb and pinky spread wide
                    for (int s = 0; s < 4; s++)
                    {
                        lm[1 + s] = new LandmarkPoint(cx - 0.05f - (s * 0.04f), cy + 0.15f - (s * 0.02f), 0f);
                        lm[17 + s] = new LandmarkPoint(cx + 0.05f + (s * 0.04f), cy + 0.10f - (s * 0.06f), 0f);
                    }
                    break;
            }

            return lm;
        }
    }
}
