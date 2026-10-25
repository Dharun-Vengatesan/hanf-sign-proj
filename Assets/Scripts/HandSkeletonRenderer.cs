using UnityEngine;
using UnityEngine.UI;

namespace ASLSignDetection
{
    /// <summary>
    /// Renders the 21 MediaPipe hand landmarks and connecting skeleton bones
    /// directly overlaid on the webcam UI display.
    /// Supports toggleable debug visualization for college project presentations.
    /// </summary>
    public class HandSkeletonRenderer : MonoBehaviour
    {
        [Header("Target UI Area")]
        [Tooltip("RectTransform of the webcam feed display.")]
        [SerializeField] private RectTransform webcamRect;

        [Header("Visualization Settings")]
        [Tooltip("Whether to show the 21 landmarks and skeleton connections.")]
        [SerializeField] private bool showSkeleton = true;

        [Tooltip("Diameter of landmark joint dots (pixels).")]
        [SerializeField] private float jointSize = 10f;

        [Tooltip("Thickness of connecting bone lines (pixels).")]
        [SerializeField] private float boneThickness = 3f;

        [Tooltip("Joint dot color.")]
        [SerializeField] private Color jointColor = new Color(0.2f, 1.0f, 0.4f, 0.9f); // Bright green

        [Tooltip("Bone line color.")]
        [SerializeField] private Color boneColor = new Color(0.0f, 0.8f, 1.0f, 0.8f);  // Cyan

        // Procedural UI markers
        private RectTransform[] jointDots = new RectTransform[21];
        private RectTransform[] boneLines = new RectTransform[21];
        private Canvas parentCanvas;

        public bool ShowSkeleton
        {
            get => showSkeleton;
            set
            {
                showSkeleton = value;
                SetVisible(showSkeleton);
            }
        }

        private void Awake()
        {
            if (webcamRect == null)
                webcamRect = GetComponent<RectTransform>();

            parentCanvas = GetComponentInParent<Canvas>();
            CreateMarkers();
        }

        private void CreateMarkers()
        {
            // Create 21 Joint Dots
            for (int i = 0; i < 21; i++)
            {
                GameObject dotObj = new GameObject($"Joint_{i}", typeof(RectTransform), typeof(Image));
                dotObj.transform.SetParent(webcamRect, false);
                RectTransform rt = dotObj.GetComponent<RectTransform>();
                rt.sizeDelta = new Vector2(jointSize, jointSize);
                rt.anchorMin = Vector2.zero;
                rt.anchorMax = Vector2.zero;
                rt.pivot = new Vector2(0.5f, 0.5f);

                Image img = dotObj.GetComponent<Image>();
                img.color = jointColor;
                img.raycastTarget = false;

                // Colorize fingertips differently (tip landmarks: 4, 8, 12, 16, 20)
                if (i == 4 || i == 8 || i == 12 || i == 16 || i == 20)
                {
                    img.color = new Color(1.0f, 0.3f, 0.3f, 1.0f); // Red for tips
                    rt.sizeDelta = new Vector2(jointSize * 1.3f, jointSize * 1.3f);
                }

                dotObj.SetActive(false);
                jointDots[i] = rt;
            }

            // Create 21 Bone Lines
            for (int i = 0; i < HandLandmarkExtractor.BONE_CONNECTIONS.Length; i++)
            {
                GameObject lineObj = new GameObject($"Bone_{i}", typeof(RectTransform), typeof(Image));
                lineObj.transform.SetParent(webcamRect, false);
                // Ensure bones render behind dots
                lineObj.transform.SetSiblingIndex(0);

                RectTransform rt = lineObj.GetComponent<RectTransform>();
                rt.anchorMin = Vector2.zero;
                rt.anchorMax = Vector2.zero;
                rt.pivot = new Vector2(0f, 0.5f);

                Image img = lineObj.GetComponent<Image>();
                img.color = boneColor;
                img.raycastTarget = false;

                lineObj.SetActive(false);
                boneLines[i] = rt;
            }
        }

        /// <summary>
        /// Updates the 21 landmark markers according to the latest hand tracking data.
        /// </summary>
        public void UpdateSkeleton(HandTrackingData handData)
        {
            if (!showSkeleton || handData == null || !handData.hand_detected || handData.LandmarkCount < 21)
            {
                SetVisible(false);
                return;
            }

            SetVisible(true);
            Vector2 rectSize = webcamRect.rect.size;

            // 1. Position all 21 Joint Dots
            Vector2[] screenPoints = new Vector2[21];
            for (int i = 0; i < 21; i++)
            {
                Vector3 lm = handData.GetLandmark(i);
                float nx = lm.x;
                float ny = 1.0f - lm.y; // Invert Y for standard UI bottom-left origin

                Vector2 pos = new Vector2(nx * rectSize.x, ny * rectSize.y);
                screenPoints[i] = pos;

                if (jointDots[i] != null)
                {
                    jointDots[i].anchoredPosition = pos;
                }
            }

            // 2. Position and rotate all Bone Lines connecting pairs of joints
            var connections = HandLandmarkExtractor.BONE_CONNECTIONS;
            for (int i = 0; i < connections.Length; i++)
            {
                if (boneLines[i] == null) continue;

                int idxA = connections[i].Item1;
                int idxB = connections[i].Item2;

                Vector2 pointA = screenPoints[idxA];
                Vector2 pointB = screenPoints[idxB];

                Vector2 dir = pointB - pointA;
                float distance = dir.magnitude;
                float angle = Mathf.Atan2(dir.y, dir.x) * Mathf.Rad2Deg;

                boneLines[i].anchoredPosition = pointA;
                boneLines[i].sizeDelta = new Vector2(distance, boneThickness);
                boneLines[i].localRotation = Quaternion.Euler(0f, 0f, angle);
            }
        }

        public void SetVisible(bool visible)
        {
            for (int i = 0; i < jointDots.Length; i++)
            {
                if (jointDots[i] != null)
                    jointDots[i].gameObject.SetActive(visible);
            }

            for (int i = 0; i < boneLines.Length; i++)
            {
                if (boneLines[i] != null)
                    boneLines[i].gameObject.SetActive(visible);
            }
        }

        /// <summary>
        /// Dynamically updates the color of a specific landmark dot (used for AR accuracy heatmaps).
        /// </summary>
        public void SetJointColor(int index, Color color)
        {
            if (index >= 0 && index < jointDots.Length && jointDots[index] != null)
            {
                Image img = jointDots[index].GetComponent<Image>();
                if (img != null) img.color = color;
            }
        }

        /// <summary>
        /// Restores standard joint dot colors.
        /// </summary>
        public void ResetJointColors()
        {
            for (int i = 0; i < jointDots.Length; i++)
            {
                if (jointDots[i] != null)
                {
                    Image img = jointDots[i].GetComponent<Image>();
                    if (img != null)
                    {
                        img.color = (i == 4 || i == 8 || i == 12 || i == 16 || i == 20)
                            ? new Color(1.0f, 0.3f, 0.3f, 1.0f)
                            : jointColor;
                    }
                }
            }
        }
    }
}
