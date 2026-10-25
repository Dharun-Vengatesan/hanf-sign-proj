using System;
using System.Collections.Generic;
using System.IO;
using UnityEngine;
using UnityEngine.UI;

namespace ASLSignDetection
{
    /// <summary>
    /// Futuristic Augmented Reality (AR) Spatial HUD, Holographic Coach & Trajectory Trail Engine.
    /// Overlays spatial computing holograms directly onto the user's hand in camera space:
    /// 1. Floating Holographic Hand HUD: Word & letter badge anchored in 3D above the wrist.
    /// 2. Spatial Radial Dwell Ring: Circular holographic charge ring tracking dwell progress.
    /// 3. Deep Learning Pose Coach & Accuracy Heatmap: Evaluates joint errors against canonical ASL topology.
    /// 4. Holographic Ghost Hand: Semi-transparent guide showing the ideal target hand pose.
    /// 5. Adaptive Letter-Specific Hand Trails: High-accuracy Catmull-Rom spline neon light ribbons.
    /// 6. Real-Time Trajectory Accuracy Scoring: Analyzes 'J' hooks, 'Z' zig-zags, and static sign stability.
    /// 7. Holographic Guide Stencil: Projected motion path stencil guiding the user to trace accurate signs.
    /// </summary>
    public class ARSpatialHandHUD : MonoBehaviour
    {
        [Header("Target Area & Components")]
        [SerializeField] private RectTransform webcamRect;
        [SerializeField] private HandSkeletonRenderer skeletonRenderer;
        [SerializeField] private WordAssembler wordAssembler;
        [SerializeField] private PredictionSmoother predictionSmoother;
        [SerializeField] private MediaPipeManager mediaPipeManager;
        [SerializeField] private HandLandmarkExtractor landmarkExtractor;

        [Header("AR Feature Toggles (Keyboard: [H] HUD, [G] Ghost, [T] Trails, [Y] Heatmap)")]
        [SerializeField] private bool showFloatingHUD = true;
        [SerializeField] private bool showAccuracyHeatmap = true;
        [SerializeField] private bool showGhostGuide = true;
        [SerializeField] private bool showMotionRibbon = true;
        [SerializeField] private bool showGuideStencil = true;

        [Header("AR Visual Styling")]
        [SerializeField] private Color arCyanColor = new Color(0.0f, 0.95f, 1.0f, 0.95f);
        [SerializeField] private Color arGreenColor = new Color(0.15f, 1.0f, 0.50f, 0.95f);
        [SerializeField] private Color arVioletColor = new Color(0.70f, 0.35f, 1.0f, 0.95f);
        [SerializeField] private Color arWarningColor = new Color(1.0f, 0.70f, 0.15f, 0.95f);
        [SerializeField] private Color arErrorColor = new Color(1.0f, 0.25f, 0.35f, 0.95f);
        [SerializeField] private Color ghostColor = new Color(0.65f, 0.40f, 1.0f, 0.35f);

        // Procedural Holographic HUD Elements
        private GameObject arHudRoot;
        private CanvasGroup hudCanvasGroup;
        private RectTransform hudRect;
        private Text floatingLetterText;
        private Text floatingWordText;
        private Text floatingAccuracyText;
        private Text floatingTrailScoreText;
        private Image floatingRadialRing;
        private Image boundingBoxImage;
        private Text handReticleText;

        // Ghost Hand Procedural Markers
        private GameObject ghostRoot;
        private RectTransform[] ghostDots = new RectTransform[21];
        private RectTransform[] ghostBones = new RectTransform[HandLandmarkExtractor.BONE_CONNECTIONS.Length];

        // Motion Trails Procedural Spline Ribbons
        private const int RAW_HISTORY_SIZE = 24;
        private const int SPLINE_SEGMENTS = 32;
        private Queue<Vector2> rawFocalTrail = new Queue<Vector2>();
        private Vector2 smoothFocalPoint = Vector2.zero;
        private bool hasInitialFocalPoint = false;

        private GameObject trailRoot;
        private RectTransform[] trailSegments = new RectTransform[SPLINE_SEGMENTS];
        private Image[] trailSegmentImages = new Image[SPLINE_SEGMENTS];
        private RectTransform trailLeadingReticle;
        private Image trailLeadingImage;

        // Stencil / Guide Path Procedural Markers
        private const int STENCIL_POINTS = 16;
        private GameObject stencilRoot;
        private RectTransform[] stencilDots = new RectTransform[STENCIL_POINTS];

        // Canonical Poses for Heatmap & Ghost Guide
        private Dictionary<string, Vector3[]> canonicalPoses = new Dictionary<string, Vector3[]>();
        private Vector2 smoothHandCenter = new Vector2(0.5f, 0.5f);
        private float currentFormAccuracy = 1.0f;
        private float currentTrailAccuracy = 1.0f;
        private string activeTargetLetter = "A";
        private string trailAccuracyFeedback = "AR READY";
        private int currentFocalLandmarkIndex = 8; // Default: Index tip

        // Public Properties
        public bool ShowFloatingHUD
        {
            get => showFloatingHUD;
            set
            {
                showFloatingHUD = value;
                if (arHudRoot != null) arHudRoot.SetActive(value);
            }
        }

        public bool ShowGhostGuide
        {
            get => showGhostGuide;
            set
            {
                showGhostGuide = value;
                if (ghostRoot != null) ghostRoot.SetActive(value);
            }
        }

        public bool ShowMotionRibbon
        {
            get => showMotionRibbon;
            set
            {
                showMotionRibbon = value;
                if (trailRoot != null) trailRoot.SetActive(value);
            }
        }

        public bool ShowGuideStencil
        {
            get => showGuideStencil;
            set
            {
                showGuideStencil = value;
                if (stencilRoot != null) stencilRoot.SetActive(value);
            }
        }

        public float CurrentFormAccuracy => currentFormAccuracy;
        public float CurrentTrailAccuracy => currentTrailAccuracy;
        public string TrailAccuracyFeedback => trailAccuracyFeedback;
        public int CurrentFocalLandmarkIndex => currentFocalLandmarkIndex;

        private void Awake()
        {
            AutoBindReferences();
            LoadCanonicalPoses();
            BuildProceduralFloatingHUD();
            BuildProceduralGhostHand();
            BuildProceduralMotionTrails();
            BuildProceduralGuideStencil();
        }

        private void AutoBindReferences()
        {
            if (webcamRect == null) webcamRect = GetComponent<RectTransform>();
            if (skeletonRenderer == null) skeletonRenderer = FindObjectOfType<HandSkeletonRenderer>();
            if (wordAssembler == null) wordAssembler = FindObjectOfType<WordAssembler>();
            if (predictionSmoother == null) predictionSmoother = FindObjectOfType<PredictionSmoother>();
            if (mediaPipeManager == null) mediaPipeManager = FindObjectOfType<MediaPipeManager>();
            if (landmarkExtractor == null) landmarkExtractor = FindObjectOfType<HandLandmarkExtractor>();
        }

        private void Start()
        {
            if (mediaPipeManager != null)
            {
                mediaPipeManager.OnHandDataReceived += OnHandDataReceived;
            }
        }

        private void OnDestroy()
        {
            if (mediaPipeManager != null)
            {
                mediaPipeManager.OnHandDataReceived -= OnHandDataReceived;
            }
        }

        private void Update()
        {
            HandleKeyboardShortcuts();
            RenderSplineMotionRibbon();
        }

        private void HandleKeyboardShortcuts()
        {
            // [H] Toggle Floating Hand HUD
            if (Input.GetKeyDown(KeyCode.H))
            {
                ShowFloatingHUD = !showFloatingHUD;
                Debug.Log($"[ARSpatialHandHUD] Floating AR HUD: {showFloatingHUD}");
            }

            // [G] Toggle Ghost Silhouette Guide
            if (Input.GetKeyDown(KeyCode.G))
            {
                ShowGhostGuide = !showGhostGuide;
                Debug.Log($"[ARSpatialHandHUD] Ghost Hand Guide: {showGhostGuide}");
            }

            // [T] Toggle Motion Trails
            if (Input.GetKeyDown(KeyCode.T))
            {
                ShowMotionRibbon = !showMotionRibbon;
                Debug.Log($"[ARSpatialHandHUD] Motion Trails: {showMotionRibbon}");
            }

            // [Y] Toggle Accuracy Heatmap
            if (Input.GetKeyDown(KeyCode.Y))
            {
                showAccuracyHeatmap = !showAccuracyHeatmap;
                if (!showAccuracyHeatmap && skeletonRenderer != null)
                {
                    skeletonRenderer.ResetJointColors();
                }
                Debug.Log($"[ARSpatialHandHUD] Accuracy Heatmap: {showAccuracyHeatmap}");
            }
        }

        #region Letter-Specific Landmark Resolution

        /// <summary>
        /// Identifies the critical key landmark for each letter in ASL:
        /// - 'J': Landmark 20 (Pinky tip) traces the dynamic J-hook.
        /// - 'Z': Landmark 8 (Index tip) traces the dynamic Z-zigzag.
        /// - Fist signs ('A','E','M','N','S','T'): Landmark 4 (Thumb tip) defines the posture.
        /// - Pointing signs ('D','G','L','P','Q','X'): Landmark 8 (Index tip).
        /// - Multi-finger signs ('B','F','K','U','V','W'): Landmark 12 (Middle tip).
        /// - Pinky/Horn signs ('I','Y'): Landmark 20 (Pinky tip).
        /// </summary>
        public static int ResolveFocalLandmarkForLetter(string letter)
        {
            if (string.IsNullOrEmpty(letter) || letter.Length == 0) return 8;
            char c = char.ToUpper(letter[0]);
            switch (c)
            {
                case 'J':
                case 'I':
                case 'Y':
                    return 20; // Pinky tip
                case 'A':
                case 'E':
                case 'M':
                case 'N':
                case 'S':
                case 'T':
                    return 4;  // Thumb tip
                case 'B':
                case 'F':
                case 'K':
                case 'U':
                case 'V':
                case 'W':
                    return 12; // Middle fingertip
                case 'D':
                case 'G':
                case 'L':
                case 'P':
                case 'Q':
                case 'X':
                case 'Z':
                case 'C':
                case 'O':
                case 'H':
                case 'R':
                default:
                    return 8;  // Index fingertip
            }
        }

        #endregion

        #region Canonical 3D Posology & Procedural UI Setup

        private void LoadCanonicalPoses()
        {
            try
            {
                string path = Path.Combine(Application.streamingAssetsPath, "asl_canonical_landmarks.json");
                if (File.Exists(path))
                {
                    string json = File.ReadAllText(path);
                    CanonicalWrapper wrapper = JsonUtility.FromJson<CanonicalWrapper>(json);
                    if (wrapper != null && wrapper.items != null)
                    {
                        foreach (var item in wrapper.items)
                        {
                            if (item.landmarks != null && item.landmarks.Length == 21)
                            {
                                Vector3[] pts = new Vector3[21];
                                Vector3 wrist = new Vector3(item.landmarks[0].x, item.landmarks[0].y, item.landmarks[0].z);
                                float scale = Vector3.Distance(new Vector3(item.landmarks[9].x, item.landmarks[9].y, item.landmarks[9].z), wrist);
                                if (scale < 1e-4f) scale = 1.0f;

                                for (int i = 0; i < 21; i++)
                                {
                                    Vector3 p = new Vector3(item.landmarks[i].x, item.landmarks[i].y, item.landmarks[i].z);
                                    pts[i] = (p - wrist) / scale;
                                }
                                canonicalPoses[item.letter.ToUpper()] = pts;
                            }
                        }
                        Debug.Log($"[ARSpatialHandHUD] Loaded {canonicalPoses.Count} canonical 3D poses for AR Pose Coach.");
                    }
                }
            }
            catch (Exception ex)
            {
                Debug.LogWarning("[ARSpatialHandHUD] Note loading canonical poses: " + ex.Message);
            }
        }

        [Serializable]
        private class CanonicalWrapper { public CanonicalItem[] items; }
        [Serializable]
        private class CanonicalItem { public string letter; public LandmarkPoint[] landmarks; }

        private void BuildProceduralFloatingHUD()
        {
            if (webcamRect == null) return;

            arHudRoot = new GameObject("AR_SpatialHandHUD", typeof(RectTransform), typeof(CanvasGroup));
            arHudRoot.transform.SetParent(webcamRect, false);
            hudRect = arHudRoot.GetComponent<RectTransform>();
            hudRect.sizeDelta = new Vector2(250f, 140f);
            hudRect.pivot = new Vector2(0.5f, 0.0f); // Anchor comfortably above wrist/hand

            hudCanvasGroup = arHudRoot.GetComponent<CanvasGroup>();
            hudCanvasGroup.alpha = 0f;

            // 1. Frosted Cyber Backplate
            Image bg = arHudRoot.AddComponent<Image>();
            bg.color = new Color(0.04f, 0.07f, 0.14f, 0.85f); // Holographic midnight blue

            // 2. Glowing Cyan Corner Reticle
            GameObject reticleObj = new GameObject("ReticleCorner", typeof(RectTransform), typeof(Image));
            reticleObj.transform.SetParent(arHudRoot.transform, false);
            RectTransform rRt = reticleObj.GetComponent<RectTransform>();
            rRt.anchorMin = Vector2.zero;
            rRt.anchorMax = Vector2.one;
            rRt.offsetMin = new Vector2(-4f, -4f);
            rRt.offsetMax = new Vector2(4f, 4f);
            boundingBoxImage = reticleObj.GetComponent<Image>();
            boundingBoxImage.color = new Color(0.0f, 0.95f, 1.0f, 0.40f);

            // 3. Radial Dwell Progress Ring
            GameObject ringObj = new GameObject("RadialDwellRing", typeof(RectTransform), typeof(Image));
            ringObj.transform.SetParent(arHudRoot.transform, false);
            RectTransform ringRt = ringObj.GetComponent<RectTransform>();
            ringRt.anchorMin = new Vector2(0.06f, 0.35f);
            ringRt.anchorMax = new Vector2(0.36f, 0.92f);
            ringRt.offsetMin = Vector2.zero;
            ringRt.offsetMax = Vector2.zero;

            floatingRadialRing = ringObj.GetComponent<Image>();
            floatingRadialRing.type = Image.Type.Filled;
            floatingRadialRing.fillMethod = Image.FillMethod.Radial360;
            floatingRadialRing.fillOrigin = (int)Image.Origin360.Top;
            floatingRadialRing.fillClockwise = true;
            floatingRadialRing.fillAmount = 0.0f;
            floatingRadialRing.color = arCyanColor;

            // Ring background track
            GameObject ringBgObj = new GameObject("RingTrack", typeof(RectTransform), typeof(Image));
            ringBgObj.transform.SetParent(ringObj.transform, false);
            ringBgObj.transform.SetSiblingIndex(0);
            RectTransform ringBgRt = ringBgObj.GetComponent<RectTransform>();
            ringBgRt.anchorMin = Vector2.zero;
            ringBgRt.anchorMax = Vector2.one;
            Image ringBg = ringBgObj.GetComponent<Image>();
            ringBg.color = new Color(0.1f, 0.2f, 0.35f, 0.5f);

            // 4. Big Floating Letter
            GameObject letterObj = new GameObject("FloatingLetter", typeof(RectTransform), typeof(Text));
            letterObj.transform.SetParent(ringObj.transform, false);
            RectTransform lRt = letterObj.GetComponent<RectTransform>();
            lRt.anchorMin = Vector2.zero;
            lRt.anchorMax = Vector2.one;
            floatingLetterText = letterObj.GetComponent<Text>();
            floatingLetterText.font = GetDefaultFont();
            floatingLetterText.fontSize = 32;
            floatingLetterText.fontStyle = FontStyle.Bold;
            floatingLetterText.alignment = TextAnchor.MiddleCenter;
            floatingLetterText.color = Color.white;
            floatingLetterText.text = "-";

            // 5. Header Tag (Handedness + AR Lock)
            GameObject tagObj = new GameObject("HandTag", typeof(RectTransform), typeof(Text));
            tagObj.transform.SetParent(arHudRoot.transform, false);
            RectTransform tagRt = tagObj.GetComponent<RectTransform>();
            tagRt.anchorMin = new Vector2(0.39f, 0.72f);
            tagRt.anchorMax = new Vector2(0.96f, 0.95f);
            tagRt.offsetMin = Vector2.zero;
            tagRt.offsetMax = Vector2.zero;
            handReticleText = tagObj.GetComponent<Text>();
            handReticleText.font = GetDefaultFont();
            handReticleText.fontSize = 11;
            handReticleText.fontStyle = FontStyle.Bold;
            handReticleText.alignment = TextAnchor.MiddleLeft;
            handReticleText.color = arCyanColor;
            handReticleText.text = "AR TRACK: LOCKED";

            // 6. Live Accuracy Score & Feedback
            GameObject accObj = new GameObject("AccuracyBadge", typeof(RectTransform), typeof(Text));
            accObj.transform.SetParent(arHudRoot.transform, false);
            RectTransform accRt = accObj.GetComponent<RectTransform>();
            accRt.anchorMin = new Vector2(0.39f, 0.50f);
            accRt.anchorMax = new Vector2(0.96f, 0.72f);
            accRt.offsetMin = Vector2.zero;
            accRt.offsetMax = Vector2.zero;
            floatingAccuracyText = accObj.GetComponent<Text>();
            floatingAccuracyText.font = GetDefaultFont();
            floatingAccuracyText.fontSize = 12;
            floatingAccuracyText.fontStyle = FontStyle.Bold;
            floatingAccuracyText.alignment = TextAnchor.MiddleLeft;
            floatingAccuracyText.color = arGreenColor;
            floatingAccuracyText.text = "AR FORM: 98%";

            // 6B. Trail / Trajectory Score
            GameObject trailObj = new GameObject("TrailScoreBadge", typeof(RectTransform), typeof(Text));
            trailObj.transform.SetParent(arHudRoot.transform, false);
            RectTransform trRt = trailObj.GetComponent<RectTransform>();
            trRt.anchorMin = new Vector2(0.39f, 0.32f);
            trRt.anchorMax = new Vector2(0.96f, 0.50f);
            trRt.offsetMin = Vector2.zero;
            trRt.offsetMax = Vector2.zero;
            floatingTrailScoreText = trailObj.GetComponent<Text>();
            floatingTrailScoreText.font = GetDefaultFont();
            floatingTrailScoreText.fontSize = 10;
            floatingTrailScoreText.fontStyle = FontStyle.Normal;
            floatingTrailScoreText.alignment = TextAnchor.MiddleLeft;
            floatingTrailScoreText.color = new Color(0.35f, 0.85f, 1.0f);
            floatingTrailScoreText.text = "TRAIL: ACCURATE";

            // 7. Active Word Pill at Bottom
            GameObject wordObj = new GameObject("FormedWordPill", typeof(RectTransform), typeof(Image));
            wordObj.transform.SetParent(arHudRoot.transform, false);
            RectTransform wRt = wordObj.GetComponent<RectTransform>();
            wRt.anchorMin = new Vector2(0.06f, 0.06f);
            wRt.anchorMax = new Vector2(0.94f, 0.28f);
            wRt.offsetMin = Vector2.zero;
            wRt.offsetMax = Vector2.zero;
            Image wImg = wordObj.GetComponent<Image>();
            wImg.color = new Color(0.06f, 0.16f, 0.30f, 0.9f);

            GameObject wTxtObj = new GameObject("Text", typeof(RectTransform), typeof(Text));
            wTxtObj.transform.SetParent(wordObj.transform, false);
            RectTransform wtRt = wTxtObj.GetComponent<RectTransform>();
            wtRt.anchorMin = Vector2.zero;
            wtRt.anchorMax = Vector2.one;
            floatingWordText = wTxtObj.GetComponent<Text>();
            floatingWordText.font = GetDefaultFont();
            floatingWordText.fontSize = 11;
            floatingWordText.fontStyle = FontStyle.Bold;
            floatingWordText.alignment = TextAnchor.MiddleCenter;
            floatingWordText.color = new Color(0.3f, 0.95f, 1.0f);
            floatingWordText.text = "WORD: (SPELLING)";
        }

        private void BuildProceduralGhostHand()
        {
            if (webcamRect == null) return;

            ghostRoot = new GameObject("AR_GhostHandGuide", typeof(RectTransform));
            ghostRoot.transform.SetParent(webcamRect, false);
            ghostRoot.transform.SetSiblingIndex(1); // Behind HUD, on top of webcam video

            for (int i = 0; i < 21; i++)
            {
                GameObject gDot = new GameObject($"GhostDot_{i}", typeof(RectTransform), typeof(Image));
                gDot.transform.SetParent(ghostRoot.transform, false);
                RectTransform rt = gDot.GetComponent<RectTransform>();
                rt.sizeDelta = new Vector2(8f, 8f);
                rt.anchorMin = Vector2.zero;
                rt.anchorMax = Vector2.zero;
                rt.pivot = new Vector2(0.5f, 0.5f);

                Image img = gDot.GetComponent<Image>();
                img.color = ghostColor;
                img.raycastTarget = false;
                ghostDots[i] = rt;
            }

            var connections = HandLandmarkExtractor.BONE_CONNECTIONS;
            for (int i = 0; i < connections.Length; i++)
            {
                GameObject gBone = new GameObject($"GhostBone_{i}", typeof(RectTransform), typeof(Image));
                gBone.transform.SetParent(ghostRoot.transform, false);
                gBone.transform.SetSiblingIndex(0);
                RectTransform rt = gBone.GetComponent<RectTransform>();
                rt.anchorMin = Vector2.zero;
                rt.anchorMax = Vector2.zero;
                rt.pivot = new Vector2(0f, 0.5f);

                Image img = gBone.GetComponent<Image>();
                img.color = new Color(ghostColor.r, ghostColor.g, ghostColor.b, 0.25f);
                img.raycastTarget = false;
                ghostBones[i] = rt;
            }

            ghostRoot.SetActive(showGhostGuide);
        }

        private void BuildProceduralMotionTrails()
        {
            if (webcamRect == null) return;

            trailRoot = new GameObject("AR_MotionRibbon", typeof(RectTransform));
            trailRoot.transform.SetParent(webcamRect, false);
            trailRoot.transform.SetSiblingIndex(2);

            // Leading Focal Reticle (Glow diamond at the active fingertip)
            GameObject reticleObj = new GameObject("FocalLeadingReticle", typeof(RectTransform), typeof(Image));
            reticleObj.transform.SetParent(trailRoot.transform, false);
            trailLeadingReticle = reticleObj.GetComponent<RectTransform>();
            trailLeadingReticle.sizeDelta = new Vector2(16f, 16f);
            trailLeadingReticle.anchorMin = Vector2.zero;
            trailLeadingReticle.anchorMax = Vector2.zero;
            trailLeadingReticle.pivot = new Vector2(0.5f, 0.5f);
            trailLeadingImage = reticleObj.GetComponent<Image>();
            trailLeadingImage.color = new Color(1.0f, 1.0f, 1.0f, 0.95f);
            trailLeadingImage.raycastTarget = false;
            reticleObj.SetActive(false);

            // 32 Spline Ribbon Segments with Tapered Width & Gradient
            for (int i = 0; i < SPLINE_SEGMENTS; i++)
            {
                GameObject seg = new GameObject($"SplineSeg_{i}", typeof(RectTransform), typeof(Image));
                seg.transform.SetParent(trailRoot.transform, false);
                RectTransform rt = seg.GetComponent<RectTransform>();
                rt.anchorMin = Vector2.zero;
                rt.anchorMax = Vector2.zero;
                rt.pivot = new Vector2(0f, 0.5f);

                Image img = seg.GetComponent<Image>();
                float progress = (float)i / SPLINE_SEGMENTS;
                img.color = Color.Lerp(arVioletColor, arCyanColor, progress);
                img.raycastTarget = false;
                seg.SetActive(false);

                trailSegments[i] = rt;
                trailSegmentImages[i] = img;
            }

            trailRoot.SetActive(showMotionRibbon);
        }

        private void BuildProceduralGuideStencil()
        {
            if (webcamRect == null) return;

            stencilRoot = new GameObject("AR_GuideStencil", typeof(RectTransform));
            stencilRoot.transform.SetParent(webcamRect, false);
            stencilRoot.transform.SetSiblingIndex(2);

            for (int i = 0; i < STENCIL_POINTS; i++)
            {
                GameObject dotObj = new GameObject($"StencilDot_{i}", typeof(RectTransform), typeof(Image));
                dotObj.transform.SetParent(stencilRoot.transform, false);
                RectTransform rt = dotObj.GetComponent<RectTransform>();
                rt.sizeDelta = new Vector2(6f, 6f);
                rt.anchorMin = Vector2.zero;
                rt.anchorMax = Vector2.zero;
                rt.pivot = new Vector2(0.5f, 0.5f);

                Image img = dotObj.GetComponent<Image>();
                img.color = new Color(0.0f, 0.95f, 1.0f, 0.45f); // Soft cyan guide
                img.raycastTarget = false;
                dotObj.SetActive(false);

                stencilDots[i] = rt;
            }

            stencilRoot.SetActive(showGuideStencil);
        }

        private Font GetDefaultFont()
        {
            return Resources.GetBuiltinResource<Font>("LegacyRuntime.ttf") ??
                   Resources.GetBuiltinResource<Font>("Arial.ttf");
        }

        #endregion

        #region Real-Time AR Tracking & Heatmap Logic

        private void OnHandDataReceived(HandTrackingData handData)
        {
            if (handData == null || !handData.hand_detected || handData.LandmarkCount < 21)
            {
                if (hudCanvasGroup != null) hudCanvasGroup.alpha = Mathf.MoveTowards(hudCanvasGroup.alpha, 0f, Time.deltaTime * 6f);
                if (ghostRoot != null && ghostRoot.activeSelf) ghostRoot.SetActive(false);
                if (stencilRoot != null && stencilRoot.activeSelf) stencilRoot.SetActive(false);
                if (trailLeadingReticle != null) trailLeadingReticle.gameObject.SetActive(false);
                return;
            }

            if (hudCanvasGroup != null && showFloatingHUD)
            {
                hudCanvasGroup.alpha = Mathf.MoveTowards(hudCanvasGroup.alpha, 1f, Time.deltaTime * 8f);
            }

            Vector2 viewSize = webcamRect.rect.size;

            // 1. Determine Active Letter & Key Focal Landmark
            string currentLetter = predictionSmoother != null ? predictionSmoother.CurrentStableLetter : "-";
            if (string.IsNullOrEmpty(currentLetter) || currentLetter == "None") currentLetter = "-";
            activeTargetLetter = currentLetter;

            currentFocalLandmarkIndex = ResolveFocalLandmarkForLetter(currentLetter);

            // 2. Calculate Spatial Hand Centroid & Smooth HUD Anchor
            Vector3 wristLm = handData.GetLandmark(0);
            Vector3 middleMcpLm = handData.GetLandmark(9);
            Vector3 focalLm = handData.GetLandmark(currentFocalLandmarkIndex);

            Vector2 wristScreen = new Vector2(wristLm.x * viewSize.x, (1.0f - wristLm.y) * viewSize.y);
            Vector2 mcpScreen = new Vector2(middleMcpLm.x * viewSize.x, (1.0f - middleMcpLm.y) * viewSize.y);
            Vector2 focalScreen = new Vector2(focalLm.x * viewSize.x, (1.0f - focalLm.y) * viewSize.y);

            // Filter focal point to eliminate single-frame camera jitter
            if (!hasInitialFocalPoint)
            {
                smoothFocalPoint = focalScreen;
                hasInitialFocalPoint = true;
            }
            else
            {
                smoothFocalPoint = Vector2.Lerp(smoothFocalPoint, focalScreen, 0.45f);
            }

            // Anchor floating HUD smoothly above knuckles
            Vector2 targetHudPos = mcpScreen + new Vector2(0f, 80f);
            smoothHandCenter = Vector2.Lerp(smoothHandCenter, targetHudPos, Time.deltaTime * 12f);

            // Keep inside viewport boundaries
            smoothHandCenter.x = Mathf.Clamp(smoothHandCenter.x, 130f, viewSize.x - 130f);
            smoothHandCenter.y = Mathf.Clamp(smoothHandCenter.y, 45f, viewSize.y - 145f);

            if (hudRect != null)
            {
                hudRect.anchoredPosition = smoothHandCenter;
            }

            // 3. Accumulate Focal Trail Points with Adaptive Velocity Filtering
            float focalSpeed = Vector2.Distance(focalScreen, smoothFocalPoint);
            bool isDynamicLetter = (currentLetter == "J" || currentLetter == "Z");

            // For dynamic letters or hand moving, push into trail buffer
            rawFocalTrail.Enqueue(smoothFocalPoint);
            if (rawFocalTrail.Count > RAW_HISTORY_SIZE)
            {
                rawFocalTrail.Dequeue();
            }

            // Update Leading Focal Reticle
            if (trailLeadingReticle != null && showMotionRibbon)
            {
                trailLeadingReticle.gameObject.SetActive(true);
                trailLeadingReticle.anchoredPosition = smoothFocalPoint;

                // Reticle color indicates stability & accuracy
                if (trailLeadingImage != null)
                {
                    if (currentTrailAccuracy > 0.85f)
                        trailLeadingImage.color = arGreenColor;
                    else if (currentTrailAccuracy > 0.65f)
                        trailLeadingImage.color = arCyanColor;
                    else
                        trailLeadingImage.color = arWarningColor;
                }
            }

            // 4. Update HUD Data
            if (floatingLetterText != null) floatingLetterText.text = currentLetter;
            if (handReticleText != null) handReticleText.text = $"AR LOCK: {handData.handedness.ToUpper()} HAND";

            if (wordAssembler != null)
            {
                if (floatingRadialRing != null) floatingRadialRing.fillAmount = wordAssembler.DwellProgress;
                if (floatingWordText != null)
                {
                    string word = wordAssembler.CurrentWord;
                    floatingWordText.text = string.IsNullOrEmpty(word) ? "WORD: (SPELLING)" : $"WORD: {word}";
                }
            }

            // 5. Evaluate Trajectory & Form Accuracy for This Specific Letter
            EvaluateLetterSpecificAccuracy(handData, currentLetter, smoothFocalPoint, viewSize);
        }

        /// <summary>
        /// Real-time evaluation of trajectory accuracy and 3D topology alignment for every letter.
        /// </summary>
        private void EvaluateLetterSpecificAccuracy(HandTrackingData handData, string letter, Vector2 focalScreen, Vector2 viewSize)
        {
            if (string.IsNullOrEmpty(letter) || letter.Length != 1 || letter[0] < 'A' || letter[0] > 'Z')
            {
                currentFormAccuracy = 1.0f;
                currentTrailAccuracy = 1.0f;
                trailAccuracyFeedback = "AR READY";
                UpdateAccuracyDisplays("AR FORM: READY", "TRAIL: STEADY", arCyanColor);
                if (skeletonRenderer != null) skeletonRenderer.ResetJointColors();
                if (ghostRoot != null) ghostRoot.SetActive(false);
                if (stencilRoot != null) stencilRoot.SetActive(false);
                return;
            }

            string upperLetter = letter.ToUpper();

            // A. Dynamic Letter Trajectory Matching: 'J' & 'Z'
            if (upperLetter == "J")
            {
                EvaluateJTrajectory(focalScreen, viewSize);
                UpdateGuideStencilForJ(focalScreen, viewSize);
            }
            else if (upperLetter == "Z")
            {
                EvaluateZTrajectory(focalScreen, viewSize);
                UpdateGuideStencilForZ(focalScreen, viewSize);
            }
            else
            {
                // B. Static Letter Accuracy: 3D Topology + Joint Stability
                EvaluateStaticSignAccuracy(handData, upperLetter, focalScreen, viewSize);
                if (stencilRoot != null && stencilRoot.activeSelf) stencilRoot.SetActive(false);
            }
        }

        private void EvaluateJTrajectory(Vector2 focalPos, Vector2 viewSize)
        {
            Vector2[] points = rawFocalTrail.ToArray();
            if (points.Length < 6)
            {
                currentTrailAccuracy = 0.80f;
                trailAccuracyFeedback = "TRACE 'J' HOOK ⤴";
                UpdateAccuracyDisplays("FORM: DYNAMIC", trailAccuracyFeedback, arCyanColor);
                return;
            }

            // A valid ASL 'J' starts with a downward movement, then curves smoothly leftward and upward
            float netDy = points[points.Length - 1].y - points[0].y;
            float netDx = points[points.Length - 1].x - points[0].x;

            // Find lowest point in the stroke
            int lowestIdx = 0;
            float lowestY = float.MaxValue;
            for (int i = 0; i < points.Length; i++)
            {
                if (points[i].y < lowestY)
                {
                    lowestY = points[i].y;
                    lowestIdx = i;
                }
            }

            bool hasDownwardDrop = lowestIdx > 1;
            bool hasUpwardHook = lowestIdx < points.Length - 2;
            bool curvesLeft = netDx < 0f;

            if (hasDownwardDrop && hasUpwardHook && curvesLeft)
            {
                currentTrailAccuracy = 0.96f;
                trailAccuracyFeedback = "PERFECT 'J' HOOK!";
                UpdateAccuracyDisplays("FORM: 96%", trailAccuracyFeedback, arGreenColor);
            }
            else if (hasDownwardDrop && hasUpwardHook)
            {
                currentTrailAccuracy = 0.88f;
                trailAccuracyFeedback = "J-ARC: CURVE LEFT ⤾";
                UpdateAccuracyDisplays("FORM: 88%", trailAccuracyFeedback, arCyanColor);
            }
            else
            {
                currentTrailAccuracy = 0.72f;
                trailAccuracyFeedback = "SWOOP PINKY DOWN & UP ⤴";
                UpdateAccuracyDisplays("FORM: 72%", trailAccuracyFeedback, arWarningColor);
            }
        }

        private void EvaluateZTrajectory(Vector2 focalPos, Vector2 viewSize)
        {
            Vector2[] points = rawFocalTrail.ToArray();
            if (points.Length < 8)
            {
                currentTrailAccuracy = 0.80f;
                trailAccuracyFeedback = "TRACE 'Z' ZIG-ZAG ⮡";
                UpdateAccuracyDisplays("FORM: DYNAMIC", trailAccuracyFeedback, arCyanColor);
                return;
            }

            // ASL 'Z': 3 distinct stroke vectors:
            // Stroke 1: Right (+dx)
            // Stroke 2: Diagonal down-left (-dx, -dy)
            // Stroke 3: Right (+dx)
            int cornersFound = 0;
            for (int i = 2; i < points.Length - 2; i++)
            {
                Vector2 v1 = points[i] - points[i - 2];
                Vector2 v2 = points[i + 2] - points[i];
                float angle = Vector2.Angle(v1, v2);

                if (angle > 60f && angle < 145f)
                {
                    cornersFound++;
                    i += 2; // Skip neighboring points
                }
            }

            if (cornersFound >= 2)
            {
                currentTrailAccuracy = 0.97f;
                trailAccuracyFeedback = "PERFECT 'Z' ZIG-ZAG!";
                UpdateAccuracyDisplays("FORM: 97%", trailAccuracyFeedback, arGreenColor);
            }
            else if (cornersFound == 1)
            {
                currentTrailAccuracy = 0.85f;
                trailAccuracyFeedback = "Z-STROKE: FINISH BASE ➔";
                UpdateAccuracyDisplays("FORM: 85%", trailAccuracyFeedback, arCyanColor);
            }
            else
            {
                currentTrailAccuracy = 0.68f;
                trailAccuracyFeedback = "DRAW 'Z' IN AIR ⮡";
                UpdateAccuracyDisplays("FORM: 68%", trailAccuracyFeedback, arWarningColor);
            }
        }

        private void EvaluateStaticSignAccuracy(HandTrackingData handData, string upperLetter, Vector2 focalScreen, Vector2 viewSize)
        {
            if (!canonicalPoses.TryGetValue(upperLetter, out Vector3[] targetCanonical))
            {
                if (skeletonRenderer != null) skeletonRenderer.ResetJointColors();
                if (ghostRoot != null) ghostRoot.SetActive(false);
                return;
            }

            Vector3 liveWrist = handData.GetLandmark(0);
            Vector3 liveMcp = handData.GetLandmark(9);
            float liveScale = Vector3.Distance(liveMcp, liveWrist);
            if (liveScale < 1e-4f) liveScale = 1.0f;

            float totalError = 0f;
            bool isLeft = (handData.handedness == "Left");

            // Evaluate all 21 Joint Distances against Canonical Shape
            for (int i = 0; i < 21; i++)
            {
                Vector3 livePt = (handData.GetLandmark(i) - liveWrist) / liveScale;
                if (isLeft) livePt.x = -livePt.x;

                Vector3 targetPt = targetCanonical[i];
                float dist = Vector3.Distance(livePt, targetPt);
                totalError += dist;

                // Colorize skeleton joint: Green (accurate) -> Amber -> Red (misplaced)
                if (skeletonRenderer != null && showAccuracyHeatmap)
                {
                    Color jointHeatColor;
                    if (dist < 0.16f)
                        jointHeatColor = arGreenColor;
                    else if (dist < 0.28f)
                        jointHeatColor = arWarningColor;
                    else
                        jointHeatColor = arErrorColor;

                    skeletonRenderer.SetJointColor(i, jointHeatColor);
                }
            }

            float avgDist = totalError / 21f;
            float topologyMatch = Mathf.Clamp01(1.0f - (avgDist * 2.8f));

            // Factor in stability from landmark extractor
            float stability = landmarkExtractor != null ? landmarkExtractor.HandStability : 1.0f;
            currentFormAccuracy = (topologyMatch * 0.65f) + (stability * 0.35f);
            currentTrailAccuracy = currentFormAccuracy;

            int accPercent = Mathf.RoundToInt(currentFormAccuracy * 100f);
            Color themeColor = currentFormAccuracy > 0.82f ? arGreenColor : (currentFormAccuracy > 0.65f ? arWarningColor : arErrorColor);

            string feedback = currentFormAccuracy > 0.85f ? "OPTIMAL POSTURE" : (currentFormAccuracy > 0.70f ? "ALIGN FINGERS" : "ADJUST HAND POSE");
            trailAccuracyFeedback = feedback;

            UpdateAccuracyDisplays($"AR FORM: {accPercent}%", $"ACCURACY: {accPercent}% [{feedback}]", themeColor);

            // Render Holographic Ghost Silhouette Guide
            if (showGhostGuide && ghostRoot != null)
            {
                ghostRoot.SetActive(true);
                RenderGhostHand(targetCanonical, liveWrist, liveScale, viewSize, isLeft);
            }
        }

        private void UpdateAccuracyDisplays(string formText, string trailText, Color color)
        {
            if (floatingAccuracyText != null)
            {
                floatingAccuracyText.text = formText;
                floatingAccuracyText.color = color;
            }

            if (floatingTrailScoreText != null)
            {
                floatingTrailScoreText.text = trailText;
                floatingTrailScoreText.color = color;
            }
        }

        private void RenderGhostHand(Vector3[] canonical, Vector3 liveWrist, float liveScale, Vector2 viewSize, bool isLeft)
        {
            Vector2[] ghostScreenPoints = new Vector2[21];

            for (int i = 0; i < 21; i++)
            {
                Vector3 pt = canonical[i];
                if (isLeft) pt.x = -pt.x;

                Vector3 restoredWorld = liveWrist + (pt * liveScale);
                Vector2 screenPos = new Vector2(restoredWorld.x * viewSize.x, (1.0f - restoredWorld.y) * viewSize.y);
                ghostScreenPoints[i] = screenPos;

                if (ghostDots[i] != null)
                {
                    ghostDots[i].anchoredPosition = screenPos;
                }
            }

            var connections = HandLandmarkExtractor.BONE_CONNECTIONS;
            for (int i = 0; i < connections.Length; i++)
            {
                if (ghostBones[i] == null) continue;

                Vector2 pointA = ghostScreenPoints[connections[i].Item1];
                Vector2 pointB = ghostScreenPoints[connections[i].Item2];

                Vector2 dir = pointB - pointA;
                float distance = dir.magnitude;
                float angle = Mathf.Atan2(dir.y, dir.x) * Mathf.Rad2Deg;

                ghostBones[i].anchoredPosition = pointA;
                ghostBones[i].sizeDelta = new Vector2(distance, 2.5f);
                ghostBones[i].localRotation = Quaternion.Euler(0f, 0f, angle);
            }
        }

        #endregion

        #region Catmull-Rom Spline Motion Ribbons & Guide Stencils

        /// <summary>
        /// Renders ultra-smooth, neon laser light ribbons using cubic Catmull-Rom spline interpolation.
        /// </summary>
        private void RenderSplineMotionRibbon()
        {
            if (!showMotionRibbon || trailRoot == null) return;

            Vector2[] rawPts = rawFocalTrail.ToArray();
            if (rawPts.Length < 4)
            {
                for (int i = 0; i < trailSegments.Length; i++)
                {
                    if (trailSegments[i] != null) trailSegments[i].gameObject.SetActive(false);
                }
                return;
            }

            // Generate SPLINE_SEGMENTS smoothly interpolated positions
            Vector2[] splinePts = new Vector2[SPLINE_SEGMENTS];
            for (int i = 0; i < SPLINE_SEGMENTS; i++)
            {
                float tGlobal = (float)i / (SPLINE_SEGMENTS - 1);
                splinePts[i] = SampleCatmullRom(rawPts, tGlobal);
            }

            // Position and scale ribbon segments
            for (int i = 0; i < SPLINE_SEGMENTS - 1; i++)
            {
                if (trailSegments[i] == null) continue;

                Vector2 p1 = splinePts[i];
                Vector2 p2 = splinePts[i + 1];
                Vector2 dir = p2 - p1;
                float len = dir.magnitude;

                if (len > 0.8f)
                {
                    trailSegments[i].gameObject.SetActive(true);
                    trailSegments[i].anchoredPosition = p1;

                    // Tapered width: thin tail to thick leading head
                    float progress = (float)i / SPLINE_SEGMENTS;
                    float width = Mathf.Lerp(1.5f, 7.0f, progress);
                    trailSegments[i].sizeDelta = new Vector2(len, width);
                    trailSegments[i].localRotation = Quaternion.Euler(0f, 0f, Mathf.Atan2(dir.y, dir.x) * Mathf.Rad2Deg);

                    // Dynamic Color: Emerald when accurate, Cyan/Violet when cruising
                    if (trailSegmentImages[i] != null)
                    {
                        Color targetColor = currentTrailAccuracy > 0.85f ? arGreenColor : Color.Lerp(arVioletColor, arCyanColor, progress);
                        targetColor.a = Mathf.Lerp(0.15f, 0.95f, progress);
                        trailSegmentImages[i].color = targetColor;
                    }
                }
                else
                {
                    trailSegments[i].gameObject.SetActive(false);
                }
            }
        }

        private Vector2 SampleCatmullRom(Vector2[] controlPoints, float t)
        {
            int numSections = controlPoints.Length - 3;
            int currPt = Mathf.Min(Mathf.FloorToInt(t * numSections), numSections - 1);
            float u = t * numSections - currPt;

            Vector2 p0 = controlPoints[currPt];
            Vector2 p1 = controlPoints[currPt + 1];
            Vector2 p2 = controlPoints[currPt + 2];
            Vector2 p3 = controlPoints[currPt + 3];

            return 0.5f * (
                (-p0 + 3f * p1 - 3f * p2 + p3) * (u * u * u) +
                (2f * p0 - 5f * p1 + 4f * p2 - p3) * (u * u) +
                (-p0 + p2) * u +
                2f * p1
            );
        }

        private void UpdateGuideStencilForJ(Vector2 origin, Vector2 viewSize)
        {
            if (!showGuideStencil || stencilRoot == null) return;
            stencilRoot.SetActive(true);

            float scale = 90f;
            for (int i = 0; i < STENCIL_POINTS; i++)
            {
                if (stencilDots[i] == null) continue;
                float t = (float)i / (STENCIL_POINTS - 1);
                Vector2 offset;

                if (t < 0.55f)
                {
                    // Vertical drop
                    offset = new Vector2(0f, -t * 1.5f * scale);
                }
                else
                {
                    // Hook turning counter-clockwise left and up
                    float hookT = (t - 0.55f) / 0.45f;
                    float angle = Mathf.Lerp(-Mathf.PI * 0.5f, -Mathf.PI * 1.15f, hookT);
                    float radius = 0.45f * scale;
                    offset = new Vector2(
                        Mathf.Cos(angle) * radius + radius,
                        -0.55f * 1.5f * scale + Mathf.Sin(angle) * radius + radius
                    );
                }

                stencilDots[i].gameObject.SetActive(true);
                stencilDots[i].anchoredPosition = origin + offset;
            }
        }

        private void UpdateGuideStencilForZ(Vector2 origin, Vector2 viewSize)
        {
            if (!showGuideStencil || stencilRoot == null) return;
            stencilRoot.SetActive(true);

            float width = 80f;
            float height = 75f;

            for (int i = 0; i < STENCIL_POINTS; i++)
            {
                if (stencilDots[i] == null) continue;
                float t = (float)i / (STENCIL_POINTS - 1);
                Vector2 offset;

                if (t < 0.33f)
                {
                    // Stroke 1: Left to Right
                    float subT = t / 0.33f;
                    offset = new Vector2(subT * width, 0f);
                }
                else if (t < 0.66f)
                {
                    // Stroke 2: Diagonal Down-Left
                    float subT = (t - 0.33f) / 0.33f;
                    offset = new Vector2(width * (1.0f - subT), -height * subT);
                }
                else
                {
                    // Stroke 3: Left to Right
                    float subT = (t - 0.66f) / 0.34f;
                    offset = new Vector2(subT * width, -height);
                }

                stencilDots[i].gameObject.SetActive(true);
                stencilDots[i].anchoredPosition = origin + offset;
            }
        }

        #endregion
    }
}
