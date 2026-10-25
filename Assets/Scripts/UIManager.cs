using System;
using System.Collections;
using UnityEngine;
using UnityEngine.UI;

namespace ASLSignDetection
{
    /// <summary>
    /// Manages the complete AR & Machine Learning Presentation UI.
    /// Connects webcam controls, detection status, big letter display, confidence bars,
    /// dwell commit progress, assembled word & sentence output, smart suggestions,
    /// real-time trail accuracy meters, and voice readout.
    /// Automatically applies futuristic cyber / glassmorphic AR styling at runtime.
    /// </summary>
    public class UIManager : MonoBehaviour
    {
        [Header("Subsystem References")]
        [SerializeField] private WebcamManager webcamManager;
        [SerializeField] private MediaPipeManager mediaPipeManager;
        [SerializeField] private HandLandmarkExtractor landmarkExtractor;
        [SerializeField] private SignLanguageClassifier classifier;
        [SerializeField] private PredictionSmoother smoother;
        [SerializeField] private HandSkeletonRenderer skeletonRenderer;
        [SerializeField] private WordAssembler wordAssembler;

        [Header("Display Elements - Sign Recognition")]
        [Tooltip("Displays the big detected letter ('A', 'B', ... 'Z' or '-').")]
        [SerializeField] private Text detectedLetterText;

        [Tooltip("Displays the confidence percentage ('96%').")]
        [SerializeField] private Text confidenceText;

        [Tooltip("Visual fill bar representing confidence level (0 to 1).")]
        [SerializeField] private Image confidenceBar;

        [Tooltip("Displays real-time gesture & trail trajectory accuracy ('Trail Accuracy: 97%').")]
        [SerializeField] private Text trailAccuracyText;

        [Tooltip("Visual fill bar representing hand trail / posture accuracy.")]
        [SerializeField] private Image trailAccuracyBar;

        [Tooltip("Displays system and hand detection status ('Hand Detected', etc.).")]
        [SerializeField] private Text statusText;

        [Tooltip("Displays real-time camera and processing framerate.")]
        [SerializeField] private Text fpsText;

        [Tooltip("Displays neural inference latency in milliseconds.")]
        [SerializeField] private Text latencyText;

        [Tooltip("Displays detected handedness ('Right Hand' / 'Left Hand').")]
        [SerializeField] private Text handednessText;

        [Header("Display Elements - Word & Sentence Output")]
        [Tooltip("Displays the assembled word merged from detected letters (e.g. 'HELLO').")]
        [SerializeField] private Text formedWordText;

        [Tooltip("Displays the accumulated sentence history (e.g. 'HELLO WORLD').")]
        [SerializeField] private Text formedSentenceText;

        [Tooltip("Visual fill bar showing dwell commit progress towards the next letter.")]
        [SerializeField] private Image commitProgressBar;

        [Tooltip("Text status of the dwell commit timer (e.g. 'Holding H: 75%').")]
        [SerializeField] private Text commitProgressText;

        [Header("Smart Word Suggestions")]
        [Tooltip("Buttons displaying the top smart word suggestions.")]
        [SerializeField] private Button[] suggestionButtons;
        [SerializeField] private Text[] suggestionTexts;
        [SerializeField] private GameObject suggestionContainer;

        [Header("Camera & Sign Controls")]
        [SerializeField] private Button startCameraButton;
        [SerializeField] private Button stopCameraButton;
        [SerializeField] private Button clearSignButton;
        [SerializeField] private Toggle debugSkeletonToggle;

        [Header("Word Assembly Buttons")]
        [SerializeField] private Button spaceButton;
        [SerializeField] private Button backspaceButton;
        [SerializeField] private Button clearWordButton;
        [SerializeField] private Button clearAllButton;
        [SerializeField] private Button speakButton;
        [SerializeField] private Button commitNowButton;

        [Header("Debug Panel")]
        [SerializeField] private GameObject debugPanel;
        [SerializeField] private Text debugCoordinatesText;

        [Header("AR Spatial HUD")]
        [SerializeField] private ARSpatialHandHUD arSpatialHUD;

        [Header("Styling Colors")]
        [SerializeField] private Color confidentColor = new Color(0.10f, 0.98f, 0.50f); // Vivid Emerald #10F980
        [SerializeField] private Color lowConfidenceColor = new Color(1.0f, 0.72f, 0.18f); // Warm Amber #FFB82E
        [SerializeField] private Color unknownColor = new Color(0.55f, 0.65f, 0.80f); // Slate Cyan #8CA6CC
        [SerializeField] private Color neonCyanColor = new Color(0.0f, 0.94f, 1.0f); // Bright Cyan #00F0FF

        // FPS and Timing calculation
        private float frameCount = 0;
        private float dt = 0f;
        private float fps = 30f;
        private Coroutine pulseCoroutine;
        private bool hasStyledUI = false;

        private void Start()
        {
            // Bind Camera & Base Buttons
            if (startCameraButton != null)
                startCameraButton.onClick.AddListener(OnStartCameraClicked);

            if (stopCameraButton != null)
                stopCameraButton.onClick.AddListener(OnStopCameraClicked);

            if (clearSignButton != null)
                clearSignButton.onClick.AddListener(OnClearSignClicked);

            if (debugSkeletonToggle != null)
            {
                debugSkeletonToggle.onValueChanged.AddListener(OnSkeletonToggleChanged);
                if (skeletonRenderer != null)
                    debugSkeletonToggle.isOn = skeletonRenderer.ShowSkeleton;
            }

            // Bind Word Assembly Buttons
            if (spaceButton != null)
                spaceButton.onClick.AddListener(OnSpaceClicked);

            if (backspaceButton != null)
                backspaceButton.onClick.AddListener(OnBackspaceClicked);

            if (clearWordButton != null)
                clearWordButton.onClick.AddListener(OnClearWordClicked);

            if (clearAllButton != null)
                clearAllButton.onClick.AddListener(OnClearAllClicked);

            if (speakButton != null)
                speakButton.onClick.AddListener(OnSpeakClicked);

            if (commitNowButton != null)
                commitNowButton.onClick.AddListener(OnCommitNowClicked);

            // Bind Event Listeners
            if (mediaPipeManager != null)
            {
                mediaPipeManager.OnHandDataReceived += OnHandDataReceived;
                mediaPipeManager.OnVideoFrameReceived += OnVideoFrameReceived;
                mediaPipeManager.OnStatusChanged += UpdateStatusText;
            }

            if (smoother != null)
            {
                smoother.OnStablePredictionChanged += OnStablePredictionChanged;
            }

            if (webcamManager != null)
            {
                webcamManager.OnCameraError += OnCameraError;
            }

            if (wordAssembler != null)
            {
                wordAssembler.OnWordUpdated += OnWordUpdated;
                wordAssembler.OnDwellProgressUpdated += OnDwellProgressUpdated;
                wordAssembler.OnLetterCommitted += OnLetterCommitted;

                if (wordAssembler.Predictor != null)
                {
                    wordAssembler.Predictor.OnSuggestionsUpdated += UpdateSuggestionsUI;
                }
            }

            EnsureSuggestionUI();
            EnsureARHUD();
            EnsureAccuracyMeters();
            ApplyModernCyberStyling();

            if (wordAssembler != null && wordAssembler.Predictor != null)
            {
                UpdateSuggestionsUI(wordAssembler.Predictor.CurrentSuggestions);
            }

            // Set Initial UI State
            ResetDisplay();
            UpdateStatusText("System Online • Press START CAMERA or [H] for AR HUD");
        }

        private void Update()
        {
            // FPS calculation
            frameCount++;
            dt += Time.unscaledDeltaTime;
            if (dt > 0.5f)
            {
                fps = frameCount / dt;
                frameCount = 0;
                dt = 0f;
                if (fpsText != null)
                {
                    fpsText.text = $"FPS: {Mathf.RoundToInt(fps)}";
                }
            }

            // Real-Time Trail & Pose Accuracy Sync from AR HUD
            UpdateTrailAccuracyMeter();

            // Word blinking cursor
            UpdateBlinkingCursor();
        }

        private void UpdateTrailAccuracyMeter()
        {
            if (arSpatialHUD == null) return;

            float trailAcc = arSpatialHUD.CurrentTrailAccuracy;
            int percent = Mathf.RoundToInt(trailAcc * 100f);

            if (trailAccuracyBar != null)
            {
                trailAccuracyBar.fillAmount = Mathf.MoveTowards(trailAccuracyBar.fillAmount, trailAcc, Time.deltaTime * 3f);
                trailAccuracyBar.color = trailAcc > 0.85f ? confidentColor : (trailAcc > 0.65f ? lowConfidenceColor : new Color(1f, 0.3f, 0.4f));
            }

            if (trailAccuracyText != null)
            {
                string feedback = arSpatialHUD.TrailAccuracyFeedback;
                trailAccuracyText.text = $"Trail Accuracy: {percent}% [{feedback}]";
                trailAccuracyText.color = trailAcc > 0.85f ? confidentColor : (trailAcc > 0.65f ? lowConfidenceColor : unknownColor);
            }
        }

        private void UpdateBlinkingCursor()
        {
            if (formedWordText == null || wordAssembler == null) return;

            string rawWord = wordAssembler.CurrentWord;
            if (string.IsNullOrEmpty(rawWord))
            {
                // Idle placeholder
                if (formedWordText.text == rawWord + "|" || formedWordText.text == rawWord)
                    formedWordText.text = "-";
                return;
            }

            bool cursorVisible = (Time.time % 0.8f) < 0.4f;
            string display = rawWord + (cursorVisible ? "<color=#00F0FF>|</color>" : " ");
            formedWordText.text = display;
        }

        private void OnHandDataReceived(HandTrackingData handData)
        {
            if (handData == null) return;

            bool detected = handData.hand_detected;

            // Update Handedness text
            if (handednessText != null)
            {
                handednessText.text = detected ? $"● {handData.handedness.ToUpper()} HAND" : "● NO HAND";
                handednessText.color = detected ? confidentColor : unknownColor;
            }

            // Notify word assembler of hand presence
            if (wordAssembler != null)
            {
                wordAssembler.SetHandDetected(detected);
            }

            // Update Skeleton Overlay
            if (skeletonRenderer != null)
            {
                skeletonRenderer.UpdateSkeleton(handData);
            }

            // If a valid hand is present, run the classification pipeline
            if (detected && landmarkExtractor != null && classifier != null && smoother != null)
            {
                float[] features = landmarkExtractor.ExtractFeatures(handData);
                if (features != null)
                {
                    ClassificationResult rawResult = classifier.Classify(features, handData);
                    smoother.ProcessFrame(rawResult);

                    if (latencyText != null)
                    {
                        latencyText.text = $"Latency: {rawResult.InferenceTimeMs:F2} ms";
                    }

                    // Update debug readout if panel active
                    if (debugPanel != null && debugPanel.activeSelf && debugCoordinatesText != null)
                    {
                        UpdateDebugReadout(handData, rawResult);
                    }
                }
            }
            else
            {
                // Hand lost: trigger unknown in smoother
                if (smoother != null)
                {
                    smoother.ProcessFrame(ClassificationResult.Unknown());
                }
            }
        }

        private void OnStablePredictionChanged(string letter, float confidence, bool isConfident)
        {
            if (detectedLetterText != null)
            {
                detectedLetterText.text = letter;
                detectedLetterText.color = isConfident ? confidentColor : (letter == "Unknown" ? unknownColor : lowConfidenceColor);
            }

            int confPercent = Mathf.RoundToInt(confidence * 100f);
            if (confidenceText != null)
            {
                confidenceText.text = $"Confidence: {confPercent}%";
                confidenceText.color = isConfident ? confidentColor : lowConfidenceColor;
            }

            if (confidenceBar != null)
            {
                confidenceBar.fillAmount = confidence;
                confidenceBar.color = isConfident ? confidentColor : lowConfidenceColor;
            }

            // Forward to WordAssembler to accumulate into word
            if (wordAssembler != null)
            {
                float stability = landmarkExtractor != null ? landmarkExtractor.HandStability : 1.0f;
                wordAssembler.ProcessLetterPrediction(letter, confidence, isConfident, stability);
            }
        }

        private void OnWordUpdated(string currentWord, string fullSentence)
        {
            if (formedWordText != null)
            {
                formedWordText.text = string.IsNullOrEmpty(currentWord) ? "-" : currentWord;
            }

            if (formedSentenceText != null)
            {
                string combinedSentence = "";
                if (!string.IsNullOrEmpty(fullSentence) && !string.IsNullOrEmpty(currentWord))
                {
                    combinedSentence = $"{fullSentence} {currentWord}";
                }
                else if (!string.IsNullOrEmpty(fullSentence))
                {
                    combinedSentence = fullSentence;
                }
                else if (!string.IsNullOrEmpty(currentWord))
                {
                    combinedSentence = currentWord;
                }

                formedSentenceText.text = string.IsNullOrEmpty(combinedSentence) ? "Sentence: (empty)" : $"Sentence: \"{combinedSentence}\"";
            }
        }

        private void OnDwellProgressUpdated(float progress, string candidateLetter, float currentSec, float targetSec)
        {
            if (commitProgressBar != null)
            {
                commitProgressBar.fillAmount = progress;
            }

            if (commitProgressText != null)
            {
                if (progress > 0f && !string.IsNullOrEmpty(candidateLetter))
                {
                    commitProgressText.text = $"Holding '{candidateLetter}' to commit: {currentSec:F1}s / {targetSec:F1}s ({Mathf.RoundToInt(progress * 100f)}%)";
                }
                else
                {
                    commitProgressText.text = $"Keep sign steady for {targetSec:F0}s or tap [Tab] to commit";
                }
            }
        }

        private void OnLetterCommitted(string letter)
        {
            if (formedWordText != null)
            {
                if (pulseCoroutine != null) StopCoroutine(pulseCoroutine);
                pulseCoroutine = StartCoroutine(PulseWordDisplay());
            }
        }

        private IEnumerator PulseWordDisplay()
        {
            if (formedWordText == null) yield break;
            Color original = formedWordText.color;
            formedWordText.color = new Color(1.0f, 0.95f, 0.35f); // Golden yellow flash
            yield return new WaitForSeconds(0.2f);
            formedWordText.color = original;
        }

        private void OnVideoFrameReceived(Texture2D frameTex)
        {
            if (webcamManager != null && webcamManager.DisplayImage != null && frameTex != null)
            {
                webcamManager.DisplayImage.texture = frameTex;
                webcamManager.DisplayImage.color = Color.white;
            }
        }

        private void OnStartCameraClicked()
        {
            Debug.Log("[UIManager] START CAMERA clicked.");
            if (webcamManager != null)
                webcamManager.StartCamera();

            if (mediaPipeManager != null)
            {
                mediaPipeManager.StartListening();
                mediaPipeManager.StartPythonBridge();
            }

            UpdateStatusText("Camera active • Detecting hand landmarks & 3D pose...");
        }

        private void OnStopCameraClicked()
        {
            Debug.Log("[UIManager] STOP CAMERA clicked.");
            if (webcamManager != null)
                webcamManager.StopCamera();

            if (mediaPipeManager != null)
                mediaPipeManager.StopPythonBridge();

            if (skeletonRenderer != null)
                skeletonRenderer.SetVisible(false);

            ResetDisplay();
            UpdateStatusText("Camera Stopped • Standby");
        }

        private void OnClearSignClicked()
        {
            Debug.Log("[UIManager] CLEAR SIGN clicked.");
            if (smoother != null)
                smoother.Clear();

            ResetDisplay();
        }

        private void OnSpaceClicked()
        {
            if (wordAssembler != null) wordAssembler.AddSpace();
        }

        private void OnBackspaceClicked()
        {
            if (wordAssembler != null) wordAssembler.Backspace();
        }

        private void OnClearWordClicked()
        {
            if (wordAssembler != null) wordAssembler.ClearWord();
        }

        private void OnClearAllClicked()
        {
            if (wordAssembler != null) wordAssembler.ClearAll();
        }

        private void OnSpeakClicked()
        {
            if (wordAssembler != null) wordAssembler.Speak();
        }

        private void OnCommitNowClicked()
        {
            if (wordAssembler != null) wordAssembler.CommitCurrentCandidateImmediately();
        }

        #region Smart Suggestion & Accuracy Meters Setup

        private void EnsureSuggestionUI()
        {
            if (suggestionButtons != null && suggestionButtons.Length >= 3 && suggestionButtons[0] != null)
                return;

            if (formedWordText == null || formedWordText.transform.parent == null)
                return;

            Transform parent = formedWordText.transform.parent;

            // Adjust formedWordText so it occupies the left side
            formedWordText.rectTransform.anchorMin = new Vector2(0.02f, 0.05f);
            formedWordText.rectTransform.anchorMax = new Vector2(0.48f, 0.65f);

            // Create suggestion container on the right side of FormedWordCard
            suggestionContainer = new GameObject("SmartSuggestionContainer");
            suggestionContainer.transform.SetParent(parent, false);
            RectTransform contRect = suggestionContainer.AddComponent<RectTransform>();
            contRect.anchorMin = new Vector2(0.48f, 0.05f);
            contRect.anchorMax = new Vector2(0.98f, 0.65f);
            contRect.offsetMin = Vector2.zero;
            contRect.offsetMax = Vector2.zero;

            suggestionButtons = new Button[3];
            suggestionTexts = new Text[3];

            float spacing = 0.02f;
            float btnWidth = (1f - (spacing * 2)) / 3f;

            for (int i = 0; i < 3; i++)
            {
                int index = i;
                float xMin = i * (btnWidth + spacing);
                float xMax = xMin + btnWidth;

                GameObject btnObj = new GameObject($"SuggestionBtn_{i + 1}");
                btnObj.transform.SetParent(suggestionContainer.transform, false);

                RectTransform bRect = btnObj.AddComponent<RectTransform>();
                bRect.anchorMin = new Vector2(xMin, 0.05f);
                bRect.anchorMax = new Vector2(xMax, 0.95f);
                bRect.offsetMin = Vector2.zero;
                bRect.offsetMax = Vector2.zero;

                Image bImg = btnObj.AddComponent<Image>();
                bImg.color = new Color(0.08f, 0.22f, 0.38f, 0.95f);

                Button btn = btnObj.AddComponent<Button>();
                ColorBlock cb = btn.colors;
                cb.normalColor = new Color(0.08f, 0.22f, 0.38f, 0.95f);
                cb.highlightedColor = new Color(0.15f, 0.45f, 0.75f, 1f);
                cb.pressedColor = new Color(0.0f, 0.65f, 0.95f, 1f);
                btn.colors = cb;
                btn.onClick.AddListener(() => OnSuggestionClicked(index));

                GameObject txtObj = new GameObject("Label");
                txtObj.transform.SetParent(btnObj.transform, false);
                RectTransform tRect = txtObj.AddComponent<RectTransform>();
                tRect.anchorMin = Vector2.zero;
                tRect.anchorMax = Vector2.one;
                tRect.offsetMin = Vector2.zero;
                tRect.offsetMax = Vector2.zero;

                Text t = txtObj.AddComponent<Text>();
                t.font = Resources.GetBuiltinResource<Font>("LegacyRuntime.ttf") ?? Resources.GetBuiltinResource<Font>("Arial.ttf");
                t.fontSize = 15;
                t.fontStyle = FontStyle.Bold;
                t.alignment = TextAnchor.MiddleCenter;
                t.color = new Color(0.3f, 0.95f, 1.0f);
                t.resizeTextForBestFit = true;
                t.resizeTextMinSize = 10;
                t.resizeTextMaxSize = 16;
                t.text = $"[{i + 1}]";

                suggestionButtons[i] = btn;
                suggestionTexts[i] = t;
            }
        }

        private void EnsureAccuracyMeters()
        {
            if (trailAccuracyText != null && trailAccuracyBar != null) return;
            if (confidenceBar == null || confidenceBar.transform.parent == null) return;

            Transform dashPanel = confidenceBar.transform.parent.parent;
            if (dashPanel == null) return;

            // If TrailAccuracyBar doesn't exist, create it below confidence bar
            if (trailAccuracyText == null)
            {
                GameObject accTextObj = new GameObject("TrailAccuracyText", typeof(RectTransform), typeof(Text));
                accTextObj.transform.SetParent(dashPanel, false);
                RectTransform rt = accTextObj.GetComponent<RectTransform>();
                rt.anchorMin = new Vector2(0.05f, 0.54f);
                rt.anchorMax = new Vector2(0.95f, 0.58f);
                rt.offsetMin = Vector2.zero;
                rt.offsetMax = Vector2.zero;

                trailAccuracyText = accTextObj.GetComponent<Text>();
                trailAccuracyText.font = Resources.GetBuiltinResource<Font>("LegacyRuntime.ttf") ?? Resources.GetBuiltinResource<Font>("Arial.ttf");
                trailAccuracyText.fontSize = 15;
                trailAccuracyText.fontStyle = FontStyle.Bold;
                trailAccuracyText.alignment = TextAnchor.MiddleCenter;
                trailAccuracyText.color = neonCyanColor;
                trailAccuracyText.text = "Trail Accuracy: 100% [READY]";
            }

            if (trailAccuracyBar == null)
            {
                // Background
                GameObject barBg = new GameObject("TrailAccuracyBarBg", typeof(RectTransform), typeof(Image));
                barBg.transform.SetParent(dashPanel, false);
                RectTransform bgRt = barBg.GetComponent<RectTransform>();
                bgRt.anchorMin = new Vector2(0.1f, 0.51f);
                bgRt.anchorMax = new Vector2(0.9f, 0.535f);
                bgRt.offsetMin = Vector2.zero;
                bgRt.offsetMax = Vector2.zero;

                Image bgImg = barBg.GetComponent<Image>();
                bgImg.color = new Color(0.12f, 0.18f, 0.28f, 1f);

                // Fill
                GameObject barFill = new GameObject("TrailAccuracyBarFill", typeof(RectTransform), typeof(Image));
                barFill.transform.SetParent(barBg.transform, false);
                RectTransform fillRt = barFill.GetComponent<RectTransform>();
                fillRt.anchorMin = Vector2.zero;
                fillRt.anchorMax = Vector2.one;
                fillRt.offsetMin = Vector2.zero;
                fillRt.offsetMax = Vector2.zero;

                trailAccuracyBar = barFill.GetComponent<Image>();
                trailAccuracyBar.type = Image.Type.Filled;
                trailAccuracyBar.fillMethod = Image.FillMethod.Horizontal;
                trailAccuracyBar.color = neonCyanColor;
                trailAccuracyBar.fillAmount = 1.0f;
            }
        }

        private void OnSuggestionClicked(int index)
        {
            if (wordAssembler != null)
            {
                wordAssembler.SelectSuggestionByIndex(index);
            }
        }

        private void UpdateSuggestionsUI(string[] suggestions)
        {
            if (suggestionButtons == null || suggestionTexts == null) return;

            for (int i = 0; i < suggestionButtons.Length; i++)
            {
                if (suggestionButtons[i] == null) continue;

                if (suggestions != null && i < suggestions.Length && !string.IsNullOrEmpty(suggestions[i]))
                {
                    suggestionButtons[i].gameObject.SetActive(true);
                    if (suggestionTexts[i] != null)
                    {
                        suggestionTexts[i].text = $"[{i + 1}] {suggestions[i]}";
                    }
                }
                else
                {
                    suggestionButtons[i].gameObject.SetActive(false);
                }
            }
        }

        private void EnsureARHUD()
        {
            if (arSpatialHUD == null)
            {
                arSpatialHUD = FindObjectOfType<ARSpatialHandHUD>();
                if (arSpatialHUD == null && skeletonRenderer != null)
                {
                    arSpatialHUD = skeletonRenderer.gameObject.AddComponent<ARSpatialHandHUD>();
                }
            }
        }

        #endregion

        #region Futuristic Cyber UI Beautification Engine

        /// <summary>
        /// Automatically enhances all visual elements with a sleek, high-contrast Cyberpunk / Glassmorphic AR aesthetic.
        /// </summary>
        public void ApplyModernCyberStyling()
        {
            if (hasStyledUI) return;
            hasStyledUI = true;

            try
            {
                // 1. Canvas Background: Deep Obsidian
                Transform bg = transform.root.Find("Background");
                if (bg != null)
                {
                    Image img = bg.GetComponent<Image>();
                    if (img != null) img.color = new Color(0.035f, 0.05f, 0.08f, 1f);
                }

                // 2. Header Bar: Glowing Cyan Accent & Cyber Titles
                Transform header = transform.root.Find("Header");
                if (header != null)
                {
                    Image hImg = header.GetComponent<Image>();
                    if (hImg != null) hImg.color = new Color(0.06f, 0.09f, 0.15f, 0.96f);

                    Transform title = header.Find("TitleText");
                    if (title != null)
                    {
                        Text t = title.GetComponent<Text>();
                        if (t != null)
                        {
                            t.text = "⚡ NEURAL SIGN AR • SPATIAL ASL TRANSLATOR";
                            t.color = neonCyanColor;
                        }
                    }

                    Transform sub = header.Find("SubtitleText");
                    if (sub != null)
                    {
                        Text st = sub.GetComponent<Text>();
                        if (st != null)
                        {
                            st.text = "Computer Vision • 21 Hand Landmarks • Sub-Millisecond Neural Classification • Adaptive Trajectory Splines";
                            st.color = new Color(0.6f, 0.75f, 0.9f);
                        }
                    }
                }

                // 3. Webcam Container: Cyan framing & badge
                Transform camPanel = transform.root.Find("WebcamContainer");
                if (camPanel != null)
                {
                    Image cpImg = camPanel.GetComponent<Image>();
                    if (cpImg != null) cpImg.color = new Color(0.02f, 0.03f, 0.05f, 1f);

                    Transform badge = camPanel.Find("CamBadge");
                    if (badge != null)
                    {
                        Image bImg = badge.GetComponent<Image>();
                        if (bImg != null) bImg.color = new Color(0.07f, 0.14f, 0.24f, 0.9f);

                        Text bt = badge.GetComponentInChildren<Text>();
                        if (bt != null)
                        {
                            bt.text = "◈ LIVE VISION & 21-JOINT AR MESH";
                            bt.color = neonCyanColor;
                        }
                    }
                }

                // 4. Sign Recognition Dashboard Panel
                Transform dashPanel = transform.root.Find("DashboardPanel");
                if (dashPanel != null)
                {
                    Image dImg = dashPanel.GetComponent<Image>();
                    if (dImg != null) dImg.color = new Color(0.06f, 0.09f, 0.16f, 0.95f);

                    Transform label = dashPanel.Find("SignLabel");
                    if (label != null)
                    {
                        Text lt = label.GetComponent<Text>();
                        if (lt != null)
                        {
                            lt.text = "⬡ REAL-TIME SIGN CLASSIFIER";
                            lt.color = new Color(0.75f, 0.88f, 1.0f);
                        }
                    }
                }

                // 5. Stylize Action Buttons with Cyber Icons & Distinct Colors
                StyleButton(startCameraButton, "▶ START CAMERA", new Color(0.06f, 0.72f, 0.40f));
                StyleButton(stopCameraButton, "■ STOP CAMERA", new Color(0.85f, 0.20f, 0.25f));
                StyleButton(clearSignButton, "↺ CLEAR SIGN", new Color(0.24f, 0.32f, 0.45f));

                StyleButton(spaceButton, "⎵ SPACE", new Color(0.08f, 0.48f, 0.85f));
                StyleButton(backspaceButton, "⌫ BACKSPACE", new Color(0.90f, 0.50f, 0.10f));
                StyleButton(speakButton, "🔊 SPEAK", new Color(0.55f, 0.25f, 0.95f));
                StyleButton(clearWordButton, "🗑 CLEAR", new Color(0.30f, 0.38f, 0.50f));
                StyleButton(clearAllButton, "✕ RESET ALL", new Color(0.80f, 0.18f, 0.18f));
                StyleButton(commitNowButton, "✓ COMMIT [Tab]", new Color(0.06f, 0.72f, 0.40f));

                // 6. Formed Word Card & Sentence Card Styling
                Transform wordConsole = transform.root.Find("WordAssemblyConsole");
                if (wordConsole != null)
                {
                    Image wcImg = wordConsole.GetComponent<Image>();
                    if (wcImg != null) wcImg.color = new Color(0.05f, 0.07f, 0.12f, 1f);

                    Transform tip = wordConsole.Find("ShortcutTip");
                    if (tip != null)
                    {
                        Text tipText = tip.GetComponent<Text>();
                        if (tipText != null)
                        {
                            tipText.text = "Hold sign 2s or [Tab] Commit • [1, 2, 3] Smart Word • [H] AR HUD • [G] Ghost Pose • [T] Motion Trails • [Y] Heatmap • [Enter] Speak";
                            tipText.color = new Color(0.45f, 0.65f, 0.85f);
                        }
                    }
                }
            }
            catch (Exception ex)
            {
                Debug.LogWarning("[UIManager] Note during UI beautification: " + ex.Message);
            }
        }

        private void StyleButton(Button btn, string label, Color color)
        {
            if (btn == null) return;
            Image img = btn.GetComponent<Image>();
            if (img != null) img.color = color;

            ColorBlock cb = btn.colors;
            cb.normalColor = color;
            cb.highlightedColor = color * 1.25f;
            cb.pressedColor = color * 0.80f;
            btn.colors = cb;

            Text txt = btn.GetComponentInChildren<Text>();
            if (txt != null && !string.IsNullOrEmpty(label))
            {
                txt.text = label;
                txt.fontStyle = FontStyle.Bold;
                txt.color = Color.white;
            }
        }

        #endregion

        private void OnSkeletonToggleChanged(bool isEnabled)
        {
            if (skeletonRenderer != null)
                skeletonRenderer.ShowSkeleton = isEnabled;
        }

        private void OnCameraError(string errorMsg)
        {
            UpdateStatusText($"Camera Error: {errorMsg}");
        }

        private void UpdateStatusText(string status)
        {
            if (statusText != null)
            {
                statusText.text = status;
            }
        }

        private void ResetDisplay()
        {
            if (detectedLetterText != null)
            {
                detectedLetterText.text = "-";
                detectedLetterText.color = unknownColor;
            }

            if (confidenceText != null)
            {
                confidenceText.text = "Confidence: 0%";
            }

            if (confidenceBar != null)
            {
                confidenceBar.fillAmount = 0f;
            }

            if (trailAccuracyText != null)
            {
                trailAccuracyText.text = "Trail Accuracy: 100% [READY]";
            }

            if (trailAccuracyBar != null)
            {
                trailAccuracyBar.fillAmount = 1f;
            }

            if (commitProgressBar != null)
            {
                commitProgressBar.fillAmount = 0f;
            }

            if (commitProgressText != null)
            {
                float targetSec = wordAssembler != null ? wordAssembler.CommitDwellTime : 5.0f;
                commitProgressText.text = $"Keep sign steady for {targetSec:F0}s or tap [Tab] to commit";
            }

            if (formedWordText != null)
            {
                formedWordText.text = (wordAssembler != null && !string.IsNullOrEmpty(wordAssembler.CurrentWord))
                    ? wordAssembler.CurrentWord : "-";
            }

            if (formedSentenceText != null)
            {
                formedSentenceText.text = (wordAssembler != null && !string.IsNullOrEmpty(wordAssembler.FullSentence))
                    ? $"Sentence: \"{wordAssembler.FullSentence}\"" : "Sentence: (empty)";
            }

            if (handednessText != null)
            {
                handednessText.text = "● NO HAND";
                handednessText.color = unknownColor;
            }
        }

        private void UpdateDebugReadout(HandTrackingData handData, ClassificationResult rawResult)
        {
            if (handData == null || handData.LandmarkCount < 21) return;

            Vector3 wrist = handData.GetLandmark(0);
            Vector3 indexTip = handData.GetLandmark(8);
            Vector3 thumbTip = handData.GetLandmark(4);

            debugCoordinatesText.text =
                $"Landmarks Detected: 21\n" +
                $"Wrist [0]: ({wrist.x:F2}, {wrist.y:F2}, {wrist.z:F2})\n" +
                $"Index Tip [8]: ({indexTip.x:F2}, {indexTip.y:F2}, {indexTip.z:F2})\n" +
                $"Thumb Tip [4]: ({thumbTip.x:F2}, {thumbTip.y:F2}, {thumbTip.z:F2})\n" +
                $"Raw Top-1: {rawResult.Letter} ({rawResult.Confidence * 100:F1}%)\n" +
                $"Latency: {rawResult.InferenceTimeMs:F2} ms";
        }

        private void OnDestroy()
        {
            if (mediaPipeManager != null)
            {
                mediaPipeManager.OnHandDataReceived -= OnHandDataReceived;
                mediaPipeManager.OnVideoFrameReceived -= OnVideoFrameReceived;
                mediaPipeManager.OnStatusChanged -= UpdateStatusText;
            }

            if (smoother != null)
            {
                smoother.OnStablePredictionChanged -= OnStablePredictionChanged;
            }

            if (webcamManager != null)
            {
                webcamManager.OnCameraError -= OnCameraError;
            }

            if (wordAssembler != null)
            {
                wordAssembler.OnWordUpdated -= OnWordUpdated;
                wordAssembler.OnDwellProgressUpdated -= OnDwellProgressUpdated;
                wordAssembler.OnLetterCommitted -= OnLetterCommitted;

                if (wordAssembler.Predictor != null)
                {
                    wordAssembler.Predictor.OnSuggestionsUpdated -= UpdateSuggestionsUI;
                }
            }
        }
    }
}
