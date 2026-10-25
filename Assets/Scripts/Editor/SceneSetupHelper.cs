#if UNITY_EDITOR
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.UI;

namespace ASLSignDetection.Editor
{
    /// <summary>
    /// Editor automation tool that builds, styles, and wires the entire
    /// ASL Real-Time Hand Sign Detection, AR Spatial Holograms & Word Formation presentation scene.
    /// Menu: "ASL Detection -> Setup Scene & UI"
    /// </summary>
    public static class SceneSetupHelper
    {
        [MenuItem("ASL Detection/Setup Scene & UI", false, 1)]
        public static void SetupCompleteScene()
        {
            // 1. Create a fresh scene
            var scene = EditorSceneManager.NewScene(NewSceneSetup.DefaultGameObjects, NewSceneMode.Single);

            // 2. Setup EventSystem
            GameObject eventSystemObj = GameObject.Find("EventSystem");
            if (eventSystemObj == null)
            {
                eventSystemObj = new GameObject("EventSystem", typeof(UnityEngine.EventSystems.EventSystem), typeof(UnityEngine.EventSystems.StandaloneInputModule));
            }

            // 3. Create Main Canvas
            GameObject canvasObj = new GameObject("ASL_Canvas", typeof(Canvas), typeof(CanvasScaler), typeof(GraphicRaycaster));
            Canvas canvas = canvasObj.GetComponent<Canvas>();
            canvas.renderMode = RenderMode.ScreenSpaceOverlay;

            CanvasScaler scaler = canvasObj.GetComponent<CanvasScaler>();
            scaler.uiScaleMode = CanvasScaler.ScaleMode.ScaleWithScreenSize;
            scaler.referenceResolution = new Vector2(1920, 1080);
            scaler.matchWidthOrHeight = 0.5f;

            // 4. Background Panel (Deep cyber obsidian)
            GameObject bgObj = CreateUIElement("Background", canvasObj.transform);
            RectTransform bgRect = bgObj.GetComponent<RectTransform>();
            SetFullStretch(bgRect);
            Image bgImg = bgObj.AddComponent<Image>();
            bgImg.color = new Color(0.035f, 0.05f, 0.08f, 1f); // Dark Obsidian #090D14

            // 5. Header Bar
            GameObject headerObj = CreateUIElement("Header", canvasObj.transform);
            RectTransform headerRect = headerObj.GetComponent<RectTransform>();
            SetAnchors(headerRect, 0f, 0.92f, 1f, 1f);
            Image headerImg = headerObj.AddComponent<Image>();
            headerImg.color = new Color(0.06f, 0.09f, 0.15f, 0.98f);

            Text titleText = CreateText("TitleText", headerObj.transform, "⚡ NEURAL SIGN AR • SPATIAL ASL TRANSLATOR", 27, FontStyle.Bold, TextAnchor.MiddleCenter);
            titleText.color = new Color(0.0f, 0.94f, 1.0f); // Bright Cyan
            SetAnchors(titleText.rectTransform, 0f, 0.35f, 1f, 0.95f);

            Text subtitleText = CreateText("SubtitleText", headerObj.transform, "Computer Vision • 21 Hand Landmarks • Sub-Millisecond Neural Classification • Adaptive Trajectory Splines • Vocal TTS", 14, FontStyle.Normal, TextAnchor.MiddleCenter);
            subtitleText.color = new Color(0.65f, 0.78f, 0.92f);
            SetAnchors(subtitleText.rectTransform, 0f, 0.05f, 1f, 0.40f);

            // Glowing header accent line
            GameObject headerLine = CreateUIElement("HeaderAccentLine", headerObj.transform);
            SetAnchors(headerLine.GetComponent<RectTransform>(), 0f, 0f, 1f, 0.035f);
            Image headerLineImg = headerLine.AddComponent<Image>();
            headerLineImg.color = new Color(0.0f, 0.94f, 1.0f, 0.85f);

            // 6. Left Upper Side: Webcam Container
            GameObject webcamPanel = CreateUIElement("WebcamContainer", canvasObj.transform);
            RectTransform webcamPanelRect = webcamPanel.GetComponent<RectTransform>();
            SetAnchors(webcamPanelRect, 0.02f, 0.30f, 0.58f, 0.90f);
            Image webcamPanelBg = webcamPanel.AddComponent<Image>();
            webcamPanelBg.color = new Color(0.02f, 0.03f, 0.05f, 1f);

            // Webcam Header Badge
            GameObject camBadge = CreateUIElement("CamBadge", webcamPanel.transform);
            SetAnchors(camBadge.GetComponent<RectTransform>(), 0.02f, 0.93f, 0.48f, 0.985f);
            Image camBadgeBg = camBadge.AddComponent<Image>();
            camBadgeBg.color = new Color(0.07f, 0.14f, 0.24f, 0.9f);
            Text camBadgeText = CreateText("BadgeLabel", camBadge.transform, "◈ LIVE VISION & 21-JOINT AR MESH", 13, FontStyle.Bold, TextAnchor.MiddleCenter);
            camBadgeText.color = new Color(0.0f, 0.94f, 1.0f);
            SetFullStretch(camBadgeText.rectTransform);

            // Webcam RawImage
            GameObject webcamFeedObj = CreateUIElement("WebcamFeed", webcamPanel.transform);
            RectTransform feedRect = webcamFeedObj.GetComponent<RectTransform>();
            SetAnchors(feedRect, 0.01f, 0.01f, 0.99f, 0.92f);
            RawImage feedImage = webcamFeedObj.AddComponent<RawImage>();
            feedImage.color = new Color(0.06f, 0.08f, 0.12f, 1f);

            // Aspect Ratio Fitter
            AspectRatioFitter aspectFitter = webcamFeedObj.AddComponent<AspectRatioFitter>();
            aspectFitter.aspectMode = AspectRatioFitter.AspectMode.FitInParent;
            aspectFitter.aspectRatio = 4f / 3f;

            // Hand Skeleton Overlay & AR Spatial Hologram HUD
            HandSkeletonRenderer skeletonRenderer = webcamFeedObj.AddComponent<HandSkeletonRenderer>();
            ARSpatialHandHUD arHUD = webcamFeedObj.AddComponent<ARSpatialHandHUD>();

            // 7. Right Upper Side: Live Sign Detection Dashboard Panel
            GameObject dashPanel = CreateUIElement("DashboardPanel", canvasObj.transform);
            RectTransform dashRect = dashPanel.GetComponent<RectTransform>();
            SetAnchors(dashRect, 0.60f, 0.30f, 0.98f, 0.90f);
            Image dashBg = dashPanel.AddComponent<Image>();
            dashBg.color = new Color(0.06f, 0.09f, 0.16f, 0.98f);

            // Dashboard Title
            Text signLabel = CreateText("SignLabel", dashPanel.transform, "⬡ REAL-TIME SIGN CLASSIFIER", 19, FontStyle.Bold, TextAnchor.MiddleCenter);
            signLabel.color = new Color(0.75f, 0.88f, 1.0f);
            SetAnchors(signLabel.rectTransform, 0.05f, 0.92f, 0.95f, 0.98f);

            // Big Detected Letter Display Card (Glowing frame)
            GameObject letterFrame = CreateUIElement("LetterFrame", dashPanel.transform);
            SetAnchors(letterFrame.GetComponent<RectTransform>(), 0.28f, 0.68f, 0.72f, 0.91f);
            Image letterFrameBg = letterFrame.AddComponent<Image>();
            letterFrameBg.color = new Color(0.04f, 0.07f, 0.14f, 0.9f);

            Text letterText = CreateText("DetectedLetterText", letterFrame.transform, "-", 100, FontStyle.Bold, TextAnchor.MiddleCenter);
            letterText.color = new Color(0.10f, 0.98f, 0.50f); // Vivid Emerald
            SetFullStretch(letterText.rectTransform);

            // Confidence Percentage Text
            Text confText = CreateText("ConfidenceText", dashPanel.transform, "Confidence: 0%", 17, FontStyle.Bold, TextAnchor.MiddleCenter);
            confText.color = Color.white;
            SetAnchors(confText.rectTransform, 0.05f, 0.62f, 0.95f, 0.67f);

            // Confidence Bar Background & Fill
            GameObject barBgObj = CreateUIElement("ConfidenceBarBg", dashPanel.transform);
            SetAnchors(barBgObj.GetComponent<RectTransform>(), 0.1f, 0.59f, 0.9f, 0.615f);
            Image barBgImg = barBgObj.AddComponent<Image>();
            barBgImg.color = new Color(0.12f, 0.18f, 0.28f, 1f);

            GameObject barFillObj = CreateUIElement("ConfidenceBarFill", barBgObj.transform);
            SetFullStretch(barFillObj.GetComponent<RectTransform>());
            Image barFillImg = barFillObj.AddComponent<Image>();
            barFillImg.color = new Color(0.10f, 0.98f, 0.50f);
            barFillImg.type = Image.Type.Filled;
            barFillImg.fillMethod = Image.FillMethod.Horizontal;
            barFillImg.fillAmount = 0f;

            // Trail & Posture Accuracy Text & Fill
            Text trailAccText = CreateText("TrailAccuracyText", dashPanel.transform, "Trail Accuracy: 100% [READY]", 15, FontStyle.Bold, TextAnchor.MiddleCenter);
            trailAccText.color = new Color(0.0f, 0.94f, 1.0f);
            SetAnchors(trailAccText.rectTransform, 0.05f, 0.535f, 0.95f, 0.58f);

            GameObject trailBarBg = CreateUIElement("TrailAccuracyBarBg", dashPanel.transform);
            SetAnchors(trailBarBg.GetComponent<RectTransform>(), 0.1f, 0.505f, 0.9f, 0.53f);
            Image trailBarBgImg = trailBarBg.AddComponent<Image>();
            trailBarBgImg.color = new Color(0.12f, 0.18f, 0.28f, 1f);

            GameObject trailBarFill = CreateUIElement("TrailAccuracyBarFill", trailBarBg.transform);
            SetFullStretch(trailBarFill.GetComponent<RectTransform>());
            Image trailBarFillImg = trailBarFill.AddComponent<Image>();
            trailBarFillImg.color = new Color(0.0f, 0.94f, 1.0f);
            trailBarFillImg.type = Image.Type.Filled;
            trailBarFillImg.fillMethod = Image.FillMethod.Horizontal;
            trailBarFillImg.fillAmount = 1.0f;

            // Commit Dwell Progress Text (Hold to type)
            Text commitProgText = CreateText("CommitProgressText", dashPanel.transform, "Keep sign steady for 2s or tap [Tab] to commit", 15, FontStyle.Normal, TextAnchor.MiddleCenter);
            commitProgText.color = new Color(0.0f, 0.94f, 1.0f);
            SetAnchors(commitProgText.rectTransform, 0.05f, 0.44f, 0.95f, 0.49f);

            // Commit Progress Bar Background
            GameObject commitBarBg = CreateUIElement("CommitBarBg", dashPanel.transform);
            SetAnchors(commitBarBg.GetComponent<RectTransform>(), 0.1f, 0.41f, 0.9f, 0.435f);
            Image commitBarBgImg = commitBarBg.AddComponent<Image>();
            commitBarBgImg.color = new Color(0.14f, 0.20f, 0.30f, 1f);

            GameObject commitBarFill = CreateUIElement("CommitBarFill", commitBarBg.transform);
            SetFullStretch(commitBarFill.GetComponent<RectTransform>());
            Image commitBarFillImg = commitBarFill.AddComponent<Image>();
            commitBarFillImg.color = new Color(0.0f, 0.94f, 1.0f);
            commitBarFillImg.type = Image.Type.Filled;
            commitBarFillImg.fillMethod = Image.FillMethod.Horizontal;
            commitBarFillImg.fillAmount = 0f;

            // Status Text
            Text statusText = CreateText("StatusText", dashPanel.transform, "Status: Ready", 16, FontStyle.Normal, TextAnchor.MiddleCenter);
            statusText.color = new Color(0.95f, 0.85f, 0.35f);
            SetAnchors(statusText.rectTransform, 0.05f, 0.35f, 0.95f, 0.40f);

            // Handedness Text
            Text handednessText = CreateText("HandednessText", dashPanel.transform, "● NO HAND", 15, FontStyle.Bold, TextAnchor.MiddleCenter);
            handednessText.color = new Color(0.6f, 0.72f, 0.88f);
            SetAnchors(handednessText.rectTransform, 0.05f, 0.29f, 0.95f, 0.34f);

            // FPS and Latency Text
            Text fpsText = CreateText("FPSText", dashPanel.transform, "FPS: 30", 15, FontStyle.Normal, TextAnchor.MiddleLeft);
            fpsText.color = new Color(0.6f, 0.72f, 0.85f);
            SetAnchors(fpsText.rectTransform, 0.1f, 0.23f, 0.5f, 0.28f);

            Text latencyText = CreateText("LatencyText", dashPanel.transform, "Latency: 0.1 ms", 15, FontStyle.Normal, TextAnchor.MiddleRight);
            latencyText.color = new Color(0.6f, 0.72f, 0.85f);
            SetAnchors(latencyText.rectTransform, 0.5f, 0.23f, 0.9f, 0.28f);

            // Camera Controls Buttons
            Button startBtn = CreateButton("StartButton", dashPanel.transform, "▶ START CAMERA", new Color(0.06f, 0.72f, 0.40f), 0.1f, 0.13f, 0.52f, 0.21f);
            Button stopBtn = CreateButton("StopButton", dashPanel.transform, "■ STOP CAMERA", new Color(0.85f, 0.20f, 0.25f), 0.55f, 0.13f, 0.9f, 0.21f);
            Button clearSignBtn = CreateButton("ClearSignButton", dashPanel.transform, "↺ CLEAR SIGN", new Color(0.24f, 0.32f, 0.45f), 0.1f, 0.035f, 0.45f, 0.11f);

            // Debug Skeleton Toggle
            Toggle debugToggle = CreateToggle("SkeletonToggle", dashPanel.transform, "Show Skeleton", 0.48f, 0.035f, 0.9f, 0.11f);

            // 8. Lower Main Section: Word & Sentence Formation Dashboard Console
            GameObject wordConsoleObj = CreateUIElement("WordAssemblyConsole", canvasObj.transform);
            RectTransform wordConsoleRect = wordConsoleObj.GetComponent<RectTransform>();
            SetAnchors(wordConsoleRect, 0.02f, 0.02f, 0.98f, 0.28f);
            Image wordConsoleBg = wordConsoleObj.AddComponent<Image>();
            wordConsoleBg.color = new Color(0.05f, 0.07f, 0.12f, 1f);

            // Cyan top accent line on Word Assembly Console
            GameObject consoleAccentLine = CreateUIElement("ConsoleAccentLine", wordConsoleObj.transform);
            SetAnchors(consoleAccentLine.GetComponent<RectTransform>(), 0f, 0.985f, 1f, 1f);
            Image consoleAccentLineImg = consoleAccentLine.AddComponent<Image>();
            consoleAccentLineImg.color = new Color(0.0f, 0.94f, 1.0f, 0.85f);

            // 8A. Formed Word Card (Left-Upper inside Console)
            GameObject wordCardObj = CreateUIElement("FormedWordCard", wordConsoleObj.transform);
            SetAnchors(wordCardObj.GetComponent<RectTransform>(), 0.02f, 0.52f, 0.68f, 0.95f);
            Image wordCardBg = wordCardObj.AddComponent<Image>();
            wordCardBg.color = new Color(0.08f, 0.12f, 0.20f, 1f);

            Text wordCardHeader = CreateText("WordHeader", wordCardObj.transform, "⚡ COMMENT BAR (FORMED WORD)", 14, FontStyle.Bold, TextAnchor.MiddleLeft);
            wordCardHeader.color = new Color(0.0f, 0.94f, 1.0f);
            SetAnchors(wordCardHeader.rectTransform, 0.02f, 0.65f, 0.98f, 0.98f);

            Text formedWordText = CreateText("FormedWordText", wordCardObj.transform, "-", 36, FontStyle.Bold, TextAnchor.MiddleLeft);
            formedWordText.color = new Color(0.10f, 0.98f, 0.50f);
            SetAnchors(formedWordText.rectTransform, 0.02f, 0.05f, 0.48f, 0.65f);

            // 8B. Translated Sentence Card (Left-Lower inside Console)
            GameObject sentenceCardObj = CreateUIElement("SentenceCard", wordConsoleObj.transform);
            SetAnchors(sentenceCardObj.GetComponent<RectTransform>(), 0.02f, 0.07f, 0.68f, 0.48f);
            Image sentenceCardBg = sentenceCardObj.AddComponent<Image>();
            sentenceCardBg.color = new Color(0.07f, 0.10f, 0.17f, 1f);

            Text sentenceCardHeader = CreateText("SentenceHeader", sentenceCardObj.transform, "◈ SPEECH TRANSLATION HISTORY", 14, FontStyle.Bold, TextAnchor.MiddleLeft);
            sentenceCardHeader.color = new Color(0.65f, 0.78f, 0.92f);
            SetAnchors(sentenceCardHeader.rectTransform, 0.02f, 0.65f, 0.98f, 0.98f);

            Text formedSentenceText = CreateText("FormedSentenceText", sentenceCardObj.transform, "Sentence: (empty)", 18, FontStyle.Normal, TextAnchor.MiddleLeft);
            formedSentenceText.color = Color.white;
            SetAnchors(formedSentenceText.rectTransform, 0.02f, 0.05f, 0.98f, 0.65f);

            // 8C. Word Action Buttons (Right side inside Console)
            // Row 1: Space & Backspace
            Button spaceBtn = CreateButton("SpaceButton", wordConsoleObj.transform, "⎵ SPACE", new Color(0.08f, 0.48f, 0.85f), 0.70f, 0.52f, 0.84f, 0.95f);
            Button backspaceBtn = CreateButton("BackspaceButton", wordConsoleObj.transform, "⌫ BACKSPACE", new Color(0.90f, 0.50f, 0.10f), 0.85f, 0.52f, 0.98f, 0.95f);

            // Row 2: Speak, Clear Word, Reset All
            Button speakBtn = CreateButton("SpeakButton", wordConsoleObj.transform, "🔊 SPEAK", new Color(0.55f, 0.25f, 0.95f), 0.70f, 0.07f, 0.84f, 0.48f);
            Button clearWordBtn = CreateButton("ClearWordButton", wordConsoleObj.transform, "🗑 CLEAR", new Color(0.30f, 0.38f, 0.50f), 0.85f, 0.07f, 0.91f, 0.48f);
            Button clearAllBtn = CreateButton("ClearAllButton", wordConsoleObj.transform, "✕ RESET ALL", new Color(0.80f, 0.18f, 0.18f), 0.92f, 0.07f, 0.98f, 0.48f);

            // Bottom tip text
            Text shortcutTip = CreateText("ShortcutTip", wordConsoleObj.transform, "Hold sign 2s or [Tab] Commit • [1, 2, 3] Smart Word • [H] AR HUD • [G] Ghost Pose • [T] Motion Trails • [Y] Heatmap • [Enter] Speak", 13, FontStyle.Italic, TextAnchor.MiddleLeft);
            shortcutTip.color = new Color(0.45f, 0.65f, 0.85f);
            SetAnchors(shortcutTip.rectTransform, 0.02f, 0.005f, 0.68f, 0.06f);

            // 9. Create Application Controller GameObject
            GameObject appController = new GameObject("ASL_ApplicationController");
            WebcamManager webcamManager = appController.AddComponent<WebcamManager>();
            MediaPipeManager mediaPipeManager = appController.AddComponent<MediaPipeManager>();
            HandLandmarkExtractor landmarkExtractor = appController.AddComponent<HandLandmarkExtractor>();
            SignLanguageClassifier classifier = appController.AddComponent<SignLanguageClassifier>();
            PredictionSmoother smoother = appController.AddComponent<PredictionSmoother>();
            WordPredictor wordPredictor = appController.AddComponent<WordPredictor>();
            WordAssembler wordAssembler = appController.AddComponent<WordAssembler>();
            UIManager uiManager = appController.AddComponent<UIManager>();
            OfflineSimulator simulator = appController.AddComponent<OfflineSimulator>();

            // 10. Wire Inspector References via SerializedProperties
            SerializedObject soWebcam = new SerializedObject(webcamManager);
            soWebcam.FindProperty("displayImage").objectReferenceValue = feedImage;
            soWebcam.FindProperty("aspectFitter").objectReferenceValue = aspectFitter;
            soWebcam.ApplyModifiedProperties();

            SerializedObject soMediaPipe = new SerializedObject(mediaPipeManager);
            soMediaPipe.FindProperty("pythonPath").stringValue = "";
            soMediaPipe.ApplyModifiedProperties();

            SerializedObject soUI = new SerializedObject(uiManager);
            soUI.FindProperty("webcamManager").objectReferenceValue = webcamManager;
            soUI.FindProperty("mediaPipeManager").objectReferenceValue = mediaPipeManager;
            soUI.FindProperty("landmarkExtractor").objectReferenceValue = landmarkExtractor;
            soUI.FindProperty("classifier").objectReferenceValue = classifier;
            soUI.FindProperty("smoother").objectReferenceValue = smoother;
            soUI.FindProperty("skeletonRenderer").objectReferenceValue = skeletonRenderer;
            soUI.FindProperty("arSpatialHUD").objectReferenceValue = arHUD;
            soUI.FindProperty("wordAssembler").objectReferenceValue = wordAssembler;

            soUI.FindProperty("detectedLetterText").objectReferenceValue = letterText;
            soUI.FindProperty("confidenceText").objectReferenceValue = confText;
            soUI.FindProperty("confidenceBar").objectReferenceValue = barFillImg;
            soUI.FindProperty("trailAccuracyText").objectReferenceValue = trailAccText;
            soUI.FindProperty("trailAccuracyBar").objectReferenceValue = trailBarFillImg;
            soUI.FindProperty("statusText").objectReferenceValue = statusText;
            soUI.FindProperty("fpsText").objectReferenceValue = fpsText;
            soUI.FindProperty("latencyText").objectReferenceValue = latencyText;
            soUI.FindProperty("handednessText").objectReferenceValue = handednessText;

            soUI.FindProperty("formedWordText").objectReferenceValue = formedWordText;
            soUI.FindProperty("formedSentenceText").objectReferenceValue = formedSentenceText;
            soUI.FindProperty("commitProgressBar").objectReferenceValue = commitBarFillImg;
            soUI.FindProperty("commitProgressText").objectReferenceValue = commitProgText;

            soUI.FindProperty("startCameraButton").objectReferenceValue = startBtn;
            soUI.FindProperty("stopCameraButton").objectReferenceValue = stopBtn;
            soUI.FindProperty("clearSignButton").objectReferenceValue = clearSignBtn;
            soUI.FindProperty("debugSkeletonToggle").objectReferenceValue = debugToggle;

            soUI.FindProperty("spaceButton").objectReferenceValue = spaceBtn;
            soUI.FindProperty("backspaceButton").objectReferenceValue = backspaceBtn;
            soUI.FindProperty("clearWordButton").objectReferenceValue = clearWordBtn;
            soUI.FindProperty("clearAllButton").objectReferenceValue = clearAllBtn;
            soUI.FindProperty("speakButton").objectReferenceValue = speakBtn;
            soUI.ApplyModifiedProperties();

            SerializedObject soSim = new SerializedObject(simulator);
            soSim.FindProperty("mediaPipeManager").objectReferenceValue = mediaPipeManager;
            soSim.ApplyModifiedProperties();

            SerializedObject soWord = new SerializedObject(wordAssembler);
            soWord.FindProperty("commitDwellTime").floatValue = 2.0f;
            soWord.FindProperty("clearSentenceAfterSpeak").boolValue = true;
            soWord.ApplyModifiedProperties();

            // 11. Save Scene
            string scenePath = "Assets/Scenes/MainScene.unity";
            EditorSceneManager.SaveScene(scene, scenePath);
            Debug.Log($"<color=green>[SceneSetupHelper] ASL MainScene created and saved to: {scenePath}</color>");
            if (!Application.isBatchMode)
            {
                EditorUtility.DisplayDialog("ASL Sign Language Detection", "MainScene has been successfully generated!\n\nAll buttons, live sign detection, Word & Sentence merger, and Windows voice synthesis are configured.", "OK");
            }
        }

        private static GameObject CreateUIElement(string name, Transform parent)
        {
            GameObject obj = new GameObject(name, typeof(RectTransform));
            obj.transform.SetParent(parent, false);
            return obj;
        }

        private static void SetFullStretch(RectTransform rt)
        {
            rt.anchorMin = Vector2.zero;
            rt.anchorMax = Vector2.one;
            rt.offsetMin = Vector2.zero;
            rt.offsetMax = Vector2.zero;
        }

        private static void SetAnchors(RectTransform rt, float minX, float minY, float maxX, float maxY)
        {
            rt.anchorMin = new Vector2(minX, minY);
            rt.anchorMax = new Vector2(maxX, maxY);
            rt.offsetMin = Vector2.zero;
            rt.offsetMax = Vector2.zero;
        }

        private static Text CreateText(string name, Transform parent, string content, int fontSize, FontStyle style, TextAnchor align)
        {
            GameObject obj = CreateUIElement(name, parent);
            Text t = obj.AddComponent<Text>();
            t.text = content;
            t.fontSize = fontSize;
            t.fontStyle = style;
            t.alignment = align;
            t.font = Resources.GetBuiltinResource<Font>("LegacyRuntime.ttf");
            if (t.font == null) t.font = Resources.GetBuiltinResource<Font>("Arial.ttf");
            t.raycastTarget = false;
            return t;
        }

        private static Button CreateButton(string name, Transform parent, string label, Color bgColor, float minX, float minY, float maxX, float maxY)
        {
            GameObject btnObj = CreateUIElement(name, parent);
            RectTransform rt = btnObj.GetComponent<RectTransform>();
            SetAnchors(rt, minX, minY, maxX, maxY);

            Image img = btnObj.AddComponent<Image>();
            img.color = bgColor;

            Button btn = btnObj.AddComponent<Button>();
            ColorBlock cb = btn.colors;
            cb.highlightedColor = bgColor * 1.25f;
            cb.pressedColor = bgColor * 0.80f;
            btn.colors = cb;

            Text t = CreateText("Label", btnObj.transform, label, 15, FontStyle.Bold, TextAnchor.MiddleCenter);
            t.color = Color.white;
            SetFullStretch(t.rectTransform);

            return btn;
        }

        private static Toggle CreateToggle(string name, Transform parent, string label, float minX, float minY, float maxX, float maxY)
        {
            GameObject toggleObj = CreateUIElement(name, parent);
            SetAnchors(toggleObj.GetComponent<RectTransform>(), minX, minY, maxX, maxY);

            Toggle toggle = toggleObj.AddComponent<Toggle>();
            toggle.isOn = true;

            // Background box
            GameObject boxObj = CreateUIElement("Background", toggleObj.transform);
            RectTransform boxRt = boxObj.GetComponent<RectTransform>();
            boxRt.anchorMin = new Vector2(0f, 0.15f);
            boxRt.anchorMax = new Vector2(0.14f, 0.85f);
            boxRt.offsetMin = Vector2.zero;
            boxRt.offsetMax = Vector2.zero;
            Image boxImg = boxObj.AddComponent<Image>();
            boxImg.color = new Color(0.14f, 0.20f, 0.30f);

            // Checkmark
            GameObject checkObj = CreateUIElement("Checkmark", boxObj.transform);
            SetFullStretch(checkObj.GetComponent<RectTransform>());
            Image checkImg = checkObj.AddComponent<Image>();
            checkImg.color = new Color(0.10f, 0.98f, 0.50f);

            toggle.targetGraphic = boxImg;
            toggle.graphic = checkImg;

            // Label
            Text t = CreateText("Label", toggleObj.transform, label, 15, FontStyle.Normal, TextAnchor.MiddleLeft);
            t.color = Color.white;
            RectTransform labelRt = t.rectTransform;
            labelRt.anchorMin = new Vector2(0.18f, 0f);
            labelRt.anchorMax = new Vector2(1f, 1f);
            labelRt.offsetMin = Vector2.zero;
            labelRt.offsetMax = Vector2.zero;

            return toggle;
        }
    }
}
#endif
