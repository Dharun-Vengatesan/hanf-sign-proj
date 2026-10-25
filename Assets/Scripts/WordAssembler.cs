using System;
using System.Text.RegularExpressions;
using System.Threading;
using UnityEngine;

namespace ASLSignDetection
{
    /// <summary>
    /// Intelligently accumulates and merges detected ASL letters into words and comment bar output.
    /// Features:
    /// - True Wall-Clock Dwell Timer: Runs in Update() using Time.unscaledDeltaTime for 100% accurate 5.0s timing.
    /// - Anti-Glitch Grace Period: Momentary camera noise or single-frame drops do not reset the timer.
    /// - Live second-by-second countdown (e.g. 3.5s / 5.0s).
    /// - Anti-flicker cooldown to prevent accidental repeated letters.
    /// - Auto-space on prolonged hand absence / pause.
    /// - Word & Comment operations: Space, Backspace, Clear Word, Clear All, Immediate Commit.
    /// - Procedural audio chime on commit without requiring external audio assets.
    /// - Non-blocking Windows Speech Synthesis (Text-to-Speech) for vocal readout.
    /// </summary>
    public class WordAssembler : MonoBehaviour
    {
        [Header("Comment Bar & Word Assembly Settings")]
        [Tooltip("Seconds a detected sign must be held stable before automatically committing to the comment bar.")]
        [Range(0.5f, 5.0f)]
        [SerializeField] private float commitDwellTime = 1.5f;

        [Tooltip("Minimum confidence required for a letter to count towards dwell progress.")]
        [Range(0.3f, 0.95f)]
        [SerializeField] private float minConfidenceToCommit = 0.50f;

        [Tooltip("Minimum hand stability required to advance dwell timer (0.0 to 1.0). Prevents accidental commits while moving hand.")]
        [Range(0.0f, 1.0f)]
        [SerializeField] private float minStabilityToDwell = 0.55f;

        [Tooltip("Grace period (seconds) during which momentary frame drops or glitches won't reset the 5s timer.")]
        [Range(0.1f, 1.0f)]
        [SerializeField] private float glitchGracePeriod = 0.45f;

        [Tooltip("Seconds after committing a letter during which the same letter won't re-commit.")]
        [Range(0.3f, 3.0f)]
        [SerializeField] private float repeatLetterCooldown = 1.0f;

        [Tooltip("Automatically insert a space (end word) if hand is absent for a set duration.")]
        [SerializeField] private bool autoSpaceOnHandAbsence = true;

        [Tooltip("Seconds of no hand before auto-inserting space.")]
        [Range(1.0f, 6.0f)]
        [SerializeField] private float autoSpaceDelay = 2.5f;

        [Tooltip("Play procedural audio chime/click when a letter commits.")]
        [SerializeField] private bool playAudioOnCommit = true;

        [Header("Speech Synthesis & Sentence Output Settings")]
        [Tooltip("Speak formed words/sentences aloud using Windows Text-to-Speech.")]
        [SerializeField] private bool enableSpeechSynthesis = true;

        [Tooltip("Automatically empty the sentence and comment bar once the sentence/word is spoken.")]
        [SerializeField] private bool clearSentenceAfterSpeak = true;

        [Tooltip("Automatically speak the entire sentence and clear it when the hand is rested/lowered for a prolonged duration (e.g. 5s).")]
        [SerializeField] private bool autoSpeakSentenceOnHandRest = false;

        [Tooltip("Seconds of no hand before auto-speaking the full sentence (e.g. 5.0s).")]
        [Range(3.0f, 15.0f)]
        [SerializeField] private float autoSpeakSentenceDelay = 5.0f;

        [Header("Predictive Text Engine")]
        [Tooltip("Smart predictive word suggestion engine.")]
        [SerializeField] private WordPredictor wordPredictor;

        [Header("Runtime State (Read Only)")]
        [SerializeField] private string currentWord = "";
        [SerializeField] private string fullSentence = "";
        [SerializeField] private string currentCandidateLetter = "";
        [SerializeField] private float dwellTimer = 0f;
        [SerializeField] private float dwellProgress = 0f;
        [SerializeField] private bool isHandDetected = false;

        // Latest streaming prediction from detector
        private string latestPredictedLetter = "";
        private float latestConfidence = 0f;
        private bool latestIsConfident = false;
        private float latestStability = 1.0f;
        private float lastValidLetterTime = -999f;

        // Internal timing and state
        private string lastCommittedLetter = "";
        private float lastCommitTime = -999f;
        private float handAbsenceTimer = 0f;
        private bool hasAutoSpacedForCurrentAbsence = false;
        private bool hasAutoSpokenForCurrentAbsence = false;

        // Audio
        private AudioSource audioSource;
        private AudioClip commitAudioClip;

        // Properties
        public string CurrentWord => currentWord;
        public string FullSentence => fullSentence;
        public string CurrentCandidateLetter => currentCandidateLetter;
        public float DwellTimer => dwellTimer;
        public float DwellProgress => dwellProgress;
        public float CommitDwellTime
        {
            get => commitDwellTime;
            set => commitDwellTime = Mathf.Max(0.5f, value);
        }
        public bool IsHandDetected => isHandDetected;
        public bool ClearSentenceAfterSpeak
        {
            get => clearSentenceAfterSpeak;
            set => clearSentenceAfterSpeak = value;
        }
        public WordPredictor Predictor => wordPredictor;

        // Events
        public event Action<string, string> OnWordUpdated; // (currentWord, fullSentence)
        public event Action<float, string, float, float> OnDwellProgressUpdated; // (progress 0..1, candidateLetter, currentSeconds, targetSeconds)
        public event Action<string> OnLetterCommitted; // (committedLetter)

        private void Awake()
        {
            SetupAudio();
            InitializePredictor();
        }

        private void InitializePredictor()
        {
            if (wordPredictor == null)
            {
                wordPredictor = GetComponent<WordPredictor>();
                if (wordPredictor == null)
                {
                    wordPredictor = gameObject.AddComponent<WordPredictor>();
                }
            }
        }

        private void Start()
        {
            NotifyWordUpdated();
        }

        private void SetupAudio()
        {
            audioSource = GetComponent<AudioSource>();
            if (audioSource == null)
            {
                audioSource = gameObject.AddComponent<AudioSource>();
            }
            audioSource.playOnAwake = false;

            // Generate procedural commit chime (A5 880Hz pleasant bell)
            commitAudioClip = CreateTone(880f, 0.09f);
        }

        private AudioClip CreateTone(float frequency, float duration)
        {
            int sampleRate = 44100;
            int count = (int)(sampleRate * duration);
            float[] samples = new float[count];
            for (int i = 0; i < count; i++)
            {
                float t = (float)i / sampleRate;
                // Exponential decay envelope for a crisp chime
                float envelope = Mathf.Exp(-5f * t / duration);
                samples[i] = Mathf.Sin(2f * Mathf.PI * frequency * t) * envelope * 0.4f;
            }
            AudioClip clip = AudioClip.Create("CommitChime", count, 1, sampleRate, false);
            clip.SetData(samples, 0);
            return clip;
        }

        private void Update()
        {
            HandleKeyboardShortcuts();
            HandleHandAbsenceTracking();
            UpdateDwellAccumulation();
        }

        private void HandleKeyboardShortcuts()
        {
            // Number keys [1], [2], [3] to select predictive suggestions
            if (Input.GetKeyDown(KeyCode.Alpha1) || Input.GetKeyDown(KeyCode.Keypad1))
            {
                SelectSuggestionByIndex(0);
            }
            else if (Input.GetKeyDown(KeyCode.Alpha2) || Input.GetKeyDown(KeyCode.Keypad2))
            {
                SelectSuggestionByIndex(1);
            }
            else if (Input.GetKeyDown(KeyCode.Alpha3) || Input.GetKeyDown(KeyCode.Keypad3))
            {
                SelectSuggestionByIndex(2);
            }

            // Tab to commit active letter immediately without waiting for dwell timer
            if (Input.GetKeyDown(KeyCode.Tab))
            {
                CommitCurrentCandidateImmediately();
            }

            // Space to add space / separate words
            if (Input.GetKeyDown(KeyCode.Space))
            {
                AddSpace();
            }

            // Backspace to delete last character
            if (Input.GetKeyDown(KeyCode.Backspace))
            {
                Backspace();
            }

            // Enter to speak word/sentence
            if (Input.GetKeyDown(KeyCode.Return) || Input.GetKeyDown(KeyCode.KeypadEnter))
            {
                Speak();
            }

            // Delete to clear current word
            if (Input.GetKeyDown(KeyCode.Delete))
            {
                ClearWord();
            }
        }

        private void HandleHandAbsenceTracking()
        {
            if (!isHandDetected)
            {
                handAbsenceTimer += Time.unscaledDeltaTime;
                if (autoSpaceOnHandAbsence && !hasAutoSpacedForCurrentAbsence && handAbsenceTimer >= autoSpaceDelay)
                {
                    if (!string.IsNullOrEmpty(currentWord))
                    {
                        AddSpace();
                        hasAutoSpacedForCurrentAbsence = true;
                    }
                }

                if (autoSpeakSentenceOnHandRest && !hasAutoSpokenForCurrentAbsence && handAbsenceTimer >= autoSpeakSentenceDelay)
                {
                    string fullText = GetFullCombinedSentence();
                    if (!string.IsNullOrEmpty(fullText))
                    {
                        Speak();
                        hasAutoSpokenForCurrentAbsence = true;
                    }
                }
            }
            else
            {
                handAbsenceTimer = 0f;
                hasAutoSpacedForCurrentAbsence = false;
                hasAutoSpokenForCurrentAbsence = false;
            }
        }

        /// <summary>
        /// True Wall-Clock Dwell Timer: Runs every render frame in Update().
        /// Guarantees that 5.0 seconds in the real world equals exactly 5.0 seconds on screen,
        /// independent of camera framerate or UDP packet arrival intervals.
        /// Includes glitch grace period so momentary sensor noise doesn't erase progress.
        /// </summary>
        private void UpdateDwellAccumulation()
        {
            float dt = Time.unscaledDeltaTime;
            float timeSinceLastValid = Time.unscaledTime - lastValidLetterTime;

            // Is the current letter active and valid?
            bool isCurrentLetterValid = isHandDetected && latestIsConfident &&
                                        !string.IsNullOrEmpty(latestPredictedLetter) &&
                                        latestPredictedLetter.Length == 1 &&
                                        latestPredictedLetter[0] >= 'A' && latestPredictedLetter[0] <= 'Z' &&
                                        latestConfidence >= minConfidenceToCommit;

            if (isCurrentLetterValid)
            {
                // If this is a new letter candidate, switch candidate and restart timer
                if (latestPredictedLetter != currentCandidateLetter)
                {
                    currentCandidateLetter = latestPredictedLetter;
                    dwellTimer = 0f;
                    dwellProgress = 0f;
                }

                // Check repeat cooldown for the same letter (prevents double-triggering immediately)
                if (currentCandidateLetter == lastCommittedLetter && (Time.unscaledTime - lastCommitTime) < repeatLetterCooldown)
                {
                    dwellTimer = 0f;
                    dwellProgress = 0f;
                    OnDwellProgressUpdated?.Invoke(0f, currentCandidateLetter, 0f, commitDwellTime);
                    return;
                }

                // Only advance dwell if hand is reasonably stable (not actively swinging or transitioning)
                if (latestStability >= minStabilityToDwell)
                {
                    dwellTimer += dt;
                    dwellProgress = Mathf.Clamp01(dwellTimer / Mathf.Max(0.1f, commitDwellTime));
                    OnDwellProgressUpdated?.Invoke(dwellProgress, currentCandidateLetter, dwellTimer, commitDwellTime);

                    // Commit to comment bar once held
                    if (dwellTimer >= commitDwellTime)
                    {
                        CommitCandidateLetter(currentCandidateLetter);
                    }
                }
                else
                {
                    // Hand is moving/shaking: pause progress without immediately resetting
                    OnDwellProgressUpdated?.Invoke(dwellProgress, currentCandidateLetter, dwellTimer, commitDwellTime);
                }
            }
            else
            {
                // Momentary drop or glitch: check if within grace period
                if (timeSinceLastValid < glitchGracePeriod && !string.IsNullOrEmpty(currentCandidateLetter) && dwellTimer > 0f)
                {
                    // Within grace period: pause timer instead of resetting to zero!
                    OnDwellProgressUpdated?.Invoke(dwellProgress, currentCandidateLetter, dwellTimer, commitDwellTime);
                }
                else
                {
                    // Genuinely lost hand or sign changed for longer than grace period: reset dwell
                    if (dwellTimer > 0f || !string.IsNullOrEmpty(currentCandidateLetter))
                    {
                        ResetDwell();
                    }
                }
            }
        }

        /// <summary>
        /// Call whenever hand tracking status changes (e.g. from MediaPipe).
        /// </summary>
        public void SetHandDetected(bool detected)
        {
            isHandDetected = detected;
            if (!detected)
            {
                latestIsConfident = false;
            }
        }

        /// <summary>
        /// Feeds the stabilized prediction from PredictionSmoother into the state machine.
        /// Real-time accumulation happens in Update() with true wall-clock time.
        /// </summary>
        public void ProcessLetterPrediction(string letter, float confidence, bool isConfident, float stability = 1.0f)
        {
            latestPredictedLetter = letter;
            latestConfidence = confidence;
            latestIsConfident = isConfident;
            latestStability = stability;

            if (isConfident && !string.IsNullOrEmpty(letter) && letter != "Unknown" && letter != "-" && confidence >= minConfidenceToCommit)
            {
                lastValidLetterTime = Time.unscaledTime;
            }
        }

        private void CommitCandidateLetter(string letter)
        {
            currentWord += letter;
            lastCommittedLetter = letter;
            lastCommitTime = Time.unscaledTime;
            dwellTimer = 0f;
            dwellProgress = 0f;

            if (playAudioOnCommit && audioSource != null && commitAudioClip != null)
            {
                audioSource.PlayOneShot(commitAudioClip, 0.5f);
            }

            OnLetterCommitted?.Invoke(letter);
            NotifyWordUpdated();
            OnDwellProgressUpdated?.Invoke(0f, "", 0f, commitDwellTime);
            Debug.Log($"[WordAssembler] Automatically committed '{letter}' to Comment Bar after {commitDwellTime:F1}s hold. Current Word: '{currentWord}'");
        }

        /// <summary>
        /// Immediately commits the active candidate letter without waiting for dwell timer.
        /// </summary>
        public void CommitCurrentCandidateImmediately()
        {
            if (!string.IsNullOrEmpty(currentCandidateLetter) && currentCandidateLetter != "Unknown")
            {
                CommitCandidateLetter(currentCandidateLetter);
            }
        }

        /// <summary>
        /// Appends the current word to the sentence with a space, and clears current word.
        /// If speech synthesis is enabled, speaks the completed word and empties the sentence if configured.
        /// </summary>
        /// <summary>
        /// Finalizes the current word into the sentence with a space, and clears the active word
        /// so the user can spell the next word. Preserves the full sentence!
        /// </summary>
        public void AddSpace()
        {
            if (!string.IsNullOrEmpty(currentWord))
            {
                if (string.IsNullOrEmpty(fullSentence))
                {
                    fullSentence = currentWord;
                }
                else
                {
                    fullSentence += " " + currentWord;
                }

                currentWord = "";
            }
            else if (!string.IsNullOrEmpty(fullSentence) && !fullSentence.EndsWith(" "))
            {
                fullSentence += " ";
            }

            ResetDwell();
            NotifyWordUpdated();
            Debug.Log($"[WordAssembler] Added Word to Sentence. Full Sentence: '{fullSentence}'");
        }

        /// <summary>
        /// Replaces the active partial word with the chosen smart word suggestion
        /// and automatically commits it to the sentence with a space.
        /// </summary>
        public void ApplySuggestion(string suggestedWord)
        {
            if (string.IsNullOrEmpty(suggestedWord)) return;

            currentWord = suggestedWord.Trim().ToUpper();
            AddSpace();

            if (playAudioOnCommit && audioSource != null && commitAudioClip != null)
            {
                audioSource.PlayOneShot(commitAudioClip, 0.75f);
            }

            Debug.Log($"[WordAssembler] Applied smart suggestion: '{suggestedWord}'");
        }

        /// <summary>
        /// Selects a suggestion by its 0-based index (0, 1, or 2).
        /// </summary>
        public void SelectSuggestionByIndex(int index)
        {
            if (wordPredictor != null && wordPredictor.CurrentSuggestions != null)
            {
                if (index >= 0 && index < wordPredictor.CurrentSuggestions.Length)
                {
                    string word = wordPredictor.CurrentSuggestions[index];
                    if (!string.IsNullOrEmpty(word))
                    {
                        ApplySuggestion(word);
                    }
                }
            }
        }

        private void NotifyWordUpdated()
        {
            if (wordPredictor != null)
            {
                wordPredictor.PredictWords(currentWord);
            }
            OnWordUpdated?.Invoke(currentWord, fullSentence);
        }

        /// <summary>
        /// Deletes the last character in the current word, or last word/space in sentence.
        /// </summary>
        public void Backspace()
        {
            if (currentWord.Length > 0)
            {
                currentWord = currentWord.Substring(0, currentWord.Length - 1);
            }
            else if (fullSentence.Length > 0)
            {
                fullSentence = fullSentence.TrimEnd();
                if (fullSentence.Length > 0)
                {
                    int lastSpace = fullSentence.LastIndexOf(' ');
                    if (lastSpace >= 0)
                    {
                        currentWord = fullSentence.Substring(lastSpace + 1);
                        fullSentence = fullSentence.Substring(0, lastSpace);
                    }
                    else
                    {
                        currentWord = fullSentence;
                        fullSentence = "";
                    }
                }
            }

            ResetDwell();
            NotifyWordUpdated();
            Debug.Log($"[WordAssembler] Backspace. Current Word: '{currentWord}', Sentence: '{fullSentence}'");
        }

        /// <summary>
        /// Clears only the current word in progress.
        /// </summary>
        public void ClearWord()
        {
            currentWord = "";
            ResetDwell();
            NotifyWordUpdated();
            Debug.Log("[WordAssembler] Cleared current word.");
        }

        /// <summary>
        /// Clears both the current word and the entire sentence history.
        /// </summary>
        public void ClearAll()
        {
            currentWord = "";
            fullSentence = "";
            ResetDwell();
            NotifyWordUpdated();
            Debug.Log("[WordAssembler] Cleared all text.");
        }

        private void ResetDwell()
        {
            dwellTimer = 0f;
            if (dwellProgress > 0f)
            {
                dwellProgress = 0f;
                OnDwellProgressUpdated?.Invoke(0f, "", 0f, commitDwellTime);
            }
            currentCandidateLetter = "";
        }

        /// <summary>
        /// Combines the full sentence history and any active word currently in progress into a single sentence string.
        /// </summary>
        public string GetFullCombinedSentence()
        {
            if (!string.IsNullOrEmpty(fullSentence) && !string.IsNullOrEmpty(currentWord))
            {
                return fullSentence + " " + currentWord;
            }
            if (!string.IsNullOrEmpty(fullSentence))
            {
                return fullSentence;
            }
            return currentWord;
        }

        /// <summary>
        /// Vocalizes the full sentence (or single word) via Windows Text-to-Speech,
        /// and empties the entire sentence and comment bar once spoken.
        /// </summary>
        public void Speak()
        {
            string sentenceToSpeak = GetFullCombinedSentence();
            if (string.IsNullOrWhiteSpace(sentenceToSpeak)) return;

            Debug.Log($"[WordAssembler] Speaking full sentence: \"{sentenceToSpeak}\"");
            SpeakText(sentenceToSpeak);

            if (clearSentenceAfterSpeak)
            {
                currentWord = "";
                fullSentence = "";
                ResetDwell();
                NotifyWordUpdated();
                Debug.Log("[WordAssembler] Spoke sentence and made the sentence empty.");
            }
        }

        private void SpeakText(string text)
        {
            if (string.IsNullOrWhiteSpace(text)) return;

            ThreadPool.QueueUserWorkItem(_ =>
            {
                try
                {
                    string safeText = Regex.Replace(text, @"[^a-zA-Z0-9\s,\.!\?]", "");
                    if (string.IsNullOrWhiteSpace(safeText)) return;

                    var psi = new System.Diagnostics.ProcessStartInfo
                    {
                        FileName = "powershell.exe",
                        Arguments = $"-NoProfile -Command \"$s = New-Object -ComObject SAPI.SpVoice; $s.Speak('{safeText}')\"",
                        CreateNoWindow = true,
                        UseShellExecute = false,
                        WindowStyle = System.Diagnostics.ProcessWindowStyle.Hidden
                    };
                    using (var proc = System.Diagnostics.Process.Start(psi))
                    {
                        proc?.WaitForExit(3500);
                    }
                }
                catch (Exception ex)
                {
                    Debug.LogWarning("[WordAssembler] TTS exception: " + ex.Message);
                }
            });
        }
    }
}
