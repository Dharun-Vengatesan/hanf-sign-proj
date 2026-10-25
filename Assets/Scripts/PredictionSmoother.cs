using System;
using System.Collections.Generic;
using UnityEngine;

namespace ASLSignDetection
{
    /// <summary>
    /// Implements temporal smoothing, majority voting, and confidence thresholding
    /// to eliminate flickering and deliver solid, stable letter predictions.
    /// </summary>
    public class PredictionSmoother : MonoBehaviour
    {
        [Header("Stability Settings (Editable in Inspector)")]
        [Tooltip("Minimum confidence required to accept a prediction (e.g. 0.55 = 55%).")]
        [Range(0.30f, 0.95f)]
        [SerializeField] private float minimumConfidence = 0.55f;

        [Tooltip("Number of consecutive frames the same letter must appear before confirming.")]
        [Range(1, 15)]
        [SerializeField] private int requiredStableFrames = 2;

        [Tooltip("Sliding window history size for majority voting.")]
        [Range(3, 30)]
        [SerializeField] private int historyBufferSize = 8;

        [Tooltip("Label displayed when hand is moving, unstable, or confidence is below threshold.")]
        [SerializeField] private string unknownLabel = "Unknown";

        [Header("Current Smoothed Output")]
        [SerializeField] private string currentStableLetter = "None";
        [SerializeField] private float smoothedConfidence = 0.0f;
        [SerializeField] private int consecutiveCount = 0;
        [SerializeField] private bool isConfident = false;

        private Queue<ClassificationResult> historyBuffer = new Queue<ClassificationResult>();
        private string candidateLetter = "";
        private int lostFramesCount = 0;
        private int lowConfCount = 0;

        public string CurrentStableLetter => currentStableLetter;
        public float SmoothedConfidence => smoothedConfidence;
        public bool IsConfident => isConfident;
        public float MinimumConfidence => minimumConfidence;

        public event Action<string, float, bool> OnStablePredictionChanged;

        /// <summary>
        /// Processes a new raw classification frame and applies temporal smoothing.
        /// </summary>
        public void ProcessFrame(ClassificationResult rawResult)
        {
            if (!rawResult.IsValid || string.IsNullOrEmpty(rawResult.Letter))
            {
                lostFramesCount++;
                if (lostFramesCount >= 4)
                {
                    HandleUnstableFrame(0f);
                }
                return;
            }

            lostFramesCount = 0;

            // Check confidence threshold
            if (rawResult.Confidence < minimumConfidence)
            {
                lowConfCount++;
                if (lowConfCount >= 4)
                {
                    HandleUnstableFrame(rawResult.Confidence);
                }
                return;
            }

            lowConfCount = 0;

            // Append to sliding history buffer
            historyBuffer.Enqueue(rawResult);
            if (historyBuffer.Count > historyBufferSize)
            {
                historyBuffer.Dequeue();
            }

            // Track consecutive frame count
            if (rawResult.Letter == candidateLetter)
            {
                consecutiveCount++;
            }
            else
            {
                candidateLetter = rawResult.Letter;
                consecutiveCount = 1;
            }

            // Majority voting across the history buffer
            Dictionary<string, int> votes = new Dictionary<string, int>();
            Dictionary<string, float> confSum = new Dictionary<string, float>();

            foreach (var item in historyBuffer)
            {
                if (!votes.ContainsKey(item.Letter))
                {
                    votes[item.Letter] = 0;
                    confSum[item.Letter] = 0f;
                }
                votes[item.Letter]++;
                confSum[item.Letter] += item.Confidence;
            }

            string majorityLetter = "";
            int maxVotes = 0;
            foreach (var kvp in votes)
            {
                if (kvp.Value > maxVotes)
                {
                    maxVotes = kvp.Value;
                    majorityLetter = kvp.Key;
                }
            }

            // Confirm letter:
            // 1. If candidate letter has appeared consecutively for requiredStableFrames (crisp transition to new sign)
            // 2. OR dominant majority in history window
            int majorityThreshold = Mathf.Max(2, Mathf.CeilToInt(historyBuffer.Count * 0.50f));
            string confirmedLetter = "";
            float confirmedConf = 0f;

            if (consecutiveCount >= requiredStableFrames && !string.IsNullOrEmpty(candidateLetter))
            {
                confirmedLetter = candidateLetter;
                confirmedConf = rawResult.Confidence;

                // When transitioning to a new confirmed letter, clear stale history of the old letter
                if (currentStableLetter != confirmedLetter)
                {
                    historyBuffer.Clear();
                    historyBuffer.Enqueue(rawResult);
                }
            }
            else if (maxVotes >= majorityThreshold && !string.IsNullOrEmpty(majorityLetter))
            {
                confirmedLetter = majorityLetter;
                confirmedConf = confSum[majorityLetter] / maxVotes;
            }

            if (!string.IsNullOrEmpty(confirmedLetter))
            {
                SetStablePrediction(confirmedLetter, confirmedConf, true);
            }
            else if (currentStableLetter == unknownLabel || string.IsNullOrEmpty(currentStableLetter))
            {
                // Only show Unknown if there is no confirmed stable letter
                HandleUnstableFrame(rawResult.Confidence);
            }
        }

        private void HandleUnstableFrame(float currentConfidence)
        {
            consecutiveCount = 0;
            candidateLetter = "";
            smoothedConfidence = currentConfidence;
            isConfident = false;

            if (currentStableLetter != unknownLabel)
            {
                currentStableLetter = unknownLabel;
                OnStablePredictionChanged?.Invoke(currentStableLetter, smoothedConfidence, false);
            }
        }

        private void SetStablePrediction(string letter, float confidence, bool confident)
        {
            currentStableLetter = letter;
            smoothedConfidence = confidence;
            isConfident = confident;
            lostFramesCount = 0;
            lowConfCount = 0;
            OnStablePredictionChanged?.Invoke(currentStableLetter, smoothedConfidence, isConfident);
        }

        /// <summary>
        /// Clears the temporal history buffer (e.g. when user clicks CLEAR button).
        /// </summary>
        public void Clear()
        {
            historyBuffer.Clear();
            candidateLetter = "";
            consecutiveCount = 0;
            currentStableLetter = unknownLabel;
            smoothedConfidence = 0.0f;
            isConfident = false;
            OnStablePredictionChanged?.Invoke(currentStableLetter, 0f, false);
            Debug.Log("[PredictionSmoother] Prediction buffer cleared.");
        }
    }
}
