using System;
using UnityEngine;
using UnityEngine.UI;

namespace ASLSignDetection
{
    /// <summary>
    /// Manages laptop webcam capture, texture binding, and lifecycle using Unity's native WebCamTexture.
    /// Does not require any external SDKs, Kinect, or freenect.
    /// </summary>
    public class WebcamManager : MonoBehaviour
    {
        [Header("UI Target")]
        [Tooltip("RawImage component that displays the webcam video feed.")]
        [SerializeField] private RawImage displayImage;

        [Tooltip("AspectRatioFitter component to preserve video proportions.")]
        [SerializeField] private AspectRatioFitter aspectFitter;

        [Header("Webcam Settings")]
        [Tooltip("Desired width of the webcam texture.")]
        [SerializeField] private int requestedWidth = 640;

        [Tooltip("Desired height of the webcam texture.")]
        [SerializeField] private int requestedHeight = 480;

        [Tooltip("Target framerate for the camera.")]
        [SerializeField] private int requestedFPS = 30;

        [Tooltip("Flip image horizontally for natural mirror selfie view.")]
        [SerializeField] private bool mirrorHorizontally = true;

        [Header("Camera Backend Mode")]
        [Tooltip("If true, lets MediaPipe manage camera capture and streams video directly to Unity (prevents Windows exclusive camera locking conflicts).")]
        [SerializeField] private bool useMediaPipeCameraStream = true;

        [Header("Status")]
        [SerializeField] private bool isPlaying = false;
        [SerializeField] private string activeDeviceName = "";

        private WebCamTexture webCamTexture;
        private WebCamDevice[] availableDevices;
        private int selectedDeviceIndex = 0;

        public event Action<WebCamTexture> OnCameraStarted;
        public event Action OnCameraStopped;
        public event Action<string> OnCameraError;

        public bool IsPlaying => isPlaying && ((useMediaPipeCameraStream) || (webCamTexture != null && webCamTexture.isPlaying));
        public WebCamTexture Texture => webCamTexture;
        public string ActiveDeviceName => activeDeviceName;
        public RawImage DisplayImage => displayImage;

        private void Awake()
        {
            if (displayImage != null && displayImage.texture == null)
            {
                displayImage.color = new Color(0.08f, 0.10f, 0.14f, 1f);
            }
        }

        private void Start()
        {
            availableDevices = WebCamTexture.devices;
            if (availableDevices.Length == 0)
            {
                Debug.LogWarning("[WebcamManager] No webcam devices found on this system.");
                OnCameraError?.Invoke("Camera not found");
            }
            else
            {
                activeDeviceName = availableDevices[0].name;
                Debug.Log($"[WebcamManager] Found {availableDevices.Length} camera(s). Default: {activeDeviceName}");
            }
        }

        /// <summary>
        /// Starts the webcam stream.
        /// </summary>
        public void StartCamera()
        {
            if (IsPlaying) return;

            if (useMediaPipeCameraStream)
            {
                // In Windows, only one process can hold the camera at a time.
                // MediaPipe bridge captures the camera and streams frames directly to displayImage.
                isPlaying = true;
                if (displayImage != null)
                {
                    displayImage.uvRect = new Rect(0f, 0f, 1f, 1f);
                    if (displayImage.texture == null)
                    {
                        displayImage.color = new Color(0.08f, 0.10f, 0.14f, 1f);
                    }
                }
                Debug.Log("[WebcamManager] Camera active via MediaPipe bridge stream (prevents Windows camera lock).");
                return;
            }

            availableDevices = WebCamTexture.devices;
            if (availableDevices == null || availableDevices.Length == 0)
            {
                OnCameraError?.Invoke("Camera not found");
                Debug.LogError("[WebcamManager] Unable to start: No video capture device found.");
                return;
            }

            try
            {
                if (selectedDeviceIndex >= availableDevices.Length)
                    selectedDeviceIndex = 0;

                activeDeviceName = availableDevices[selectedDeviceIndex].name;
                webCamTexture = new WebCamTexture(activeDeviceName, requestedWidth, requestedHeight, requestedFPS);
                webCamTexture.Play();
                isPlaying = true;

                if (displayImage != null)
                {
                    displayImage.texture = webCamTexture;
                    displayImage.color = Color.white;

                    // Mirror horizontally if requested
                    if (mirrorHorizontally)
                    {
                        displayImage.uvRect = new Rect(1f, 0f, -1f, 1f);
                    }
                    else
                    {
                        displayImage.uvRect = new Rect(0f, 0f, 1f, 1f);
                    }
                }

                Debug.Log($"[WebcamManager] Camera '{activeDeviceName}' started successfully ({requestedWidth}x{requestedHeight} @ {requestedFPS}fps).");
                OnCameraStarted?.Invoke(webCamTexture);
            }
            catch (Exception ex)
            {
                Debug.LogError($"[WebcamManager] Failed to start webcam: {ex.Message}");
                OnCameraError?.Invoke("Webcam start failed");
            }
        }

        /// <summary>
        /// Stops the webcam stream.
        /// </summary>
        public void StopCamera()
        {
            if (webCamTexture != null)
            {
                if (webCamTexture.isPlaying)
                {
                    webCamTexture.Stop();
                }
                Destroy(webCamTexture);
                webCamTexture = null;
            }

            isPlaying = false;
            if (displayImage != null)
            {
                displayImage.texture = null;
                displayImage.color = new Color(0.08f, 0.10f, 0.14f, 1f);
            }

            Debug.Log("[WebcamManager] Camera stopped.");
            OnCameraStopped?.Invoke();
        }

        private void Update()
        {
            // Adjust aspect ratio dynamically when texture dimensions become valid
            if (webCamTexture != null && webCamTexture.isPlaying && aspectFitter != null && webCamTexture.width > 16)
            {
                float aspect = (float)webCamTexture.width / webCamTexture.height;
                if (Math.Abs(aspectFitter.aspectRatio - aspect) > 0.01f)
                {
                    aspectFitter.aspectRatio = aspect;
                }
            }
        }

        private void OnDestroy()
        {
            StopCamera();
        }

        private void OnApplicationQuit()
        {
            StopCamera();
        }
    }
}
