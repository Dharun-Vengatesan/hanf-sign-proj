using System;
using System.IO;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using System.Diagnostics;
using UnityEngine;
using Debug = UnityEngine.Debug;

namespace ASLSignDetection
{
    /// <summary>
    /// Handles communication with MediaPipe Hands via high-speed local UDP socket.
    /// Manages Python worker process lifecycle and receives both 21 landmarks and video feed.
    /// </summary>
    public class MediaPipeManager : MonoBehaviour
    {
        [Header("UDP Network Settings")]
        [Tooltip("Local port to listen for incoming MediaPipe landmark data.")]
        [SerializeField] private int listenPort = 5052;

        [Header("Auto-Launch MediaPipe Bridge")]
        [Tooltip("If true, automatically starts python/mediapipe_bridge.py when camera starts.")]
        [SerializeField] private bool autoStartPythonBridge = true;

        [Tooltip("Path to python executable containing mediapipe. Leave empty to auto-detect.")]
        [SerializeField] private string pythonPath = "";

        [Tooltip("Whether to show the OpenCV preview window for the Python bridge.")]
        [SerializeField] private bool showPythonPreview = false;

        [Header("Status")]
        [SerializeField] private bool isListening = false;
        [SerializeField] private bool isBridgeRunning = false;
        [SerializeField] private string lastStatus = "Idle";

        // Networking
        private UdpClient udpClient;
        private Thread receiveThread;
        private volatile bool isRunningThread = false;

        // Process management
        private Process pythonProcess;

        // Video texture received from MediaPipe bridge
        private Texture2D videoTexture;
        private byte[] imageByteBuffer;

        // Thread-safe data buffer
        private readonly object lockObj = new object();
        private HandTrackingData latestHandData;
        private bool hasNewData = false;
        private string pendingBase64Image = null;
        private float lastDataReceivedTime = 0f;

        public event Action<HandTrackingData> OnHandDataReceived;
        public event Action<Texture2D> OnVideoFrameReceived;
        public event Action<string> OnStatusChanged;

        public bool IsListening => isListening;
        public bool IsBridgeRunning => isBridgeRunning;
        public bool AutoStartPythonBridge => autoStartPythonBridge;
        public string LastStatus => lastStatus;
        public HandTrackingData LatestData => latestHandData;
        public Texture2D VideoTexture => videoTexture;

        private void Awake()
        {
            videoTexture = new Texture2D(2, 2, TextureFormat.RGB24, false);
        }

        private void Start()
        {
            StartListening();
        }

        public void StartListening()
        {
            if (isListening) return;

            try
            {
                udpClient = new UdpClient(listenPort);
                // Set buffer to 128 KB for smooth video transmission
                udpClient.Client.ReceiveBufferSize = 131072;

                isRunningThread = true;
                receiveThread = new Thread(ReceiveDataLoop)
                {
                    IsBackground = true,
                    Name = "MediaPipeUDPListener"
                };
                receiveThread.Start();
                isListening = true;

                SetStatus("MediaPipe Listener active on port " + listenPort);
                Debug.Log($"[MediaPipeManager] Listening for hand landmarks on UDP port {listenPort}...");
            }
            catch (Exception ex)
            {
                Debug.LogError($"[MediaPipeManager] Failed to bind UDP port {listenPort}: {ex.Message}");
                SetStatus("UDP Port Error");
            }
        }

        public void StopListening()
        {
            isRunningThread = false;

            if (udpClient != null)
            {
                try { udpClient.Close(); } catch { }
                udpClient = null;
            }

            if (receiveThread != null && receiveThread.IsAlive)
            {
                receiveThread.Join(500);
                receiveThread = null;
            }

            isListening = false;
        }

        public void StartPythonBridge()
        {
            if (!autoStartPythonBridge || isBridgeRunning || (pythonProcess != null && !pythonProcess.HasExited)) return;

            try
            {
                string scriptPath = Path.Combine(Application.dataPath, "..", "python", "mediapipe_bridge.py");
                scriptPath = Path.GetFullPath(scriptPath);

                if (!File.Exists(scriptPath))
                {
                    Debug.LogWarning($"[MediaPipeManager] Python bridge script not found at {scriptPath}");
                    return;
                }

                string resolvedPython = GetPythonExecutablePath();
                string previewArg = showPythonPreview ? " --preview" : "";
                string args = $"\"{scriptPath}\" --port {listenPort}{previewArg}";

                ProcessStartInfo startInfo = new ProcessStartInfo
                {
                    FileName = resolvedPython,
                    Arguments = args,
                    UseShellExecute = false,
                    CreateNoWindow = !showPythonPreview,
                    RedirectStandardOutput = false,
                    RedirectStandardError = true,
                    WorkingDirectory = Path.GetDirectoryName(scriptPath)
                };

                pythonProcess = new Process { StartInfo = startInfo, EnableRaisingEvents = true };
                pythonProcess.ErrorDataReceived += (s, e) =>
                {
                    if (!string.IsNullOrEmpty(e.Data))
                    {
                        Debug.LogWarning($"[MediaPipe Bridge Error] {e.Data}");
                    }
                };
                pythonProcess.Exited += (s, e) =>
                {
                    isBridgeRunning = false;
                    Debug.LogWarning($"[MediaPipeManager] Python bridge process exited.");
                };

                pythonProcess.Start();
                pythonProcess.BeginErrorReadLine();
                isBridgeRunning = true;
                SetStatus("MediaPipe Starting...");
                Debug.Log($"[MediaPipeManager] Started MediaPipe bridge with '{resolvedPython}' (PID: {pythonProcess.Id}).");
            }
            catch (Exception ex)
            {
                Debug.LogWarning($"[MediaPipeManager] Auto-start Python bridge note: {ex.Message}. (Can also run run_mediapipe_bridge.bat)");
                SetStatus("External MediaPipe expected");
            }
        }

        public void StopPythonBridge()
        {
            if (pythonProcess != null)
            {
                try
                {
                    if (!pythonProcess.HasExited)
                    {
                        pythonProcess.Kill();
                    }
                    pythonProcess.Dispose();
                }
                catch { }
                pythonProcess = null;
            }
            isBridgeRunning = false;
            Debug.Log("[MediaPipeManager] MediaPipe bridge process stopped.");
        }

        private string GetPythonExecutablePath()
        {
            if (!string.IsNullOrEmpty(pythonPath) && pythonPath != "python" && File.Exists(pythonPath))
                return pythonPath;

            var baseDirs = new System.Collections.Generic.List<string>();

            string localAppData = Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData);
            if (!string.IsNullOrEmpty(localAppData)) baseDirs.Add(localAppData);

            string envLocal = Environment.GetEnvironmentVariable("LOCALAPPDATA");
            if (!string.IsNullOrEmpty(envLocal) && !baseDirs.Contains(envLocal)) baseDirs.Add(envLocal);

            string userProfile = Environment.GetEnvironmentVariable("USERPROFILE");
            if (!string.IsNullOrEmpty(userProfile))
            {
                string userLocal = Path.Combine(userProfile, "AppData", "Local");
                if (!baseDirs.Contains(userLocal)) baseDirs.Add(userLocal);
            }

            var candidates = new System.Collections.Generic.List<string>();

            // Prioritize Python 3.11 and 3.10 which are fully compatible with MediaPipe Solutions
            foreach (var b in baseDirs)
            {
                candidates.Add(Path.Combine(b, "Programs", "Python", "Python311", "python.exe"));
                candidates.Add(Path.Combine(b, "Programs", "Python", "Python310", "python.exe"));
            }

            candidates.Add(@"C:\Python311\python.exe");
            candidates.Add(@"C:\Python310\python.exe");

            foreach (var b in baseDirs)
            {
                candidates.Add(Path.Combine(b, "Programs", "Python", "Python312", "python.exe"));
                candidates.Add(Path.Combine(b, "Programs", "Python", "Python39", "python.exe"));
                candidates.Add(Path.Combine(b, "Programs", "Python", "Python313", "python.exe"));
                candidates.Add(Path.Combine(b, "Programs", "Python", "Python314", "python.exe"));
            }

            candidates.Add(@"C:\Python312\python.exe");
            candidates.Add(@"C:\Python39\python.exe");
            candidates.Add(@"C:\Python313\python.exe");
            candidates.Add(@"C:\Python314\python.exe");

            foreach (var path in candidates)
            {
                if (File.Exists(path))
                    return path;
            }

            return "python";
        }

        private void ReceiveDataLoop()
        {
            IPEndPoint anyIP = new IPEndPoint(IPAddress.Any, 0);

            while (isRunningThread && udpClient != null)
            {
                try
                {
                    byte[] data = udpClient.Receive(ref anyIP);
                    if (data != null && data.Length > 0)
                    {
                        string json = Encoding.UTF8.GetString(data);
                        HandTrackingData parsed = JsonUtility.FromJson<HandTrackingData>(json);

                        lock (lockObj)
                        {
                            latestHandData = parsed;
                            hasNewData = true;

                            if (!string.IsNullOrEmpty(parsed.image))
                            {
                                pendingBase64Image = parsed.image;
                            }
                        }
                    }
                }
                catch (SocketException)
                {
                    break;
                }
                catch (Exception ex)
                {
                    Debug.LogWarning($"[MediaPipeManager] Error receiving UDP packet: {ex.Message}");
                }
            }
        }

        private void Update()
        {
            HandTrackingData dataToDispatch = null;
            string b64ToLoad = null;

            lock (lockObj)
            {
                if (hasNewData && latestHandData != null)
                {
                    dataToDispatch = latestHandData;
                    hasNewData = false;
                    lastDataReceivedTime = Time.realtimeSinceStartup;

                    if (!string.IsNullOrEmpty(pendingBase64Image))
                    {
                        b64ToLoad = pendingBase64Image;
                        pendingBase64Image = null;
                    }
                }
            }

            // Update video frame texture if received from bridge
            if (!string.IsNullOrEmpty(b64ToLoad))
            {
                try
                {
                    imageByteBuffer = Convert.FromBase64String(b64ToLoad);
                    if (videoTexture.LoadImage(imageByteBuffer))
                    {
                        OnVideoFrameReceived?.Invoke(videoTexture);
                    }
                }
                catch { }
            }

            if (dataToDispatch != null)
            {
                if (dataToDispatch.hand_detected)
                {
                    SetStatus("Hand Detected");
                }
                else
                {
                    SetStatus(string.IsNullOrEmpty(dataToDispatch.status) ? "No Hand Detected" : dataToDispatch.status);
                }

                OnHandDataReceived?.Invoke(dataToDispatch);
            }
            else if (isListening && Time.realtimeSinceStartup - lastDataReceivedTime > 3.0f && lastDataReceivedTime > 0f)
            {
                SetStatus("Waiting for MediaPipe...");
            }
        }

        public void FeedSimulatedData(HandTrackingData simulatedData)
        {
            lock (lockObj)
            {
                latestHandData = simulatedData;
                hasNewData = true;
                lastDataReceivedTime = Time.realtimeSinceStartup;
            }
        }

        private void SetStatus(string status)
        {
            if (lastStatus != status)
            {
                lastStatus = status;
                OnStatusChanged?.Invoke(status);
            }
        }

        private void OnDestroy()
        {
            StopPythonBridge();
            StopListening();
            if (videoTexture != null)
            {
                Destroy(videoTexture);
                videoTexture = null;
            }
        }

        private void OnApplicationQuit()
        {
            StopPythonBridge();
            StopListening();
        }
    }
}
