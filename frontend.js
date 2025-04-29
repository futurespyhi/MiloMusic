async function main() {
  const script1 = document.createElement("script");
  script1.src = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.14.0/dist/ort.js";
  document.head.appendChild(script1);

  const script2 = document.createElement("script");
  script2.onload = async () => {
    console.log("vad loaded");

    let isRecording = false;
    let manualControl = false;

    var recordButton = document.querySelector('.record-button');
    recordButton.textContent = "Click to Start Recording";
    recordButton.style = "width: fit-content; padding-right: 0.5vw;";

    // Store original click behavior
    const originalClick = recordButton.onclick;

    // Override the click handler
    recordButton.onclick = function(event) {
      manualControl = true;
      if (!isRecording) {
        // Start recording
        isRecording = true;
        recordButton.textContent = "Stop Recording";

        // Call the original record functionality
        if (originalClick) originalClick.call(this, event);
      } else {
        // Stop recording
        isRecording = false;
        manualControl = false;
        recordButton.textContent = "Click to Start Recording";

        // Find and click the stop button
        var stopButton = document.querySelector('.stop-button');
        if (stopButton) stopButton.click();
      }
    };

    // Create VAD but disable auto-control
    const myvad = await vad.MicVAD.new({
      onSpeechStart: () => {
        // Only auto-start if not in manual control mode
        if (!manualControl && !isRecording) {
          console.log("Speech detected, but not starting due to manual control mode");
        }
      },
      onSpeechEnd: (audio) => {
        // Only auto-stop if not in manual control mode
        if (!manualControl && isRecording) {
          console.log("Speech ended, but not stopping due to manual control mode");
        }
      }
    });

    // Start VAD for detection but not control
    myvad.start();
  };

  script2.src = "https://cdn.jsdelivr.net/npm/@ricky0123/vad-web@0.0.7/dist/bundle.min.js";
  script1.onload = () => {
    console.log("onnx loaded");
    document.head.appendChild(script2);
  };
}