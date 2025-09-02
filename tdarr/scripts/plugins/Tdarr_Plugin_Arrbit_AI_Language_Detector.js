const details = () => ({
  id: "Tdarr_Plugin_Arrbit_AI_Language_Detector",
  Stage: "Pre-processing",
  Name: "Arrbit - AI Language Detector",
  Type: "Video",
  Operation: "Transcode",
  Description: `Uses WhisperX AI speech recognition to detect audio language and adds language metadata to the file. 
    Supports automatic language detection with confidence scoring and mixed-language detection using Lingua.
    Requires Python environment with WhisperX and lingua-language-detector packages installed.`,
  Version: "1.0",
  Tags: "pre-processing,ai,language,detection,speech,recognition,metadata",
  Inputs: [
    {
      name: "confidence_threshold",
      type: "number",
      defaultValue: 0.7,
      inputUI: {
        type: "text",
      },
      tooltip: `Minimum confidence threshold for language detection (0.0-1.0).
        Lower values are more permissive but may include false positives.
        Higher values are more strict but may miss valid detections.
        \\nExample:\\n
        0.7`,
    },
    {
      name: "sample_duration",
      type: "number",
      defaultValue: 30,
      inputUI: {
        type: "text",
      },
      tooltip: `Duration in seconds of audio sample to analyze for language detection.
        Longer samples provide better accuracy but increase processing time.
        \\nExample:\\n
        30`,
    },
    {
      name: "whisper_model",
      type: "string",
      defaultValue: "base",
      inputUI: {
        type: "dropdown",
        options: ["tiny", "base", "small", "medium", "large"],
      },
      tooltip: `WhisperX model size to use for speech recognition.
        Larger models are more accurate but slower and use more memory.
        \\nModel sizes:\\n
        tiny: ~39 MB, fastest
        base: ~74 MB, good balance
        small: ~244 MB, better accuracy
        medium: ~769 MB, high accuracy
        large: ~1550 MB, highest accuracy`,
    },
    {
      name: "enable_mixed_language",
      type: "boolean",
      defaultValue: true,
      inputUI: {
        type: "dropdown",
        options: ["false", "true"],
      },
      tooltip: `Enable mixed-language detection using Lingua library.
        When enabled, can detect multiple languages within the same audio stream.
        \\nExample:\\n
        true`,
    },
    {
      name: "skip_if_language_exists",
      type: "boolean",
      defaultValue: true,
      inputUI: {
        type: "dropdown",
        options: ["false", "true"],
      },
      tooltip: `Skip processing if language metadata already exists on audio streams.
        When true, only processes files without existing language tags.
        \\nExample:\\n
        true`,
    },
  ],
});

const plugin = (file, librarySettings, inputs, otherArguments) => {
  const lib = require("../methods/lib")();
  // eslint-disable-next-line @typescript-eslint/no-unused-vars,no-param-reassign
  inputs = lib.loadDefaultValues(inputs, details);

  const response = {
    processFile: false,
    preset: "",
    container: `.${file.container}`,
    handBrakeMode: false,
    FFmpegMode: true,
    reQueueAfter: false,
    infoLog: "",
  };

  // Check if the file is a video
  if (file.fileMedium !== "video") {
    response.infoLog += "☒ File is not a video.\n";
    response.processFile = false;
    return response;
  }

  const streams = file.ffProbeData.streams;

  if (!streams || streams.length === 0) {
    response.infoLog += "☒ No streams found.\n";
    response.processFile = false;
    return response;
  }

  // Find audio streams
  const audioStreams = streams.filter(
    (stream) => stream.codec_type && stream.codec_type.toLowerCase() === "audio"
  );

  if (audioStreams.length === 0) {
    response.infoLog += "☒ No audio streams found for language detection.\n";
    response.processFile = false;
    return response;
  }

  // Check if we should skip files that already have language metadata
  if (inputs.skip_if_language_exists) {
    const hasLanguageMetadata = audioStreams.some((stream) => {
      return (
        stream.tags &&
        (stream.tags.language ||
          stream.tags.Language ||
          stream.tags.LANGUAGE ||
          stream.tags.lang ||
          stream.tags.Lang ||
          stream.tags.LANG)
      );
    });

    if (hasLanguageMetadata) {
      response.infoLog +=
        "☑ Audio streams already have language metadata, skipping AI detection.\n";
      response.processFile = false;
      return response;
    }
  }

  response.infoLog += `☑ Found ${audioStreams.length} audio stream(s) for language detection.\n`;

  // Validate input parameters
  const confidenceThreshold = parseFloat(inputs.confidence_threshold);
  if (
    isNaN(confidenceThreshold) ||
    confidenceThreshold < 0 ||
    confidenceThreshold > 1
  ) {
    response.infoLog +=
      "☒ Invalid confidence threshold. Must be between 0.0 and 1.0.\n";
    response.processFile = false;
    return response;
  }

  const sampleDuration = parseInt(inputs.sample_duration, 10);
  if (isNaN(sampleDuration) || sampleDuration < 1 || sampleDuration > 300) {
    response.infoLog +=
      "☒ Invalid sample duration. Must be between 1 and 300 seconds.\n";
    response.processFile = false;
    return response;
  }

  const whisperModel = inputs.whisper_model || "base";
  const validModels = ["tiny", "base", "small", "medium", "large"];
  if (!validModels.includes(whisperModel)) {
    response.infoLog += `☒ Invalid WhisperX model: ${whisperModel}. Must be one of: ${validModels.join(
      ", "
    )}\n`;
    response.processFile = false;
    return response;
  }

  const enableMixedLanguage =
    inputs.enable_mixed_language === true ||
    inputs.enable_mixed_language === "true";

  response.infoLog += `☑ Configuration: model=${whisperModel}, confidence=${confidenceThreshold}, sample=${sampleDuration}s, mixed_lang=${enableMixedLanguage}\n`;

  // Build Python command for AI language detection
  const pythonEnvPath = "/app/arrbit/environments/ai_language_detection";
  const pythonBin = `${pythonEnvPath}/bin/python`;

  // Create temporary script for language detection
  const tempScript = `
import sys
import os
import json
import traceback
import subprocess
import tempfile
from pathlib import Path

def extract_audio_sample(video_path, duration=30, start_time=60):
    """Extract audio sample from video for analysis"""
    try:
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio:
            temp_audio_path = temp_audio.name
        
        # Use ffmpeg to extract audio sample
        cmd = [
            'ffmpeg', '-y',
            '-ss', str(start_time),
            '-i', video_path,
            '-t', str(duration),
            '-vn',  # No video
            '-acodec', 'pcm_s16le',
            '-ar', '16000',  # 16kHz sample rate for WhisperX
            '-ac', '1',  # Mono
            temp_audio_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"FFmpeg extraction failed: {result.stderr}", file=sys.stderr)
            return None
            
        return temp_audio_path
    except Exception as e:
        print(f"Audio extraction error: {e}", file=sys.stderr)
        return None

def detect_language_whisperx(audio_path, model="base", confidence_threshold=0.7):
    """Detect language using WhisperX"""
    try:
        import whisperx
        
        # Load model with CPU device for compatibility
        device = "cpu"
        model = whisperx.load_model(model, device)
        
        # Load audio
        audio = whisperx.load_audio(audio_path)
        
        # Transcribe
        result = model.transcribe(audio, batch_size=16)
        
        # Get detected language with confidence
        if result.get("language") and result.get("language_probability", 0) >= confidence_threshold:
            return {
                "language": result["language"],
                "confidence": result.get("language_probability", 0),
                "method": "whisperx"
            }
        
        return None
    except Exception as e:
        print(f"WhisperX detection error: {e}", file=sys.stderr)
        return None

def detect_language_lingua(audio_path, confidence_threshold=0.7):
    """Detect language using Lingua with text from speech"""
    try:
        import whisperx
        from lingua import Language, LanguageDetectorBuilder
        
        # First transcribe with WhisperX
        device = "cpu"
        model = whisperx.load_model("base", device)
        audio = whisperx.load_audio(audio_path)
        result = model.transcribe(audio, batch_size=16)
        
        if not result.get("segments"):
            return None
            
        # Extract text from segments
        text_segments = []
        for segment in result["segments"]:
            if segment.get("text"):
                text_segments.append(segment["text"].strip())
        
        if not text_segments:
            return None
            
        full_text = " ".join(text_segments)
        if len(full_text.strip()) < 10:  # Need sufficient text
            return None
        
        # Build Lingua detector for multiple languages
        languages = [
            Language.ENGLISH, Language.SPANISH, Language.FRENCH, Language.GERMAN,
            Language.ITALIAN, Language.PORTUGUESE, Language.RUSSIAN, Language.CHINESE,
            Language.JAPANESE, Language.KOREAN, Language.ARABIC, Language.HINDI,
            Language.DUTCH, Language.SWEDISH, Language.NORWEGIAN, Language.DANISH,
            Language.FINNISH, Language.POLISH, Language.CZECH, Language.HUNGARIAN
        ]
        
        detector = LanguageDetectorBuilder.from_languages(*languages).build()
        
        # Detect language with confidence
        detection = detector.detect_language_of(full_text)
        confidence_values = detector.compute_language_confidence_values(full_text)
        
        if detection and confidence_values:
            confidence = max([cv.value for cv in confidence_values if cv.language == detection])
            
            if confidence >= confidence_threshold:
                # Convert Lingua language to ISO 639-1 code
                language_code = detection.iso_code_639_1.name.lower()
                return {
                    "language": language_code,
                    "confidence": confidence,
                    "method": "lingua",
                    "text_sample": full_text[:100] + "..." if len(full_text) > 100 else full_text
                }
        
        return None
    except Exception as e:
        print(f"Lingua detection error: {e}", file=sys.stderr)
        return None

def main():
    if len(sys.argv) != 6:
        print("Usage: python script.py <video_path> <model> <confidence> <sample_duration> <enable_mixed>", file=sys.stderr)
        sys.exit(1)
    
    video_path = sys.argv[1]
    model = sys.argv[2]
    confidence_threshold = float(sys.argv[3])
    sample_duration = int(sys.argv[4])
    enable_mixed = sys.argv[5].lower() == "true"
    
    try:
        # Extract audio sample
        audio_path = extract_audio_sample(video_path, duration=sample_duration)
        if not audio_path:
            print(json.dumps({"error": "Failed to extract audio sample"}))
            sys.exit(1)
        
        try:
            # Try WhisperX detection first
            result = detect_language_whisperx(audio_path, model, confidence_threshold)
            
            # If WhisperX fails or mixed language is enabled, try Lingua
            if not result and enable_mixed:
                result = detect_language_lingua(audio_path, confidence_threshold)
            
            if result:
                print(json.dumps({
                    "success": True,
                    "language": result["language"],
                    "confidence": result["confidence"],
                    "method": result["method"],
                    "text_sample": result.get("text_sample", "")
                }))
            else:
                print(json.dumps({
                    "success": False,
                    "error": "No language detected with sufficient confidence"
                }))
        
        finally:
            # Clean up temporary audio file
            if audio_path and os.path.exists(audio_path):
                os.unlink(audio_path)
    
    except Exception as e:
        print(json.dumps({
            "success": False,
            "error": f"Detection failed: {str(e)}",
            "traceback": traceback.format_exc()
        }))
        sys.exit(1)

if __name__ == "__main__":
    main()
`;

  // Write temporary Python script
  const { execSync } = require("child_process");
  const path = require("path");
  const fs = require("fs");

  const tempDir = librarySettings.cache || "/tmp";
  const tempScriptPath = path.join(
    tempDir,
    `ai_language_detection_${Date.now()}.py`
  );

  try {
    fs.writeFileSync(tempScriptPath, tempScript);
    response.infoLog += `☑ Created temporary detection script: ${tempScriptPath}\n`;

    // Execute AI language detection
    const aiCommand = [
      pythonBin,
      tempScriptPath,
      `"${file.file}"`,
      whisperModel,
      confidenceThreshold.toString(),
      sampleDuration.toString(),
      enableMixedLanguage.toString(),
    ].join(" ");

    response.infoLog += `☑ Executing AI language detection...\n`;
    response.infoLog += `Command: ${aiCommand}\n`;

    const aiResult = execSync(aiCommand, {
      encoding: "utf8",
      timeout: 300000, // 5 minute timeout
      maxBuffer: 1024 * 1024, // 1MB buffer
    });

    const detectionResult = JSON.parse(aiResult.trim());

    if (detectionResult.success) {
      const detectedLanguage = detectionResult.language;
      const confidence = detectionResult.confidence;
      const method = detectionResult.method;

      response.infoLog += `☑ Language detected: ${detectedLanguage} (confidence: ${confidence.toFixed(
        3
      )}, method: ${method})\n`;

      if (detectionResult.text_sample) {
        response.infoLog += `☑ Text sample: ${detectionResult.text_sample}\n`;
      }

      // Build FFmpeg command to add language metadata to audio streams
      let ffmpegCommandInsert = "";
      let outputStreamIndex = 0;

      streams.forEach((stream, inputStreamIndex) => {
        // Map each input stream individually
        ffmpegCommandInsert += `-map 0:${inputStreamIndex} `;

        if (stream.codec_type && stream.codec_type.toLowerCase() === "audio") {
          // Add language metadata to audio streams
          ffmpegCommandInsert += `-metadata:s:${outputStreamIndex} language=${detectedLanguage} `;
          response.infoLog += `☑ Adding language metadata '${detectedLanguage}' to audio stream ${outputStreamIndex}.\n`;
        }

        outputStreamIndex++;
      });

      if (ffmpegCommandInsert) {
        response.processFile = true;
        response.preset = `, ${ffmpegCommandInsert}-c copy -max_muxing_queue_size 9999`;
        response.reQueueAfter = true;
        response.infoLog += `✔ AI language detection completed successfully. Language '${detectedLanguage}' added to audio streams.\n`;
      }
    } else {
      response.infoLog += `☒ AI language detection failed: ${detectionResult.error}\n`;
      if (detectionResult.traceback) {
        response.infoLog += `Debug info: ${detectionResult.traceback}\n`;
      }
      response.processFile = false;
    }
  } catch (error) {
    response.infoLog += `☒ AI language detection error: ${error.message}\n`;
    response.processFile = false;
  } finally {
    // Clean up temporary script
    try {
      if (fs.existsSync(tempScriptPath)) {
        fs.unlinkSync(tempScriptPath);
        response.infoLog += `☑ Cleaned up temporary script.\n`;
      }
    } catch (cleanupError) {
      response.infoLog += `⚠ Failed to clean up temporary script: ${cleanupError.message}\n`;
    }
  }

  return response;
};

module.exports.details = details;
module.exports.plugin = plugin;
