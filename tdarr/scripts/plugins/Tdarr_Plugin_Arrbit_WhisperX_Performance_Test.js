const details = () => ({
  id: "Tdarr_Plugin_Arrbit_WhisperX_Performance_Test",
  Stage: "Pre-processing",
  Name: "Arrbit - WhisperX Performance Test",
  Type: "Video",
  Operation: "Transcode",
  Description: `Performance test plugin that transcribes the entire audio track using WhisperX to measure transcription speed and accuracy. 
    Extracts complete audio, runs full WhisperX pipeline with timing measurements, and reports performance metrics.
    Supports Intel Arc GPU acceleration via Intel Extension for PyTorch.
    All temporary data is stored in /app/arrbit/data/temp for analysis.`,
  Version: "1.1",
  Tags: "pre-processing,performance,whisperx,transcription,benchmark,timing,intel-arc,gpu-acceleration",
  Inputs: [
    {
      name: "whisper_model",
      type: "string",
      defaultValue: "large-v2",
      inputUI: {
        type: "dropdown",
        options: [
          "tiny",
          "base",
          "small",
          "medium",
          "large",
          "large-v2",
          "large-v3",
        ],
      },
      tooltip: `WhisperX model size to use for transcription performance test.
        Larger models are more accurate but slower and use more memory.
        \\nModel sizes:\\n
        tiny: ~39 MB, fastest
        base: ~74 MB, good balance
        small: ~244 MB, better accuracy
        medium: ~769 MB, high accuracy
        large: ~1550 MB, highest accuracy
        large-v2: ~1550 MB, latest large model
        large-v3: ~1550 MB, newest large model`,
    },
    {
      name: "batch_size",
      type: "number",
      defaultValue: 16,
      inputUI: {
        type: "text",
      },
      tooltip: `Batch size for WhisperX transcription processing.
        Higher values are faster but use more memory.
        Reduce if running out of memory.
        \\nExample:\\n
        16`,
    },
    {
      name: "compute_type",
      type: "string",
      defaultValue: "float16",
      inputUI: {
        type: "dropdown",
        options: ["float16", "float32", "int8"],
      },
      tooltip: `Compute precision for WhisperX model.
        float16: Good balance of speed and accuracy (GPU)
        float32: Higher accuracy, slower (CPU/GPU)
        int8: Fastest, lower memory, may reduce accuracy
        \\nExample:\\n
        float16`,
    },
    {
      name: "device_type",
      type: "string",
      defaultValue: "auto",
      inputUI: {
        type: "dropdown",
        options: ["auto", "cpu", "intel_gpu"],
      },
      tooltip: `Device to use for WhisperX transcription processing.
        auto: Automatically select best available device (Intel GPU > CPU)
        cpu: Force CPU-only processing (slower but compatible)
        intel_gpu: Force Intel Arc GPU processing (requires Intel Extension for PyTorch)
        \\nNote: Intel Arc A380 GPU support requires Intel Extension for PyTorch.
        \\nExample:\\n
        auto`,
    },
    {
      name: "enable_alignment",
      type: "boolean",
      defaultValue: true,
      inputUI: {
        type: "dropdown",
        options: ["false", "true"],
      },
      tooltip: `Enable forced alignment for accurate word-level timestamps.
        This improves timestamp accuracy but adds processing time.
        \\nExample:\\n
        true`,
    },
    {
      name: "enable_diarization",
      type: "boolean",
      defaultValue: false,
      inputUI: {
        type: "dropdown",
        options: ["false", "true"],
      },
      tooltip: `Enable speaker diarization to identify different speakers.
        Significantly increases processing time but provides speaker segmentation.
        \\nExample:\\n
        false`,
    },
    {
      name: "save_transcription",
      type: "boolean",
      defaultValue: true,
      inputUI: {
        type: "dropdown",
        options: ["false", "true"],
      },
      tooltip: `Save the full transcription results to file for analysis.
        Files saved in /app/arrbit/data/temp with timestamps and performance metrics.
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

  response.infoLog += "=== WhisperX Performance Test Plugin Started ===\n";
  response.infoLog += `File: ${file.file}\n`;
  response.infoLog += `File medium: ${file.fileMedium}\n`;
  response.infoLog += `Container: ${file.container}\n`;

  // Check if the file is a video
  if (file.fileMedium !== "video") {
    response.infoLog += "☒ File is not a video.\n";
    response.processFile = false;
    return response;
  }

  const streams = file.ffProbeData.streams;

  response.infoLog += `Streams available: ${
    streams ? streams.length : "none"
  }\n`;
  if (streams && streams.length > 0) {
    response.infoLog += `Stream types: ${streams
      .map((s) => s.codec_type)
      .join(", ")}\n`;
  }

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
    response.infoLog += "☒ No audio streams found for transcription.\n";
    response.processFile = false;
    return response;
  }

  response.infoLog += `☑ Found ${audioStreams.length} audio stream(s) for transcription performance test.\n`;

  // Get audio stream duration for performance calculations
  const primaryAudioStream = audioStreams[0];
  response.infoLog += `Primary audio stream duration: ${primaryAudioStream.duration}\n`;
  const audioDuration = parseFloat(primaryAudioStream.duration) || 0;

  response.infoLog += `Parsed audio duration: ${audioDuration}\n`;

  if (audioDuration === 0) {
    response.infoLog +=
      "☒ Could not determine audio duration from stream metadata.\n";
    // Let's try to continue anyway for testing
    response.infoLog +=
      "⚠ Continuing with unknown duration for testing purposes...\n";
  }

  if (audioDuration > 0) {
    response.infoLog += `☑ Audio duration: ${audioDuration.toFixed(
      2
    )} seconds (${(audioDuration / 60).toFixed(2)} minutes)\n`;
  } else {
    response.infoLog += `⚠ Audio duration unknown, will be determined during processing\n`;
  }

  // Validate input parameters
  response.infoLog += `Raw inputs: ${JSON.stringify(inputs)}\n`;

  const whisperModel = inputs.whisper_model || "large-v2";
  const validModels = [
    "tiny",
    "base",
    "small",
    "medium",
    "large",
    "large-v2",
    "large-v3",
  ];
  if (!validModels.includes(whisperModel)) {
    response.infoLog += `☒ Invalid WhisperX model: ${whisperModel}. Must be one of: ${validModels.join(
      ", "
    )}\n`;
    response.processFile = false;
    return response;
  }

  const batchSize = parseInt(inputs.batch_size, 10);
  if (isNaN(batchSize) || batchSize < 1 || batchSize > 64) {
    response.infoLog += "☒ Invalid batch size. Must be between 1 and 64.\n";
    response.processFile = false;
    return response;
  }

  const computeType = inputs.compute_type || "float16";
  const validComputeTypes = ["float16", "float32", "int8"];
  if (!validComputeTypes.includes(computeType)) {
    response.infoLog += `☒ Invalid compute type: ${computeType}. Must be one of: ${validComputeTypes.join(
      ", "
    )}\n`;
    response.processFile = false;
    return response;
  }

  const deviceType = inputs.device_type || "auto";
  const validDeviceTypes = ["auto", "cpu", "intel_gpu"];
  if (!validDeviceTypes.includes(deviceType)) {
    response.infoLog += `☒ Invalid device type: ${deviceType}. Must be one of: ${validDeviceTypes.join(
      ", "
    )}\n`;
    response.processFile = false;
    return response;
  }

  const enableAlignment =
    inputs.enable_alignment === true || inputs.enable_alignment === "true";

  const enableDiarization =
    inputs.enable_diarization === true || inputs.enable_diarization === "true";

  const saveTranscription =
    inputs.save_transcription === true || inputs.save_transcription === "true";

  response.infoLog += `☑ Configuration: model=${whisperModel}, batch_size=${batchSize}, compute_type=${computeType}, device=${deviceType}, alignment=${enableAlignment}, diarization=${enableDiarization}\n`;

  // Build Python command for WhisperX performance test
  const { execSync } = require("child_process");
  const path = require("path");
  const fs = require("fs");

  const pythonEnvPath = "/app/arrbit/environments/ai-language-detection";
  const pythonBin = `${pythonEnvPath}/bin/python`;

  response.infoLog += `Python binary path: ${pythonBin}\n`;

  // Test Python availability first
  try {
    const pythonTest = execSync(`${pythonBin} --version`, {
      encoding: "utf8",
      timeout: 10000,
    });
    response.infoLog += `Python version: ${pythonTest.trim()}\n`;
  } catch (pythonError) {
    response.infoLog += `☒ Python test failed: ${pythonError.message}\n`;
    response.infoLog += `This suggests the Python environment is not properly configured.\n`;
    response.processFile = false;
    return response;
  }

  // Create temporary script for full transcription performance test
  const tempScript = `
import sys
import os
import json
import traceback
import subprocess
import tempfile
import time
import gc
from pathlib import Path
from datetime import datetime

def extract_full_audio(video_path, temp_dir, log_file):
    """Extract complete audio track from video for transcription"""
    try:
        # Create unique filename for this test
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = Path(video_path).stem
        audio_filename = f"{base_name}_{timestamp}_full_audio.wav"
        audio_path = os.path.join(temp_dir, audio_filename)
        
        # Use ffmpeg to extract complete audio track at 16kHz for WhisperX
        cmd = [
            'ffmpeg', '-y',
            '-i', video_path,
            '-vn',  # No video
            '-acodec', 'pcm_s16le',
            '-ar', '16000',  # 16kHz sample rate for WhisperX
            '-ac', '1',  # Mono
            audio_path
        ]
        
        log_message = f"[{datetime.now().strftime('%H:%M:%S')}] Extracting audio to: {audio_path}"
        print(log_message, file=sys.stderr)
        with open(log_file, 'a') as f:
            f.write(log_message + "\\n")
            f.write(f"[{datetime.now().strftime('%H:%M:%S')}] FFmpeg command: {' '.join(cmd)}\\n")
            f.flush()
        
        start_time = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True)
        extraction_time = time.time() - start_time
        
        if result.returncode != 0:
            error_msg = f"[{datetime.now().strftime('%H:%M:%S')}] FFmpeg extraction failed: {result.stderr}"
            print(error_msg, file=sys.stderr)
            with open(log_file, 'a') as f:
                f.write(error_msg + "\\n")
                f.flush()
            return None, 0
            
        # Get audio file size
        audio_size = os.path.getsize(audio_path) if os.path.exists(audio_path) else 0
        success_msg = f"[{datetime.now().strftime('%H:%M:%S')}] Audio extraction completed in {extraction_time:.2f}s, size: {audio_size} bytes"
        print(success_msg, file=sys.stderr)
        with open(log_file, 'a') as f:
            f.write(success_msg + "\\n")
            f.flush()
        
        return audio_path, extraction_time
    except Exception as e:
        error_msg = f"[{datetime.now().strftime('%H:%M:%S')}] Audio extraction error: {e}"
        print(error_msg, file=sys.stderr)
        with open(log_file, 'a') as f:
            f.write(error_msg + "\\n")
            f.flush()
        return None, 0

def transcribe_with_whisperx(audio_path, model_name, batch_size, compute_type, device_type, enable_alignment, enable_diarization, log_file, progress_file):
    """Perform full transcription with WhisperX and timing measurements"""
    
    # Debug the parameters passed
    print(f"DEBUG: transcribe_with_whisperx called with:", file=sys.stderr)
    print(f"  audio_path: {audio_path}", file=sys.stderr)
    print(f"  model_name: {model_name}", file=sys.stderr)
    print(f"  batch_size: {batch_size}", file=sys.stderr)
    print(f"  compute_type: {compute_type}", file=sys.stderr)
    print(f"  device_type: {device_type}", file=sys.stderr)
    print(f"  log_file: {log_file}", file=sys.stderr)
    print(f"  progress_file: {progress_file}", file=sys.stderr)
    print(f"  log_file exists: {os.path.exists(os.path.dirname(log_file)) if log_file else 'None'}", file=sys.stderr)
    print(f"  progress_file dir exists: {os.path.exists(os.path.dirname(progress_file)) if progress_file else 'None'}", file=sys.stderr)
    
    try:
        def log_progress(message):
            timestamp = datetime.now().strftime('%H:%M:%S')
            full_message = f"[{timestamp}] {message}"
            print(full_message, file=sys.stderr)
            
            # Write to log file
            try:
                with open(log_file, 'a') as f:
                    f.write(full_message + "\\n")
                    f.flush()
            except Exception as log_e:
                print(f"[{timestamp}] Error writing to log file {log_file}: {log_e}", file=sys.stderr)
            
            # Write to progress file
            try:
                with open(progress_file, 'w') as f:
                    f.write(f"WhisperX Performance Test Progress\\n")
                    f.write(f"Current Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\\n")
                    f.write(f"Status: {message}\\n")
                    f.flush()
            except Exception as prog_e:
                print(f"[{timestamp}] Error writing to progress file {progress_file}: {prog_e}", file=sys.stderr)
        
        log_progress("Starting WhisperX import...")
        import whisperx
        import torch
        log_progress("WhisperX imported successfully")
        
        # Import Intel Extension for PyTorch if Intel GPU is requested
        intel_gpu_available = False
        try:
            import intel_extension_for_pytorch as ipex
            if torch.xpu.is_available():
                intel_gpu_available = True
                device_count = torch.xpu.device_count()
                device_name = torch.xpu.get_device_name(0) if device_count > 0 else "Unknown Intel GPU"
                log_progress(f"Intel Extension for PyTorch available with {device_count} XPU device(s): {device_name}")
            else:
                log_progress("Intel Extension for PyTorch installed but no XPU devices detected")
        except ImportError:
            log_progress("Intel Extension for PyTorch not available - only CPU processing available")
        except Exception as e:
            log_progress(f"Intel GPU detection error: {e}")
        
        performance_metrics = {
            "model_loading_time": 0,
            "transcription_time": 0,
            "alignment_time": 0,
            "diarization_time": 0,
            "total_processing_time": 0,
            "memory_usage": {},
            "model_info": {
                "name": model_name,
                "batch_size": batch_size,
                "compute_type": compute_type
            },
            "device_info": {}
        }
        
        # Device selection logic
        requested_device_type = device_type
        if requested_device_type == "auto":
            if intel_gpu_available:
                device = "xpu"
                log_progress("Auto-selected Intel XPU device (Intel Arc GPU)")
            else:
                device = "cpu"
                log_progress("Auto-selected CPU device (Intel GPU not available)")
        elif requested_device_type == "intel_gpu":
            if intel_gpu_available:
                device = "xpu"
                log_progress("Force-selected Intel XPU device (Intel Arc GPU)")
            else:
                device = "cpu"
                log_progress("Intel GPU requested but not available - falling back to CPU")
        else:  # device_type == "cpu"
            device = "cpu"
            log_progress("Force-selected CPU device")
        
        performance_metrics["device_info"] = {
            "requested_device": requested_device_type,
            "actual_device": device,
            "intel_gpu_available": intel_gpu_available
        }
        
        log_progress(f"Using device: {device} (requested: {requested_device_type})")
        
        # 1. Load WhisperX model with timing
        log_progress(f"Loading WhisperX model: {model_name}")
        log_progress(f"Model config: batch_size={batch_size}, compute_type={compute_type}")
        start_time = time.time()
        
        # Create model cache directory info
        model_cache_info = f"Model will be cached in ~/.cache/whisper or similar location"
        log_progress(model_cache_info)
        
        model = whisperx.load_model(model_name, device, compute_type=compute_type)
        model_loading_time = time.time() - start_time
        performance_metrics["model_loading_time"] = model_loading_time
        log_progress(f"Model loaded successfully in {model_loading_time:.2f}s")
        
        # Load audio
        log_progress("Loading audio for transcription...")
        audio = whisperx.load_audio(audio_path)
        audio_duration = len(audio) / 16000.0  # 16kHz sample rate
        log_progress(f"Audio loaded successfully, duration: {audio_duration:.2f}s ({audio_duration/60:.2f} minutes)")
        
        # 2. Transcribe with timing
        log_progress("Starting transcription... This may take several minutes for long audio.")
        log_progress(f"Expected processing time: ~{audio_duration/60:.1f} minutes for {model_name} model")
        start_time = time.time()
        
        # Run transcription
        result = model.transcribe(audio, batch_size=batch_size)
        transcription_time = time.time() - start_time
        performance_metrics["transcription_time"] = transcription_time
        performance_metrics["audio_duration"] = audio_duration
        performance_metrics["speed_ratio"] = audio_duration / transcription_time if transcription_time > 0 else 0
        
        speed_analysis = f"faster than realtime" if performance_metrics['speed_ratio'] > 1 else f"slower than realtime"
        log_progress(f"Transcription completed in {transcription_time:.2f}s (speed ratio: {performance_metrics['speed_ratio']:.2f}x - {speed_analysis})")
        
        # Log transcription results
        segments_count = len(result.get("segments", []))
        detected_language = result.get("language", "unknown")
        log_progress(f"Transcription results: {segments_count} segments, language: {detected_language}")
        
        # Clean up model to free memory
        log_progress("Cleaning up transcription model from memory...")
        del model
        gc.collect()
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
        elif device == "xpu":
            torch.xpu.empty_cache()
        log_progress("Model cleanup completed")
        
        # 3. Optional: Forced alignment for accurate timestamps
        if enable_alignment and result.get("language"):
            log_progress(f"Starting forced alignment for language: {result['language']}")
            start_time = time.time()
            try:
                model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
                log_progress("Alignment model loaded, running alignment...")
                result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)
                alignment_time = time.time() - start_time
                performance_metrics["alignment_time"] = alignment_time
                log_progress(f"Alignment completed successfully in {alignment_time:.2f}s")
                
                # Clean up alignment model
                del model_a
                gc.collect()
                log_progress("Alignment model cleanup completed")
            except Exception as e:
                log_progress(f"Alignment failed: {e}")
                performance_metrics["alignment_time"] = 0
        else:
            log_progress("Forced alignment skipped (disabled or no language detected)")
        
        # 4. Optional: Speaker diarization
        if enable_diarization:
            log_progress("Speaker diarization requested...")
            start_time = time.time()
            try:
                # Note: Diarization requires HuggingFace token for pyannote models
                # This is a placeholder - would need proper token configuration
                log_progress("Diarization skipped - requires HuggingFace token configuration")
                log_progress("To enable: set HF_TOKEN environment variable with pyannote.audio access")
                diarization_time = 0
            except Exception as e:
                log_progress(f"Diarization failed: {e}")
                diarization_time = 0
            performance_metrics["diarization_time"] = diarization_time
        else:
            log_progress("Speaker diarization skipped (disabled)")
        
        # Calculate total processing time
        performance_metrics["total_processing_time"] = (
            performance_metrics["model_loading_time"] + 
            performance_metrics["transcription_time"] + 
            performance_metrics["alignment_time"] + 
            performance_metrics["diarization_time"]
        )
        
        # Count segments and words
        segments = result.get("segments", [])
        total_words = sum(len(segment.get("text", "").split()) for segment in segments)
        performance_metrics["segment_count"] = len(segments)
        performance_metrics["word_count"] = total_words
        performance_metrics["words_per_second"] = total_words / transcription_time if transcription_time > 0 else 0
        
        log_progress(f"Final stats: {len(segments)} segments, {total_words} words, {performance_metrics['words_per_second']:.2f} words/sec")
        log_progress("WhisperX transcription pipeline completed successfully!")
        
        return result, performance_metrics
        
    except Exception as e:
        error_msg = f"WhisperX transcription error: {e}"
        print(error_msg, file=sys.stderr)
        with open(log_file, 'a') as f:
            f.write(f"[{datetime.now().strftime('%H:%M:%S')}] {error_msg}\\n")
            f.write(f"[{datetime.now().strftime('%H:%M:%S')}] Traceback: {traceback.format_exc()}\\n")
            f.flush()
        return None, {"error": str(e), "traceback": traceback.format_exc()}

def save_results(result, performance_metrics, temp_dir, base_filename, log_file):
    """Save transcription results and performance metrics to files"""
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        def log_progress(message):
            timestamp_str = datetime.now().strftime('%H:%M:%S')
            full_message = f"[{timestamp_str}] {message}"
            print(full_message, file=sys.stderr)
            with open(log_file, 'a') as f:
                f.write(full_message + "\\n")
                f.flush()
        
        log_progress("Saving transcription results to files...")
        
        # Save transcription result
        transcription_file = os.path.join(temp_dir, f"{base_filename}_{timestamp}_transcription.json")
        with open(transcription_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        log_progress(f"Transcription saved: {transcription_file}")
        
        # Save performance metrics
        metrics_file = os.path.join(temp_dir, f"{base_filename}_{timestamp}_performance.json")
        with open(metrics_file, 'w', encoding='utf-8') as f:
            json.dump(performance_metrics, f, indent=2)
        log_progress(f"Performance metrics saved: {metrics_file}")
        
        # Create human-readable summary
        summary_file = os.path.join(temp_dir, f"{base_filename}_{timestamp}_summary.txt")
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(f"WhisperX Performance Test Summary\\n")
            f.write(f"================================\\n\\n")
            f.write(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\\n")
            f.write(f"Model: {performance_metrics.get('model_info', {}).get('name', 'unknown')}\\n")
            f.write(f"Batch Size: {performance_metrics.get('model_info', {}).get('batch_size', 'unknown')}\\n")
            f.write(f"Compute Type: {performance_metrics.get('model_info', {}).get('compute_type', 'unknown')}\\n\\n")
            
            f.write(f"Audio Duration: {performance_metrics.get('audio_duration', 0):.2f} seconds\\n")
            f.write(f"Model Loading Time: {performance_metrics.get('model_loading_time', 0):.2f} seconds\\n")
            f.write(f"Transcription Time: {performance_metrics.get('transcription_time', 0):.2f} seconds\\n")
            f.write(f"Alignment Time: {performance_metrics.get('alignment_time', 0):.2f} seconds\\n")
            f.write(f"Diarization Time: {performance_metrics.get('diarization_time', 0):.2f} seconds\\n")
            f.write(f"Total Processing Time: {performance_metrics.get('total_processing_time', 0):.2f} seconds\\n\\n")
            
            f.write(f"Speed Ratio: {performance_metrics.get('speed_ratio', 0):.2f}x realtime\\n")
            f.write(f"Segments: {performance_metrics.get('segment_count', 0)}\\n")
            f.write(f"Words: {performance_metrics.get('word_count', 0)}\\n")
            f.write(f"Words per Second: {performance_metrics.get('words_per_second', 0):.2f}\\n")
            
            # Add transcription sample
            segments = result.get("segments", [])
            if segments:
                f.write(f"\\n--- Sample Transcription (first 3 segments) ---\\n")
                for i, segment in enumerate(segments[:3]):
                    start_time = segment.get("start", 0)
                    end_time = segment.get("end", 0)
                    text = segment.get("text", "").strip()
                    f.write(f"[{start_time:.2f}s - {end_time:.2f}s]: {text}\\n")
        
        log_progress(f"Summary saved: {summary_file}")
        
        # Create detailed transcription text file
        transcript_text_file = os.path.join(temp_dir, f"{base_filename}_{timestamp}_transcript.txt")
        with open(transcript_text_file, 'w', encoding='utf-8') as f:
            f.write(f"Full Transcription - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\\n")
            f.write(f"=" * 50 + "\\n\\n")
            
            segments = result.get("segments", [])
            for segment in segments:
                start_time = segment.get("start", 0)
                end_time = segment.get("end", 0)
                text = segment.get("text", "").strip()
                f.write(f"[{start_time:.2f}s - {end_time:.2f}s]: {text}\\n")
        
        log_progress(f"Full transcript saved: {transcript_text_file}")
        
        return {
            "transcription_file": transcription_file,
            "metrics_file": metrics_file,
            "summary_file": summary_file,
            "transcript_text_file": transcript_text_file
        }
        
    except Exception as e:
        error_msg = f"Failed to save results: {e}"
        print(error_msg, file=sys.stderr)
        with open(log_file, 'a') as f:
            f.write(f"[{datetime.now().strftime('%H:%M:%S')}] {error_msg}\\n")
            f.flush()
        return None

def main():
    if len(sys.argv) != 9:
        print("Usage: python script.py <video_path> <model> <batch_size> <compute_type> <device_type> <enable_alignment> <enable_diarization> <save_results>", file=sys.stderr)
        sys.exit(1)
    
    video_path = sys.argv[1]
    model_name = sys.argv[2]
    batch_size = int(sys.argv[3])
    compute_type = sys.argv[4]
    device_type = sys.argv[5]
    enable_alignment = sys.argv[6].lower() == "true"
    enable_diarization = sys.argv[7].lower() == "true"
    save_results_flag = sys.argv[8].lower() == "true"
    
    # Temp directory for all processing
    temp_dir = "/app/arrbit/data/temp"
    os.makedirs(temp_dir, exist_ok=True)
    
    # Create timestamped log file in temp directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_filename = Path(video_path).stem
    log_file = os.path.join(temp_dir, f"{base_filename}_{timestamp}_whisperx_log.txt")
    
    def log_progress(message):
        timestamp_str = datetime.now().strftime('%H:%M:%S')
        full_message = f"[{timestamp_str}] {message}"
        print(full_message, file=sys.stderr)
        with open(log_file, 'a') as f:
            f.write(full_message + "\\n")
            f.flush()
    
    try:
        total_start_time = time.time()
        
        log_progress("=== Starting WhisperX Performance Test ===")
        log_progress(f"Video file: {video_path}")
        log_progress(f"Model: {model_name}")
        log_progress(f"Batch size: {batch_size}")
        log_progress(f"Compute type: {compute_type}")
        log_progress(f"Device type: {device_type}")
        log_progress(f"Alignment: {enable_alignment}")
        log_progress(f"Diarization: {enable_diarization}")
        log_progress(f"Save results: {save_results_flag}")
        log_progress(f"Temp directory: {temp_dir}")
        log_progress(f"Log file: {log_file}")
        
        # Create progress tracking file
        progress_file = os.path.join(temp_dir, f"{base_filename}_{timestamp}_progress.txt")
        with open(progress_file, 'w') as f:
            f.write(f"WhisperX Performance Test Progress\\n")
            f.write(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\\n")
            f.write(f"Status: Starting audio extraction...\\n")
        
        log_progress(f"Progress tracking file: {progress_file}")
        
        # 1. Extract full audio
        log_progress("Step 1: Extracting audio...")
        audio_path, extraction_time = extract_full_audio(video_path, temp_dir, log_file)
        if not audio_path:
            log_progress("Failed to extract audio")
            print(json.dumps({"error": "Failed to extract audio"}))
            sys.exit(1)
        
        # Update progress
        with open(progress_file, 'w') as f:
            f.write(f"WhisperX Performance Test Progress\\n")
            f.write(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\\n")
            f.write(f"Status: Audio extracted, starting transcription...\\n")
            f.write(f"Audio file: {audio_path}\\n")
            f.write(f"Extraction time: {extraction_time:.2f}s\\n")
        
        log_progress(f"Audio extracted to: {audio_path} (took {extraction_time:.2f}s)")
        
        # 2. Transcribe with WhisperX
        log_progress("Step 2: Starting transcription with WhisperX...")
        result, performance_metrics = transcribe_with_whisperx(
            audio_path, model_name, batch_size, compute_type, device_type,
            enable_alignment, enable_diarization, log_file, progress_file
        )
        
        if not result:
            log_progress("Transcription failed")
            print(json.dumps({
                "success": False,
                "error": "Transcription failed",
                "performance_metrics": performance_metrics,
                "log_file": log_file,
                "progress_file": progress_file
            }))
            sys.exit(1)
        
        # Add extraction time to metrics
        performance_metrics["audio_extraction_time"] = extraction_time
        performance_metrics["total_test_time"] = time.time() - total_start_time
        
        # Update progress
        with open(progress_file, 'w') as f:
            f.write(f"WhisperX Performance Test Progress\\n")
            f.write(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\\n")
            f.write(f"Status: Transcription complete, processing results...\\n")
            f.write(f"Total processing time: {performance_metrics.get('total_processing_time', 0):.2f}s\\n")
            f.write(f"Speed ratio: {performance_metrics.get('speed_ratio', 0):.2f}x realtime\\n")
        
        # 3. Save results if requested
        saved_files = None
        if save_results_flag:
            log_progress("Step 3: Saving results...")
            saved_files = save_results(result, performance_metrics, temp_dir, base_filename, log_file)
            if saved_files:
                log_progress("Results saved successfully")
                for file_type, file_path in saved_files.items():
                    log_progress(f"  - {file_type}: {file_path}")
        
        # Final progress update
        with open(progress_file, 'w') as f:
            f.write(f"WhisperX Performance Test Progress\\n")
            f.write(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\\n")
            f.write(f"Completion Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\\n")
            f.write(f"Status: COMPLETED SUCCESSFULLY\\n")
            f.write(f"Total test time: {performance_metrics['total_test_time']:.2f}s\\n")
            f.write(f"Speed ratio: {performance_metrics.get('speed_ratio', 0):.2f}x realtime\\n")
            f.write(f"Segments: {len(result.get('segments', []))}\\n")
            f.write(f"Words: {performance_metrics.get('word_count', 0)}\\n")
            if saved_files:
                f.write(f"\\nSaved files:\\n")
                for file_type, file_path in saved_files.items():
                    f.write(f"  - {file_type}: {file_path}\\n")
        
        # 4. Output final results
        output = {
            "success": True,
            "transcription_complete": True,
            "language": result.get("language", "unknown"),
            "performance_metrics": performance_metrics,
            "segment_count": len(result.get("segments", [])),
            "audio_file": audio_path,
            "saved_files": saved_files,
            "log_file": log_file,
            "progress_file": progress_file,
            "temp_directory": temp_dir
        }
        
        log_progress("=== Test completed successfully ===")
        log_progress(f"Check {temp_dir} for all output files")
        
        print(json.dumps(output))
        
    except Exception as e:
        error_msg = f"Performance test failed: {str(e)}"
        log_progress(f"ERROR: {error_msg}")
        
        # Update progress with error
        try:
            with open(progress_file, 'w') as f:
                f.write(f"WhisperX Performance Test Progress\\n")
                f.write(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\\n")
                f.write(f"Error Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\\n")
                f.write(f"Status: FAILED\\n")
                f.write(f"Error: {error_msg}\\n")
                f.write(f"Full traceback:\\n{traceback.format_exc()}\\n")
        except:
            pass
        
        print(json.dumps({
            "success": False,
            "error": error_msg,
            "traceback": traceback.format_exc(),
            "log_file": log_file if 'log_file' in locals() else None,
            "progress_file": progress_file if 'progress_file' in locals() else None
        }))
        sys.exit(1)
    
    finally:
        # Cleanup temporary audio file if it exists
        if 'audio_path' in locals() and audio_path and os.path.exists(audio_path):
            try:
                # Keep audio file for analysis if save_results is enabled
                if not save_results_flag:
                    os.unlink(audio_path)
                    log_progress(f"Cleaned up temporary audio file: {audio_path}")
                else:
                    log_progress(f"Keeping audio file for analysis: {audio_path}")
            except Exception as e:
                log_progress(f"Warning: Could not clean up temporary file {audio_path}: {e}")

if __name__ == "__main__":
    main()
`;

  // Write temporary Python script
  const tempDir = librarySettings.cache || "/tmp";
  const tempScriptPath = path.join(
    tempDir,
    `whisperx_performance_test_${Date.now()}.py`
  );

  try {
    fs.writeFileSync(tempScriptPath, tempScript);
    response.infoLog += `☑ Created temporary performance test script: ${tempScriptPath}\n`;

    // Execute WhisperX performance test
    const testCommand = [
      pythonBin,
      tempScriptPath,
      `"${file.file}"`,
      whisperModel,
      batchSize.toString(),
      computeType,
      deviceType,
      enableAlignment.toString(),
      enableDiarization.toString(),
      saveTranscription.toString(),
    ].join(" ");

    response.infoLog += `☑ Starting WhisperX performance test...\n`;
    response.infoLog += `Command: ${testCommand}\n`;
    response.infoLog += `Expected audio duration: ${audioDuration.toFixed(
      2
    )}s - this will take some time for full transcription...\n`;

    const testResult = execSync(testCommand, {
      encoding: "utf8",
      timeout: 1800000, // 30 minute timeout for long audio
      maxBuffer: 5 * 1024 * 1024, // 5MB buffer for large outputs
    });

    const performanceResult = JSON.parse(testResult.trim());

    if (performanceResult.success) {
      const metrics = performanceResult.performance_metrics;
      const detectedLanguage = performanceResult.language;

      response.infoLog += `\n=== WhisperX Performance Test Results ===\n`;
      response.infoLog += `☑ Language detected: ${detectedLanguage}\n`;
      response.infoLog += `☑ Audio duration: ${
        metrics.audio_duration?.toFixed(2) || audioDuration.toFixed(2)
      } seconds\n`;
      response.infoLog += `☑ Model loading time: ${
        metrics.model_loading_time?.toFixed(2) || 0
      } seconds\n`;
      response.infoLog += `☑ Audio extraction time: ${
        metrics.audio_extraction_time?.toFixed(2) || 0
      } seconds\n`;
      response.infoLog += `☑ Transcription time: ${
        metrics.transcription_time?.toFixed(2) || 0
      } seconds\n`;

      if (enableAlignment && metrics.alignment_time > 0) {
        response.infoLog += `☑ Alignment time: ${metrics.alignment_time?.toFixed(
          2
        )} seconds\n`;
      }

      if (enableDiarization && metrics.diarization_time > 0) {
        response.infoLog += `☑ Diarization time: ${metrics.diarization_time?.toFixed(
          2
        )} seconds\n`;
      }

      response.infoLog += `☑ Total processing time: ${
        metrics.total_processing_time?.toFixed(2) || 0
      } seconds\n`;
      response.infoLog += `☑ Total test time: ${
        metrics.total_test_time?.toFixed(2) || 0
      } seconds\n`;
      response.infoLog += `☑ Speed ratio: ${
        metrics.speed_ratio?.toFixed(2) || 0
      }x realtime\n`;
      response.infoLog += `☑ Segments generated: ${
        performanceResult.segment_count || 0
      }\n`;
      response.infoLog += `☑ Words transcribed: ${metrics.word_count || 0}\n`;
      response.infoLog += `☑ Words per second: ${
        metrics.words_per_second?.toFixed(2) || 0
      }\n`;

      // Performance analysis
      const speedRatio = metrics.speed_ratio || 0;
      if (speedRatio > 1) {
        response.infoLog += `✔ Excellent performance: ${speedRatio.toFixed(
          2
        )}x faster than realtime\n`;
      } else if (speedRatio > 0.5) {
        response.infoLog += `⚠ Moderate performance: ${speedRatio.toFixed(
          2
        )}x realtime (slower than realtime)\n`;
      } else {
        response.infoLog += `☒ Slow performance: ${speedRatio.toFixed(
          2
        )}x realtime (significantly slower)\n`;
      }

      // File information
      if (saveTranscription && performanceResult.saved_files) {
        response.infoLog += `\n=== Saved Files ===\n`;
        if (performanceResult.saved_files.transcription_file) {
          response.infoLog += `☑ Transcription: ${performanceResult.saved_files.transcription_file}\n`;
        }
        if (performanceResult.saved_files.metrics_file) {
          response.infoLog += `☑ Performance metrics: ${performanceResult.saved_files.metrics_file}\n`;
        }
        if (performanceResult.saved_files.summary_file) {
          response.infoLog += `☑ Summary report: ${performanceResult.saved_files.summary_file}\n`;
        }
        if (performanceResult.saved_files.transcript_text_file) {
          response.infoLog += `☑ Full transcript: ${performanceResult.saved_files.transcript_text_file}\n`;
        }
        if (performanceResult.audio_file) {
          response.infoLog += `☑ Extracted audio: ${performanceResult.audio_file}\n`;
        }
      }

      // Logging and monitoring information
      response.infoLog += `\n=== Debug & Monitoring Files ===\n`;
      response.infoLog += `☑ Temp directory: ${
        performanceResult.temp_directory || "/app/arrbit/data/temp"
      }\n`;
      if (performanceResult.log_file) {
        response.infoLog += `☑ Detailed log: ${performanceResult.log_file}\n`;
      }
      if (performanceResult.progress_file) {
        response.infoLog += `☑ Progress tracking: ${performanceResult.progress_file}\n`;
      }
      response.infoLog += `\n📁 All temporary files and logs are stored in /app/arrbit/data/temp\n`;
      response.infoLog += `📊 Monitor progress in real-time by checking the progress file\n`;
      response.infoLog += `🔍 Check the detailed log file for complete execution trace\n`;

      response.infoLog += `\n✔ WhisperX performance test completed successfully!\n`;

      // For a performance test, we always return processFile: false
      // But we could set it to true if we wanted to continue with normal processing
      response.processFile = false;
    } else {
      response.infoLog += `☒ WhisperX performance test failed: ${performanceResult.error}\n`;

      // Add logging file information even for failures
      if (performanceResult.log_file) {
        response.infoLog += `🔍 Check detailed log: ${performanceResult.log_file}\n`;
      }
      if (performanceResult.progress_file) {
        response.infoLog += `📊 Check progress file: ${performanceResult.progress_file}\n`;
      }
      response.infoLog += `📁 Debug files available in: /app/arrbit/data/temp\n`;

      if (performanceResult.traceback) {
        response.infoLog += `Debug info: ${performanceResult.traceback}\n`;
      }
      response.processFile = false;
    }
  } catch (error) {
    response.infoLog += `☒ WhisperX performance test error: ${error.message}\n`;
    response.infoLog += `Error details: ${error.toString()}\n`;
    response.infoLog += `📁 Check /app/arrbit/data/temp for any debug files\n`;

    if (error.status) {
      response.infoLog += `Exit code: ${error.status}\n`;
    }
    if (error.stderr) {
      response.infoLog += `Stderr: ${error.stderr}\n`;
    }
    if (error.stdout) {
      response.infoLog += `Stdout: ${error.stdout}\n`;
    }
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
