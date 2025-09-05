const details = () => ({
  id: "Tdarr_Plugin_Arrbit_AI_Language_Detector",
  Stage: "Pre-processing",
  Name: "Arrbit - AI Language Detector",
  Type: "Video",
  Operation: "Transcode",
  Description: `Enhanced AI language detection using WhisperX with VAD and Lingua for maximum accuracy.
    Extracts 5 random 30-second audio samples from different timestamps for robust language detection.
    Uses WhisperX tiny model with int8 for optimal CPU performance and lingua-language-detector for confidence voting.
    Processes all available audio tracks for comprehensive language detection. Detection results are logged without applying metadata tags.
    All processing logged to /app/arrbit/data/temp.`,
  Version: "3.0",
  Tags: "pre-processing,ai,language,detection,speech,recognition,metadata,vad,lingua,multi-track",
  Inputs: [
    {
      name: "confidence_threshold",
      type: "number",
      defaultValue: 0.8,
      inputUI: {
        type: "text",
      },
      tooltip: `Minimum confidence threshold for language detection (0.0-1.0).
        Higher values require stronger consensus from the 5 audio samples.
        Recommended: 0.8 for reliable detection.
        \\nExample:\\n
        0.8`,
    },
    {
      name: "sample_count",
      type: "number",
      defaultValue: 5,
      inputUI: {
        type: "text",
      },
      tooltip: `Number of random audio samples to extract for language detection.
        More samples provide better accuracy for mixed-language content.
        Recommended: 5 samples for balanced speed and accuracy.
        \\nExample:\\n
        5`,
    },
    {
      name: "sample_duration",
      type: "number",
      defaultValue: 30,
      inputUI: {
        type: "text",
      },
      tooltip: `Duration in seconds of each audio sample for analysis.
        Longer samples provide better context but increase processing time.
        \\nExample:\\n
        30`,
    },
    {
      name: "majority_vote_threshold",
      type: "number",
      defaultValue: 0.6,
      inputUI: {
        type: "text",
      },
      tooltip: `Minimum percentage of samples that must agree on a language (0.0-1.0).
        Example: 0.6 means at least 60% of samples must detect the same language.
        \\nExample:\\n
        0.6`,
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

  response.infoLog += "=== Enhanced AI Language Detector Started ===\n";
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

  response.infoLog += `☑ Found ${audioStreams.length} audio stream(s) for enhanced multi-track language detection.\n`; // Get audio duration for sample selection - try multiple sources
  const primaryAudioStream = audioStreams[0];
  let audioDuration = 0;

  // Try to get duration from multiple sources in order of preference
  if (primaryAudioStream.duration) {
    audioDuration = parseFloat(primaryAudioStream.duration);
    response.infoLog += `Duration source: audio stream metadata\n`;
  } else if (file.ffProbeData.format && file.ffProbeData.format.duration) {
    audioDuration = parseFloat(file.ffProbeData.format.duration);
    response.infoLog += `Duration source: format metadata\n`;
  } else if (primaryAudioStream.tags && primaryAudioStream.tags.DURATION) {
    audioDuration = parseFloat(primaryAudioStream.tags.DURATION);
    response.infoLog += `Duration source: audio stream tags\n`;
  } else if (file.meta && file.meta.Duration) {
    // Parse duration from ExifTool format (HH:MM:SS.mmm)
    const durationStr = file.meta.Duration;
    const timeParts = durationStr.split(":");
    if (timeParts.length === 3) {
      const hours = parseFloat(timeParts[0]) || 0;
      const minutes = parseFloat(timeParts[1]) || 0;
      const seconds = parseFloat(timeParts[2]) || 0;
      audioDuration = hours * 3600 + minutes * 60 + seconds;
      response.infoLog += `Duration source: ExifTool metadata (${durationStr})\n`;
    }
  }

  response.infoLog += `Audio duration: ${audioDuration.toFixed(2)} seconds\n`;

  if (audioDuration === 0) {
    response.infoLog +=
      "☒ Could not determine audio duration from stream metadata.\n";
    response.processFile = false;
    return response;
  }

  if (audioDuration < 60) {
    response.infoLog +=
      "☒ Audio too short for reliable multi-sample detection (minimum 60 seconds required).\n";
    response.processFile = false;
    return response;
  }

  response.infoLog += `☑ Audio duration: ${audioDuration.toFixed(
    2
  )} seconds (${(audioDuration / 60).toFixed(
    2
  )} minutes) - sufficient for multi-sample detection\n`;

  // Validate input parameters
  response.infoLog += `Raw inputs: ${JSON.stringify(inputs)}\n`;

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

  const sampleCount = parseInt(inputs.sample_count, 10);
  if (isNaN(sampleCount) || sampleCount < 3 || sampleCount > 10) {
    response.infoLog += "☒ Invalid sample count. Must be between 3 and 10.\n";
    response.processFile = false;
    return response;
  }

  const sampleDuration = parseInt(inputs.sample_duration, 10);
  if (isNaN(sampleDuration) || sampleDuration < 15 || sampleDuration > 60) {
    response.infoLog +=
      "☒ Invalid sample duration. Must be between 15 and 60 seconds.\n";
    response.processFile = false;
    return response;
  }

  const majorityVoteThreshold = parseFloat(inputs.majority_vote_threshold);
  if (
    isNaN(majorityVoteThreshold) ||
    majorityVoteThreshold < 0.5 ||
    majorityVoteThreshold > 1
  ) {
    response.infoLog +=
      "☒ Invalid majority vote threshold. Must be between 0.5 and 1.0.\n";
    response.processFile = false;
    return response;
  }

  const useMkvpropedit = false; // Tagging logic removed

  response.infoLog += `☑ Configuration: samples=${sampleCount}, duration=${sampleDuration}s, confidence=${confidenceThreshold}, majority=${majorityVoteThreshold}\n`;

  // Check total required duration vs available duration
  const totalSampleDuration = sampleCount * sampleDuration;
  const minAudioDuration = totalSampleDuration + 60; // Add buffer for random selection

  if (audioDuration < minAudioDuration) {
    response.infoLog += `⚠ Audio duration (${audioDuration.toFixed(
      2
    )}s) is close to required sample duration (${totalSampleDuration}s). May use overlapping samples.\n`;
  }

  // Build Python command for enhanced AI language detection
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

  // Create enhanced temporary script for multi-sample language detection
  const tempScript = `
import sys
import os
import json
import traceback
import subprocess
import tempfile
import random
import time
import gc
from pathlib import Path
from datetime import datetime

def log_progress(message, log_file):
    """Log progress with timestamp"""
    timestamp = datetime.now().strftime('%H:%M:%S')
    full_message = f"[{timestamp}] {message}"
    print(full_message, file=sys.stderr)
    if log_file:
        try:
            with open(log_file, 'a') as f:
                f.write(full_message + "\\n")
                f.flush()
        except Exception as e:
            print(f"[{timestamp}] Error writing to log file: {e}", file=sys.stderr)

def generate_random_timestamps(audio_duration, sample_count, sample_duration):
    """Generate random non-overlapping timestamps for audio sampling"""
    try:
        # Ensure we have enough space for all samples
        max_start_time = audio_duration - sample_duration
        if max_start_time <= 0:
            # Audio too short, use overlapping samples
            timestamps = []
            for i in range(sample_count):
                start_time = (audio_duration / sample_count) * i
                start_time = max(0, min(start_time, audio_duration - sample_duration))
                timestamps.append(start_time)
            return timestamps
        
        # Generate random non-overlapping timestamps
        timestamps = []
        used_ranges = []
        
        for _ in range(sample_count):
            attempts = 0
            while attempts < 50:  # Max attempts to find non-overlapping timestamp
                start_time = random.uniform(0, max_start_time)
                end_time = start_time + sample_duration
                
                # Check for overlap with existing ranges
                overlap = False
                for used_start, used_end in used_ranges:
                    if not (end_time <= used_start or start_time >= used_end):
                        overlap = True
                        break
                
                if not overlap:
                    timestamps.append(start_time)
                    used_ranges.append((start_time, end_time))
                    break
                
                attempts += 1
            
            # If we couldn't find non-overlapping timestamp, use a spaced one
            if len(timestamps) < len(used_ranges):
                fallback_start = (audio_duration / sample_count) * len(timestamps)
                fallback_start = max(0, min(fallback_start, max_start_time))
                timestamps.append(fallback_start)
        
        return sorted(timestamps)
    except Exception as e:
        print(f"Error generating timestamps: {e}", file=sys.stderr)
        # Fallback: evenly spaced timestamps
        return [i * (audio_duration / sample_count) for i in range(sample_count)]

def extract_audio_samples(video_path, timestamps, sample_duration, temp_dir, audio_track_index, log_file):
    """Extract multiple audio samples from a specific audio track at specified timestamps"""
    try:
        log_progress(f"Extracting {len(timestamps)} audio samples from track {audio_track_index} of {sample_duration}s each", log_file)
        
        audio_samples = []
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = Path(video_path).stem
        
        for i, start_time in enumerate(timestamps):
            sample_filename = f"{base_name}_{timestamp_str}_track{audio_track_index}_sample_{i+1}.wav"
            sample_path = os.path.join(temp_dir, sample_filename)
            
            # Use ffmpeg to extract audio sample with VAD-friendly settings from specific audio track
            cmd = [
                'ffmpeg', '-y',
                '-ss', str(start_time),
                '-i', video_path,
                '-t', str(sample_duration),
                '-map', f'0:a:{audio_track_index}',  # Select specific audio track
                '-vn',  # No video
                '-acodec', 'pcm_s16le',
                '-ar', '16000',  # 16kHz sample rate for WhisperX
                '-ac', '1',  # Mono
                '-af', 'volume=1.5,highpass=f=80,lowpass=f=8000',  # Audio filtering for speech
                sample_path
            ]
            
            log_progress(f"Extracting sample {i+1}/{len(timestamps)} from track {audio_track_index} at {start_time:.2f}s", log_file)
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                log_progress(f"FFmpeg extraction failed for track {audio_track_index} sample {i+1}: {result.stderr}", log_file)
                continue
                
            # Verify file was created and has reasonable size
            if os.path.exists(sample_path) and os.path.getsize(sample_path) > 1000:
                audio_samples.append({
                    'path': sample_path,
                    'start_time': start_time,
                    'duration': sample_duration,
                    'index': i + 1,
                    'track_index': audio_track_index
                })
                log_progress(f"Track {audio_track_index} sample {i+1} extracted successfully: {os.path.getsize(sample_path)} bytes", log_file)
            else:
                log_progress(f"Track {audio_track_index} sample {i+1} extraction failed or file too small", log_file)
        
        log_progress(f"Successfully extracted {len(audio_samples)} audio samples from track {audio_track_index}", log_file)
        return audio_samples
        
    except Exception as e:
        log_progress(f"Audio extraction error for track {audio_track_index}: {e}", log_file)
        return []

def transcribe_sample_whisperx(sample_info, log_file):
    """Transcribe a single audio sample using WhisperX with VAD"""
    try:
        import whisperx
        
        sample_path = sample_info['path']
        sample_index = sample_info['index']
        
        log_progress(f"Transcribing sample {sample_index} with WhisperX tiny model", log_file)
        
        # Use CPU device with int8 for maximum performance
        device = "cpu"
        compute_type = "int8"
        
        # Load tiny model for speed (VAD preprocessing reduces hallucination)
        model = whisperx.load_model("tiny", device, compute_type=compute_type)
        
        # Load audio
        audio = whisperx.load_audio(sample_path)
        
        # Transcribe with batching and VAD preprocessing
        # WhisperX automatically applies VAD preprocessing to reduce hallucination
        result = model.transcribe(audio, batch_size=8)
        
        # Clean up model to free memory
        del model
        gc.collect()
        
        # Extract text from segments
        segments = result.get("segments", [])
        text_segments = []
        total_words = 0
        
        for segment in segments:
            if segment.get("text"):
                text = segment["text"].strip()
                if len(text) > 2:  # Skip very short segments
                    text_segments.append(text)
                    total_words += len(text.split())
        
        full_text = " ".join(text_segments)
        
        log_progress(f"Sample {sample_index}: extracted {len(text_segments)} segments, {total_words} words", log_file)
        
        return {
            'sample_index': sample_index,
            'text': full_text,
            'segment_count': len(text_segments),
            'word_count': total_words,
            'detected_language': result.get("language"),
            'language_confidence': result.get("language_probability", 0),
            'success': len(full_text.strip()) >= 10  # Need minimum text for reliable detection
        }
        
    except Exception as e:
        log_progress(f"WhisperX transcription error for sample {sample_info.get('index', '?')}: {e}", log_file)
        return {
            'sample_index': sample_info.get('index', 0),
            'text': "",
            'success': False,
            'error': str(e)
        }

def detect_language_lingua(text, sample_index, log_file):
    """Detect language using Lingua with high accuracy"""
    try:
        from lingua import Language, LanguageDetectorBuilder
        
        if len(text.strip()) < 10:
            log_progress(f"Sample {sample_index}: insufficient text for Lingua detection", log_file)
            return None
        
        # Build detector with all available spoken languages for maximum coverage
        # Using from_all_spoken_languages() to avoid issues with specific language constants
        detector = LanguageDetectorBuilder.from_all_spoken_languages()\\
            .with_minimum_relative_distance(0.25)\\
            .build()
        
        # Detect language
        detected_language = detector.detect_language_of(text)
        
        if not detected_language:
            log_progress(f"Sample {sample_index}: Lingua could not detect language", log_file)
            return None
        
        # Get confidence values for all languages
        confidence_values = detector.compute_language_confidence_values(text)
        confidence_dict = {cv.language: cv.value for cv in confidence_values}
        
        primary_confidence = confidence_dict.get(detected_language, 0)
        
        # Convert to ISO 639-1 code
        language_code = detected_language.iso_code_639_1.name.lower()
        
        log_progress(f"Sample {sample_index}: Lingua detected {language_code} with confidence {primary_confidence:.3f}", log_file)
        
        return {
            'language': language_code,
            'confidence': primary_confidence,
            'method': 'lingua',
            'text_length': len(text),
            'all_confidences': {lang.iso_code_639_1.name.lower(): conf for lang, conf in confidence_dict.items()}
        }
        
    except Exception as e:
        log_progress(f"Lingua detection error for sample {sample_index}: {e}", log_file)
        return None

def determine_final_language(detection_results, majority_threshold, confidence_threshold, log_file):
    """Determine final language using majority voting and confidence analysis"""
    try:
        log_progress("Analyzing detection results for final language determination", log_file)
        
        # Filter successful detections with sufficient confidence
        valid_detections = []
        for result in detection_results:
            if (result.get('lingua_detection') and 
                result['lingua_detection'].get('confidence', 0) >= confidence_threshold):
                valid_detections.append(result)
        
        if not valid_detections:
            log_progress("No detections met confidence threshold", log_file)
            return None
        
        log_progress(f"Found {len(valid_detections)} valid detections out of {len(detection_results)} samples", log_file)
        
        # Count votes for each language
        language_votes = {}
        total_confidence = {}
        sample_details = {}
        
        for result in valid_detections:
            lingua_result = result['lingua_detection']
            lang = lingua_result['language']
            confidence = lingua_result['confidence']
            
            if lang not in language_votes:
                language_votes[lang] = 0
                total_confidence[lang] = 0
                sample_details[lang] = []
            
            language_votes[lang] += 1
            total_confidence[lang] += confidence
            sample_details[lang].append({
                'sample': result['sample_index'],
                'confidence': confidence,
                'text_length': lingua_result['text_length']
            })
        
        # Calculate vote percentages and average confidences
        total_valid_samples = len(valid_detections)
        results_summary = {}
        
        for lang, votes in language_votes.items():
            vote_percentage = votes / total_valid_samples
            avg_confidence = total_confidence[lang] / votes
            
            results_summary[lang] = {
                'votes': votes,
                'vote_percentage': vote_percentage,
                'average_confidence': avg_confidence,
                'sample_details': sample_details[lang]
            }
            
            log_progress(f"Language {lang}: {votes}/{total_valid_samples} votes ({vote_percentage:.1%}), avg confidence: {avg_confidence:.3f}", log_file)
        
        # Find language with highest vote percentage that meets majority threshold
        winning_language = None
        best_percentage = 0
        
        for lang, stats in results_summary.items():
            if stats['vote_percentage'] >= majority_threshold and stats['vote_percentage'] > best_percentage:
                winning_language = lang
                best_percentage = stats['vote_percentage']
        
        if winning_language:
            winner_stats = results_summary[winning_language]
            log_progress(f"Final language determination: {winning_language} ({winner_stats['vote_percentage']:.1%} consensus, {winner_stats['average_confidence']:.3f} avg confidence)", log_file)
            
            return {
                'language': winning_language,
                'confidence': winner_stats['average_confidence'],
                'consensus_percentage': winner_stats['vote_percentage'],
                'supporting_samples': winner_stats['votes'],
                'total_samples': total_valid_samples,
                'method': 'enhanced_multi_sample',
                'all_results': results_summary
            }
        else:
            log_progress(f"No language achieved majority threshold of {majority_threshold:.1%}", log_file)
            return None
            
    except Exception as e:
        log_progress(f"Error in final language determination: {e}", log_file)
        return None

def main():
    if len(sys.argv) != 8:
        print("Usage: python script.py <video_path> <sample_count> <sample_duration> <confidence_threshold> <majority_threshold> <audio_duration> <audio_track_count>", file=sys.stderr)
        sys.exit(1)
    
    video_path = sys.argv[1]
    sample_count = int(sys.argv[2])
    sample_duration = int(sys.argv[3])
    confidence_threshold = float(sys.argv[4])
    majority_threshold = float(sys.argv[5])
    audio_duration = float(sys.argv[6])
    audio_track_count = int(sys.argv[7])
    
    # Setup temp directory and logging
    temp_dir = "/app/arrbit/data/temp"
    os.makedirs(temp_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_filename = Path(video_path).stem
    log_file = os.path.join(temp_dir, f"{base_filename}_{timestamp}_language_detection.log")
    
    try:
        log_progress("=== Enhanced Multi-Track Language Detection Started ===", log_file)
        log_progress(f"Video: {video_path}", log_file)
        log_progress(f"Parameters: {sample_count} samples, {sample_duration}s each, confidence≥{confidence_threshold}, majority≥{majority_threshold:.1%}", log_file)
        log_progress(f"Audio duration: {audio_duration:.2f} seconds", log_file)
        log_progress(f"Audio tracks to process: {audio_track_count}", log_file)
        
        # 1. Generate random timestamps
        log_progress("Step 1: Generating random timestamps for audio sampling", log_file)
        timestamps = generate_random_timestamps(audio_duration, sample_count, sample_duration)
        log_progress(f"Generated timestamps: {[f'{t:.2f}s' for t in timestamps]}", log_file)
        
        # Results for all tracks
        all_tracks_results = {}
        
        # 2. Process each audio track
        for track_index in range(audio_track_count):
            log_progress(f"Step 2.{track_index + 1}: Processing audio track {track_index}", log_file)
            
            # Extract audio samples for this track
            audio_samples = extract_audio_samples(video_path, timestamps, sample_duration, temp_dir, track_index, log_file)
            
            if len(audio_samples) < 2:
                log_progress(f"Insufficient audio samples extracted for track {track_index}: {len(audio_samples)}/{sample_count}", log_file)
                all_tracks_results[f"track_{track_index}"] = {
                    'success': False,
                    'error': f'Insufficient samples: {len(audio_samples)}/{sample_count}',
                    'track_index': track_index
                }
                continue
            
            # 3. Transcribe each sample with WhisperX for this track
            log_progress(f"Step 3.{track_index + 1}: Transcribing samples for track {track_index} with WhisperX tiny model + VAD", log_file)
            detection_results = []
            
            for sample_info in audio_samples:
                try:
                    # Transcribe with WhisperX
                    transcription_result = transcribe_sample_whisperx(sample_info, log_file)
                    
                    if transcription_result['success'] and len(transcription_result['text'].strip()) >= 10:
                        # Detect language with Lingua
                        lingua_result = detect_language_lingua(
                            transcription_result['text'], 
                            transcription_result['sample_index'], 
                            log_file
                        )
                        
                        detection_results.append({
                            'sample_index': transcription_result['sample_index'],
                            'transcription': transcription_result,
                            'lingua_detection': lingua_result,
                            'start_time': sample_info['start_time'],
                            'track_index': track_index
                        })
                    else:
                        log_progress(f"Track {track_index} sample {transcription_result['sample_index']}: insufficient transcription quality", log_file)
                        
                except Exception as e:
                    log_progress(f"Error processing track {track_index} sample {sample_info.get('index', '?')}: {e}", log_file)
                
                finally:
                    # Clean up sample file
                    if 'path' in sample_info and os.path.exists(sample_info['path']):
                        try:
                            os.unlink(sample_info['path'])
                        except:
                            pass
            
            # 4. Determine final language for this track
            log_progress(f"Step 4.{track_index + 1}: Determining final language for track {track_index} through consensus", log_file)
            final_result = determine_final_language(detection_results, majority_threshold, confidence_threshold, log_file)
            
            if final_result:
                log_progress(f"=== Track {track_index} detection completed successfully: {final_result['language']} ===", log_file)
                all_tracks_results[f"track_{track_index}"] = {
                    'success': True,
                    'track_index': track_index,
                    'language': final_result['language'],
                    'confidence': final_result['confidence'],
                    'consensus_percentage': final_result['consensus_percentage'],
                    'supporting_samples': final_result['supporting_samples'],
                    'total_samples': final_result['total_samples'],
                    'method': final_result['method'],
                    'sample_results': detection_results
                }
            else:
                log_progress(f"=== Track {track_index} detection failed: no consensus reached ===", log_file)
                all_tracks_results[f"track_{track_index}"] = {
                    'success': False,
                    'track_index': track_index,
                    'error': 'No language consensus reached',
                    'sample_results': detection_results
                }
        
        # 5. Save comprehensive results
        results_file = os.path.join(temp_dir, f"{base_filename}_{timestamp}_multi_track_detection_results.json")
        with open(results_file, 'w') as f:
            json.dump({
                'all_tracks_results': all_tracks_results,
                'parameters': {
                    'sample_count': sample_count,
                    'sample_duration': sample_duration,
                    'confidence_threshold': confidence_threshold,
                    'majority_threshold': majority_threshold,
                    'audio_duration': audio_duration,
                    'audio_track_count': audio_track_count
                },
                'processing_info': {
                    'timestamps': timestamps,
                    'tracks_processed': audio_track_count,
                    'successful_tracks': len([r for r in all_tracks_results.values() if r.get('success', False)])
                }
            }, f, indent=2)
        
        # Check if any tracks were successfully processed
        successful_tracks = [r for r in all_tracks_results.values() if r.get('success', False)]
        
        if successful_tracks:
            log_progress(f"=== Multi-track detection completed: {len(successful_tracks)}/{audio_track_count} tracks processed successfully ===", log_file)
            
            print(json.dumps({
                'success': True,
                'tracks_processed': audio_track_count,
                'successful_tracks': len(successful_tracks),
                'results': all_tracks_results,
                'log_file': log_file,
                'results_file': results_file,
                'temp_directory': temp_dir
            }))
        else:
            log_progress("=== Multi-track detection failed: no tracks processed successfully ===", log_file)
            print(json.dumps({
                'success': False,
                'error': 'No tracks processed successfully',
                'tracks_processed': audio_track_count,
                'results': all_tracks_results,
                'log_file': log_file,
                'temp_directory': temp_dir
            }))
    
    except Exception as e:
        error_msg = f"Enhanced multi-track language detection failed: {str(e)}"
        log_progress(f"ERROR: {error_msg}", log_file)
        
        print(json.dumps({
            'success': False,
            'error': error_msg,
            'traceback': traceback.format_exc(),
            'log_file': log_file if 'log_file' in locals() else None,
            'temp_directory': temp_dir
        }))
        sys.exit(1)

if __name__ == "__main__":
    main()
`;

  // Write enhanced temporary Python script
  const tempDir = librarySettings.cache || "/tmp";
  const tempScriptPath = path.join(
    tempDir,
    `enhanced_ai_language_detection_${Date.now()}.py`
  );

  try {
    fs.writeFileSync(tempScriptPath, tempScript);
    response.infoLog += `☑ Created enhanced detection script: ${tempScriptPath}\n`;

    // Execute enhanced AI language detection
    const aiCommand = [
      pythonBin,
      tempScriptPath,
      `"${file.file}"`,
      sampleCount.toString(),
      sampleDuration.toString(),
      confidenceThreshold.toString(),
      majorityVoteThreshold.toString(),
      audioDuration.toString(),
      audioStreams.length.toString(), // Number of audio tracks to process
    ].join(" ");

    response.infoLog += `☑ Executing enhanced multi-track language detection...\n`;
    response.infoLog += `Command: ${aiCommand}\n`;
    response.infoLog += `Expected processing time: ~${Math.ceil(
      (sampleCount * audioStreams.length * 15) / 60
    )} minute(s) for ${sampleCount} samples across ${
      audioStreams.length
    } audio track(s)\n`;

    const aiResult = execSync(aiCommand, {
      encoding: "utf8",
      timeout: 900000, // 15 minute timeout for multiple samples
      maxBuffer: 5 * 1024 * 1024, // 5MB buffer
    });

    const detectionResult = JSON.parse(aiResult.trim());

    if (detectionResult.success) {
      const tracksProcessed = detectionResult.tracks_processed;
      const successfulTracks = detectionResult.successful_tracks;
      const results = detectionResult.results;

      response.infoLog += `\n=== Enhanced Multi-Track Language Detection Results ===\n`;
      response.infoLog += `☑ Audio tracks processed: ${tracksProcessed}\n`;
      response.infoLog += `☑ Successful detections: ${successfulTracks}/${tracksProcessed}\n`;

      // Log results for each track
      Object.keys(results).forEach((trackKey) => {
        const trackResult = results[trackKey];
        const trackIndex = trackResult.track_index;

        if (trackResult.success) {
          response.infoLog += `\n--- Track ${trackIndex} Results ---\n`;
          response.infoLog += `☑ Language detected: ${trackResult.language}\n`;
          response.infoLog += `☑ Average confidence: ${trackResult.confidence.toFixed(
            3
          )}\n`;
          response.infoLog += `☑ Consensus: ${(
            trackResult.consensus_percentage * 100
          ).toFixed(1)}% (${trackResult.supporting_samples}/${
            trackResult.total_samples
          } samples)\n`;
          response.infoLog += `☑ Detection method: ${trackResult.method}\n`;
        } else {
          response.infoLog += `\n--- Track ${trackIndex} Results ---\n`;
          response.infoLog += `☒ Detection failed: ${trackResult.error}\n`;
        }
      });

      if (detectionResult.log_file) {
        response.infoLog += `\n☑ Detailed log: ${detectionResult.log_file}\n`;
      }
      if (detectionResult.results_file) {
        response.infoLog += `☑ Results file: ${detectionResult.results_file}\n`;
      }
      response.infoLog += `☑ Temp directory: ${detectionResult.temp_directory}\n`;

      response.infoLog += `\n✔ Enhanced multi-track language detection completed successfully!\n`;
      response.infoLog += `📁 Check ${detectionResult.temp_directory} for detailed logs and analysis\n`;

      // No tagging applied - detection only
      response.processFile = false;
    } else {
      response.infoLog += `☒ Enhanced multi-track language detection failed: ${detectionResult.error}\n`;

      if (detectionResult.log_file) {
        response.infoLog += `🔍 Check detailed log: ${detectionResult.log_file}\n`;
      }
      if (detectionResult.temp_directory) {
        response.infoLog += `📁 Debug files available in: ${detectionResult.temp_directory}\n`;
      }

      if (detectionResult.traceback) {
        response.infoLog += `Debug info: ${detectionResult.traceback}\n`;
      }
      response.processFile = false;
    }
  } catch (error) {
    response.infoLog += `☒ Enhanced multi-track language detection error: ${error.message}\n`;
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
