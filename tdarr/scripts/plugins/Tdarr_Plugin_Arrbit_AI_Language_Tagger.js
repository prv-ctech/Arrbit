const details = () => ({
  id: "Tdarr_Plugin_Arrbit_AI_Language_Tagger",
  Stage: "Pre-processing",
  Name: "Arrbit - AI Language Tagger",
  Type: "Video",
  Operation: "Transcode",
  Description: `Applies language tags to audio tracks based on AI language detection results from Arrbit AI Language Detector.
    Reads detection results from temp files and applies language metadata using mkvpropedit (for MKV) or FFmpeg (for other containers).
    Should be run after Arrbit AI Language Detector plugin. All processing logged to /app/arrbit/data/temp.`,
  Version: "1.0",
  Tags: "pre-processing,language,tagging,metadata,mkvpropedit,ffmpeg",
  Inputs: [
    {
      name: "require_detection_results",
      type: "boolean",
      defaultValue: true,
      inputUI: {
        type: "dropdown",
        options: ["false", "true"],
      },
      tooltip: `Require detection results file to exist before processing.
        When true, plugin will fail if no AI detection results are found.
        When false, will skip processing if no results found.
        \\nExample:\\n
        true`,
    },
    {
      name: "prefer_mkvpropedit",
      type: "boolean",
      defaultValue: true,
      inputUI: {
        type: "dropdown",
        options: ["false", "true"],
      },
      tooltip: `Prefer mkvpropedit for MKV files when possible.
        When true, uses mkvpropedit for direct metadata editing without re-encoding.
        Falls back to FFmpeg if mkvpropedit fails or container is not MKV.
        \\nExample:\\n
        true`,
    },
    {
      name: "overwrite_existing_language",
      type: "boolean",
      defaultValue: false,
      inputUI: {
        type: "dropdown",
        options: ["false", "true"],
      },
      tooltip: `Overwrite existing language metadata on audio streams.
        When true, applies detected language even if language metadata already exists.
        When false, only applies to streams without existing language metadata.
        \\nExample:\\n
        false`,
    },
    {
      name: "minimum_confidence",
      type: "number",
      defaultValue: 0.7,
      inputUI: {
        type: "text",
      },
      tooltip: `Minimum confidence threshold for applying language tags (0.0-1.0).
        Only detection results with confidence above this threshold will be applied.
        Recommended: 0.7 for reliable tagging.
        \\nExample:\\n
        0.7`,
    },
    {
      name: "cleanup_detection_files",
      type: "boolean",
      defaultValue: true,
      inputUI: {
        type: "dropdown",
        options: ["false", "true"],
      },
      tooltip: `Clean up detection result files after successful tagging.
        When true, removes detection result files to save disk space.
        When false, keeps files for debugging or manual review.
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

  response.infoLog += "=== AI Language Tagger Started ===\n";
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
    response.infoLog += "☒ No audio streams found for language tagging.\n";
    response.processFile = false;
    return response;
  }

  response.infoLog += `☑ Found ${audioStreams.length} audio stream(s) for language tagging.\n`;

  // Validate input parameters
  const requireResults =
    inputs.require_detection_results === true ||
    inputs.require_detection_results === "true";
  const preferMkvpropedit =
    inputs.prefer_mkvpropedit === true || inputs.prefer_mkvpropedit === "true";
  const overwriteExisting =
    inputs.overwrite_existing_language === true ||
    inputs.overwrite_existing_language === "true";
  const cleanupFiles =
    inputs.cleanup_detection_files === true ||
    inputs.cleanup_detection_files === "true";

  const minConfidence = parseFloat(inputs.minimum_confidence);
  if (isNaN(minConfidence) || minConfidence < 0 || minConfidence > 1) {
    response.infoLog +=
      "☒ Invalid minimum confidence. Must be between 0.0 and 1.0.\n";
    response.processFile = false;
    return response;
  }

  response.infoLog += `☑ Configuration: require_results=${requireResults}, prefer_mkvpropedit=${preferMkvpropedit}, overwrite=${overwriteExisting}, min_confidence=${minConfidence}, cleanup=${cleanupFiles}\n`;

  // Look for detection results files
  const path = require("path");
  const fs = require("fs");
  const tempDir = "/app/arrbit/data/temp";

  if (!fs.existsSync(tempDir)) {
    response.infoLog += `☒ Temp directory does not exist: ${tempDir}\n`;
    if (requireResults) {
      response.processFile = false;
      return response;
    } else {
      response.infoLog += "⚠ No detection results found, skipping tagging.\n";
      response.processFile = false;
      return response;
    }
  }

  // Find detection results file for this video
  const baseFileName = path.basename(file.file, path.extname(file.file));
  const detectionFiles = fs
    .readdirSync(tempDir)
    .filter(
      (filename) =>
        filename.includes(baseFileName) &&
        filename.includes("language_detection_results.json")
    );

  if (detectionFiles.length === 0) {
    response.infoLog += `☒ No language detection results found for file: ${baseFileName}\n`;
    response.infoLog += `Expected pattern: *${baseFileName}*language_detection_results.json in ${tempDir}\n`;

    if (requireResults) {
      response.processFile = false;
      return response;
    } else {
      response.infoLog += "⚠ No detection results found, skipping tagging.\n";
      response.processFile = false;
      return response;
    }
  }

  // Use the most recent detection results file
  detectionFiles.sort((a, b) => {
    const statA = fs.statSync(path.join(tempDir, a));
    const statB = fs.statSync(path.join(tempDir, b));
    return statB.mtime - statA.mtime;
  });

  const detectionResultsFile = path.join(tempDir, detectionFiles[0]);
  response.infoLog += `☑ Using detection results: ${detectionResultsFile}\n`;

  // Read and parse detection results
  let detectionData;
  try {
    const resultsContent = fs.readFileSync(detectionResultsFile, "utf8");
    detectionData = JSON.parse(resultsContent);
    response.infoLog += `☑ Loaded detection results: ${detectionData.processedStreams} streams processed\n`;
  } catch (parseError) {
    response.infoLog += `☒ Failed to read detection results: ${parseError.message}\n`;
    response.processFile = false;
    return response;
  }

  if (
    !detectionData.detectionResults ||
    detectionData.detectionResults.length === 0
  ) {
    response.infoLog += "☒ No valid detection results found in file.\n";
    response.processFile = false;
    return response;
  }

  // Check if streams already have language metadata (if not overwriting)
  if (!overwriteExisting) {
    const streamsWithLanguage = audioStreams.filter((stream) => {
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

    if (streamsWithLanguage.length > 0) {
      response.infoLog += `⚠ ${streamsWithLanguage.length} audio streams already have language metadata and overwrite is disabled.\n`;
      if (streamsWithLanguage.length === audioStreams.length) {
        response.infoLog +=
          "☑ All audio streams already have language metadata, skipping tagging.\n";
        response.processFile = false;
        return response;
      }
    }
  }

  // Process detection results and build tagging commands
  response.infoLog += `\n=== Processing Detection Results ===\n`;

  const validDetections = [];
  const streamsToTag = new Map(); // streamId -> language

  for (const result of detectionData.detectionResults) {
    const streamId = result.streamId;
    const language = result.language;
    const confidence = result.confidence;
    const consensus = result.consensus;

    response.infoLog += `Stream ${streamId}: ${language} (confidence: ${confidence.toFixed(
      3
    )}, consensus: ${(consensus * 100).toFixed(1)}%)\n`;

    if (confidence >= minConfidence) {
      // Find the corresponding audio stream in current file
      const audioStream = audioStreams.find(
        (stream) => stream.index === streamId
      );
      if (audioStream) {
        // Check if this stream should be tagged
        let shouldTag = overwriteExisting;

        if (!overwriteExisting) {
          // Only tag if no existing language metadata
          shouldTag =
            !audioStream.tags ||
            !(
              audioStream.tags.language ||
              audioStream.tags.Language ||
              audioStream.tags.LANGUAGE ||
              audioStream.tags.lang ||
              audioStream.tags.Lang ||
              audioStream.tags.LANG
            );
        }

        if (shouldTag) {
          streamsToTag.set(streamId, language);
          validDetections.push(result);
          response.infoLog += `  → Will apply language: ${language}\n`;
        } else {
          response.infoLog += `  → Skipping (existing language metadata)\n`;
        }
      } else {
        response.infoLog += `  → Stream ${streamId} not found in current file\n`;
      }
    } else {
      response.infoLog += `  → Confidence too low (${confidence.toFixed(
        3
      )} < ${minConfidence})\n`;
    }
  }

  if (streamsToTag.size === 0) {
    response.infoLog += "☒ No streams meet criteria for language tagging.\n";
    response.processFile = false;
    return response;
  }

  response.infoLog += `☑ Will tag ${streamsToTag.size} audio stream(s)\n`;

  // Apply language tags using mkvpropedit or FFmpeg
  const { execSync } = require("child_process");
  let taggingSuccessful = false;

  if (preferMkvpropedit && file.container.toLowerCase() === "mkv") {
    response.infoLog += `\n=== Applying Language Tags with mkvpropedit ===\n`;

    try {
      // Build mkvpropedit command to set language for detected streams
      let mkvpropeditCmd = `mkvpropedit "${file.file}"`;

      for (const [streamId, language] of streamsToTag) {
        // Find track number for mkvpropedit (1-based)
        const trackNum = streams.findIndex((s) => s.index === streamId) + 1;
        mkvpropeditCmd += ` --edit track:${trackNum} --set language=${language}`;
        response.infoLog += `☑ Setting track ${trackNum} (stream ${streamId}) language to ${language}\n`;
      }

      response.infoLog += `Command: ${mkvpropeditCmd}\n`;

      const mkvResult = execSync(mkvpropeditCmd, {
        encoding: "utf8",
        timeout: 120000, // 2 minute timeout
      });

      response.infoLog += `☑ mkvpropedit completed successfully\n`;
      response.infoLog += `☑ Language tags applied directly to MKV file\n`;
      taggingSuccessful = true;
    } catch (mkvError) {
      response.infoLog += `☒ mkvpropedit failed: ${mkvError.message}\n`;
      response.infoLog += `⚠ Falling back to FFmpeg processing...\n`;
    }
  }

  // Use FFmpeg if mkvpropedit failed or was not preferred
  if (!taggingSuccessful) {
    response.infoLog += `\n=== Building FFmpeg Command for Language Tags ===\n`;

    // Build FFmpeg command to add language metadata to audio streams
    let ffmpegCommandInsert = "";
    let outputStreamIndex = 0;

    streams.forEach((stream, inputStreamIndex) => {
      // Map each input stream individually
      ffmpegCommandInsert += `-map 0:${inputStreamIndex} `;

      if (stream.codec_type && stream.codec_type.toLowerCase() === "audio") {
        const streamId = stream.index;
        if (streamsToTag.has(streamId)) {
          const language = streamsToTag.get(streamId);
          // Add language metadata for this audio stream
          ffmpegCommandInsert += `-metadata:s:${outputStreamIndex} language=${language} `;
          response.infoLog += `☑ Setting stream ${outputStreamIndex} (input stream ${streamId}) language to ${language}\n`;
        }
      }

      outputStreamIndex++;
    });

    if (ffmpegCommandInsert) {
      response.processFile = true;
      response.preset = `, ${ffmpegCommandInsert}-c copy -max_muxing_queue_size 9999`;
      response.reQueueAfter = true;
      response.infoLog += `☑ FFmpeg preset: ${response.preset}\n`;
      taggingSuccessful = true;
    }
  }

  // Clean up detection files if successful and requested
  if (taggingSuccessful && cleanupFiles) {
    response.infoLog += `\n=== Cleaning Up Detection Files ===\n`;

    try {
      // Remove the main detection results file
      fs.unlinkSync(detectionResultsFile);
      response.infoLog += `☑ Removed detection results: ${detectionResultsFile}\n`;

      // Remove related log and detail files for processed streams
      for (const result of validDetections) {
        if (result.logFile && fs.existsSync(result.logFile)) {
          try {
            fs.unlinkSync(result.logFile);
            response.infoLog += `☑ Removed log file: ${result.logFile}\n`;
          } catch (logError) {
            response.infoLog += `⚠ Failed to remove log file: ${logError.message}\n`;
          }
        }

        if (result.resultsFile && fs.existsSync(result.resultsFile)) {
          try {
            fs.unlinkSync(result.resultsFile);
            response.infoLog += `☑ Removed results file: ${result.resultsFile}\n`;
          } catch (resultError) {
            response.infoLog += `⚠ Failed to remove results file: ${resultError.message}\n`;
          }
        }
      }
    } catch (cleanupError) {
      response.infoLog += `⚠ Cleanup error: ${cleanupError.message}\n`;
    }
  }

  if (taggingSuccessful) {
    response.infoLog += `\n✔ Language tagging completed successfully!\n`;
    response.infoLog += `📊 Applied language tags to ${streamsToTag.size} audio stream(s)\n`;

    // Log final summary
    for (const [streamId, language] of streamsToTag) {
      response.infoLog += `  • Stream ${streamId}: ${language}\n`;
    }
  } else {
    response.infoLog += `☒ Language tagging failed for all attempted methods\n`;
    response.processFile = false;
  }

  return response;
};

module.exports.details = details;
module.exports.plugin = plugin;
