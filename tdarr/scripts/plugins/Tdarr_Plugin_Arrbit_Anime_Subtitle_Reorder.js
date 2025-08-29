/* eslint no-plusplus: ["error", { "allowForLoopAfterthoughts": true }] */
const details = () => ({
  id: "Tdarr_Plugin_Arrbit_Anime_Subtitle_Reorder",
  Stage: "Pre-processing",
  Name: "Arrbit - Anime Subtitle Reorder",
  Type: "Subtitle",
  Operation: "Transcode",
  Description:
    "Reorders subtitles to ensure main subtitles are first, secondary subtitles (e.g., signs and songs, forced) are second, and others follow, without affecting other streams or metadata.",
  Version: "1.9", // Incremented version for tracking
  Tags: "pre-processing,ffmpeg,subtitle only,reorder,anime",
  Inputs: [], // No external inputs
});

const plugin = (file, librarySettings, inputs, otherArguments) => {
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

  // Define keywords internally
  const mainLanguage = "eng"; // Set your desired main subtitle language code here

  const secondaryKeywords = [
    "signs & songs",
    "signs and songs",
    "signs & song",
    "sign and song",
    "sign/song",
    "signs",
    "songs",
    "sign",
    "song",
    "forced",
  ];

  const subtitleStreams = streams.filter(
    (stream) => stream.codec_type.toLowerCase() === "subtitle"
  );

  if (subtitleStreams.length === 0) {
    response.infoLog += "☒ No subtitle streams found.\n";
    response.processFile = false;
    return response;
  }

  response.infoLog += `☑ Found ${subtitleStreams.length} subtitle stream(s) for reordering.\n`;

  let mainSubtitleIndex = -1;
  let secondarySubtitleIndex = -1;

  // Identify secondary subtitles first
  subtitleStreams.forEach((stream, index) => {
    const title = stream.tags?.title?.toLowerCase() || "";
    if (secondaryKeywords.some((keyword) => title.includes(keyword))) {
      if (secondarySubtitleIndex === -1) {
        secondarySubtitleIndex = index;
        response.infoLog += `☑ Identified secondary subtitle stream at index ${index}.\n`;
      }
    }
  });

  // Identify main subtitles based on language and not being secondary
  subtitleStreams.forEach((stream, index) => {
    if (mainSubtitleIndex !== -1) return; // Already found
    const language = stream.tags?.language?.toLowerCase() || "";
    const title = stream.tags?.title?.toLowerCase() || "";
    if (
      language === mainLanguage &&
      index !== secondarySubtitleIndex &&
      !secondaryKeywords.some((keyword) => title.includes(keyword))
    ) {
      mainSubtitleIndex = index;
      response.infoLog += `☑ Identified main subtitle stream at index ${index}.\n`;
    }
  });

  // If main subtitle not found based on language, select the first subtitle that is not secondary
  if (mainSubtitleIndex === -1) {
    subtitleStreams.forEach((stream, index) => {
      if (index !== secondarySubtitleIndex && mainSubtitleIndex === -1) {
        mainSubtitleIndex = index;
        response.infoLog += `☑ Defaulting main subtitle stream to index ${index}.\n`;
      }
    });
  }

  // Check if reordering is needed
  let needsReordering = false;

  if (
    mainSubtitleIndex !== -1 &&
    mainSubtitleIndex !== 0 // Main subtitle is not the first subtitle stream
  ) {
    needsReordering = true;
  }

  if (
    secondarySubtitleIndex !== -1 &&
    secondarySubtitleIndex !== 1 // Secondary subtitle is not the second subtitle stream
  ) {
    needsReordering = true;
  }

  if (!needsReordering) {
    response.infoLog +=
      "☑ Subtitles are already in the correct order. Skipping processing.\n";
    return response;
  }

  // Build FFmpeg command to reorder subtitles
  let ffmpegCommandInsert = "";

  // Map all streams except subtitle streams
  ffmpegCommandInsert += "-map 0 -map -0:s ";

  // Map main subtitle first
  if (mainSubtitleIndex !== -1) {
    ffmpegCommandInsert += `-map 0:s:${mainSubtitleIndex} `;
    response.infoLog += `☑ Mapping main subtitle stream 0:s:${mainSubtitleIndex}.\n`;
  }

  // Map secondary subtitle next
  if (secondarySubtitleIndex !== -1) {
    ffmpegCommandInsert += `-map 0:s:${secondarySubtitleIndex} `;
    response.infoLog += `☑ Mapping secondary subtitle stream 0:s:${secondarySubtitleIndex}.\n`;
  }

  // Map any remaining subtitle streams
  subtitleStreams.forEach((stream, index) => {
    if (index !== mainSubtitleIndex && index !== secondarySubtitleIndex) {
      ffmpegCommandInsert += `-map 0:s:${index} `;
      response.infoLog += `☑ Mapping additional subtitle stream 0:s:${index}.\n`;
    }
  });

  // Finalize the command
  response.processFile = true;
  response.preset = `, ${ffmpegCommandInsert}-c copy -max_muxing_queue_size 9999`;
  response.reQueueAfter = true;
  response.infoLog += "✔ Subtitle streams have been reordered appropriately.\n";

  return response;
};

module.exports.details = details;
module.exports.plugin = plugin;
