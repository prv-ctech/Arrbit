/* eslint no-plusplus: ["error", { "allowForLoopAfterthoughts": true }] */
const details = () => ({
  id: "Tdarr_Plugin_Arrbit_Anime_Subtitle_Flagger",
  Stage: "Pre-processing",
  Name: "Arrbit - Anime Subtitle Flagger",
  Type: "Subtitle",
  Operation: "Transcode",
  Description:
    "This plugin removes all subtitle flags, sets the first subtitle stream as default, and the second as forced, without changing the stream order.",
  Version: "1.0",
  Tags: "pre-processing,ffmpeg,subtitle only,flagging",
  Inputs: [],
});

// eslint-disable-next-line @typescript-eslint/no-unused-vars
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

  // Check if file is a video. If it isn't, then exit plugin.
  if (file.fileMedium !== "video") {
    response.infoLog += "☒ File is not video\n";
    return response;
  }

  // Map all streams as-is
  let ffmpegCommandInsert = "-map 0 ";

  // Collect indices of subtitle streams
  const subtitleStreamIndices = [];
  file.ffProbeData.streams.forEach((stream, index) => {
    if (stream.codec_type.toLowerCase() === "subtitle") {
      subtitleStreamIndices.push(index);
    }
  });

  if (subtitleStreamIndices.length === 0) {
    response.infoLog += "☒ No subtitle streams found.\n";
    return response;
  }

  // Remove all dispositions from all subtitle streams
  subtitleStreamIndices.forEach((streamIndex) => {
    ffmpegCommandInsert += `-disposition:${streamIndex} 0 `;
  });

  // Set first subtitle stream as default
  const firstSubtitleIndex = subtitleStreamIndices[0];
  ffmpegCommandInsert += `-disposition:${firstSubtitleIndex} default `;
  response.infoLog += `☑ Setting subtitle stream ${firstSubtitleIndex} as default.\n`;

  // Set second subtitle stream as forced, if it exists
  if (subtitleStreamIndices.length > 1) {
    const secondSubtitleIndex = subtitleStreamIndices[1];
    ffmpegCommandInsert += `-disposition:${secondSubtitleIndex} forced `;
    response.infoLog += `☑ Setting subtitle stream ${secondSubtitleIndex} as forced.\n`;
  }

  // Set the processFile flag to true to proceed with processing
  response.processFile = true;
  response.preset = `, ${ffmpegCommandInsert}-c copy -max_muxing_queue_size 9999`;
  response.infoLog += "✔ Subtitle tracks have been flagged appropriately.\n";

  return response;
};

module.exports.details = details;
module.exports.plugin = plugin;
