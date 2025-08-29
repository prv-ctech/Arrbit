/* eslint-disable */
const details = () => {
  return {
    id: "Tdarr_Plugin_Arrbit_Subtitle_KeepTwo",
    Stage: "Pre-processing",
    Name: "Arrbit - Subtitle: Keep First Two",
    Type: "Video",
    Operation: "Transcode",
    Description: `[Contains built-in filter] This plugin retains only the first two subtitle streams in a video file. The output container remains the same as the original.\n\n`,
    Version: "1.01",
    Tags: "pre-processing,ffmpeg,subtitle only,limit",
    Inputs: [],
  };
};

// eslint-disable-next-line @typescript-eslint/no-unused-vars
const plugin = (file, librarySettings, inputs, otherArguments) => {
  const lib = require("../methods/lib")();
  // eslint-disable-next-line @typescript-eslint/no-unused-vars,no-param-reassign
  inputs = lib.loadDefaultValues(inputs, details);
  const log = otherArguments.logger || console;

  // Initialize the response object
  var response = {
    processFile: false,
    preset: "",
    container: "." + file.container,
    handBrakeMode: false,
    FFmpegMode: false,
    reQueueAfter: false,
    infoLog: "",
  };

  // Check if the file is a video
  if (file.fileMedium !== "video") {
    log.log("File is not video");

    response.infoLog += "☒ File is not a video.\n";
    response.processFile = false;

    return response;
  } else {
    response.FFmpegMode = true;
    response.container = "." + file.container;

    // Collect all subtitle stream indices
    let subtitleStreams = [];
    file.ffProbeData.streams.forEach((stream, index) => {
      if (stream.codec_type.toLowerCase() === "subtitle") {
        subtitleStreams.push(index);
      }
    });

    if (subtitleStreams.length > 2) {
      // There are more than two subtitle streams
      response.infoLog += `☒ File has ${subtitleStreams.length} subtitle streams.\n`;

      // Streams to keep: first two
      let streamsToKeep = subtitleStreams.slice(0, 2);
      // Streams to remove: beyond the first two
      let streamsToRemove = subtitleStreams.slice(2);

      // Construct the FFmpeg mapping options
      let ffmpegMapping = " -map 0:v -map 0:a"; // Map all video and audio streams

      // Map the first two subtitle streams
      streamsToKeep.forEach((streamIndex) => {
        ffmpegMapping += ` -map 0:${streamIndex}`;
        response.infoLog += `☒ Keeping subtitle stream 0:${streamIndex}.\n`;
      });

      // Log the removal of extra subtitle streams
      streamsToRemove.forEach((streamIndex) => {
        response.infoLog += `☒ Removing subtitle stream 0:${streamIndex}.\n`;
      });

      // Append codec copy and muxing queue size
      ffmpegMapping += " -c copy -max_muxing_queue_size 9999";

      // Prefix with a comma to correctly append after the input file
      response.preset = `,${ffmpegMapping}`;
      response.reQueueAfter = true;
      response.processFile = true;
      response.infoLog += "☒ Excess subtitle streams have been removed.\n";
    } else if (subtitleStreams.length === 2) {
      // Exactly two subtitle streams
      response.infoLog +=
        "☑ File has exactly two subtitle streams; no action needed.\n";
      // Optionally, ensure that all streams are mapped
      response.preset = ",-map 0 -c copy -max_muxing_queue_size 9999";
      response.processFile = false; // No processing needed
    } else if (subtitleStreams.length === 1) {
      // Only one subtitle stream
      response.infoLog += "☑ File has one subtitle stream; no action needed.\n";
      // Optionally, ensure that all streams are mapped
      response.preset = ",-map 0 -c copy -max_muxing_queue_size 9999";
      response.processFile = false; // No processing needed
    } else {
      // No subtitle streams
      response.infoLog += "☑ File has no subtitle streams; no action needed.\n";
      response.processFile = false;
    }

    return response;
  }
};

module.exports.details = details;
module.exports.plugin = plugin;
