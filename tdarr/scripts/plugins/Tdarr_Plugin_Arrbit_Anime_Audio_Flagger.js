const details = () => ({
  id: "Tdarr_Plugin_Arrbit_Anime_Audio_Flagger",
  Stage: "Pre-processing",
  Name: "Arrbit - Anime Audio Flagger",
  Type: "Audio",
  Operation: "Transcode",
  Description:
    "Sets the first audio track as default. Unsets all dispositions on other audio tracks. Preserves metadata.",
  Version: "1.1",
  Tags: "pre-processing,ffmpeg,audio,flagging",
  Inputs: [],
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

  let ffmpegCommandInsert = "";
  let convert = false;
  let firstAudioFound = false;

  // Keep track of output stream index
  let outputStreamIndex = 0;

  streams.forEach((stream, inputStreamIndex) => {
    // Map each input stream individually
    ffmpegCommandInsert += `-map 0:${inputStreamIndex} `;

    if (stream.codec_type.toLowerCase() === "audio") {
      if (!firstAudioFound) {
        // First audio track
        ffmpegCommandInsert += `-disposition:${outputStreamIndex} default `;
        response.infoLog += `☑ Setting audio stream ${outputStreamIndex} as default.\n`;
        firstAudioFound = true;
        convert = true;
      } else {
        // Other audio tracks
        ffmpegCommandInsert += `-disposition:${outputStreamIndex} 0 `;
        response.infoLog += `☑ Unsetting dispositions on audio stream ${outputStreamIndex}.\n`;
        convert = true;
      }
    }
    // Increment output stream index
    outputStreamIndex++;
  });

  if (convert) {
    response.processFile = true;
    response.preset = `, ${ffmpegCommandInsert}-c copy -max_muxing_queue_size 9999`;
    response.reQueueAfter = true;
    response.infoLog += "✔ Audio tracks have been flagged appropriately.\n";
  } else {
    response.infoLog += "☑ No changes required for audio tracks.\n";
  }

  return response;
};

module.exports.details = details;
module.exports.plugin = plugin;
