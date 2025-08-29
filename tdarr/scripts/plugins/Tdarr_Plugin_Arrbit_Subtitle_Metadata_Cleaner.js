const details = () => ({
  id: "Tdarr_Plugin_Arrbit_Subtitle_Metadata_Cleaner",
  Stage: "Pre-processing",
  Name: "Arrbit - Subtitle Metadata Cleaner",
  Type: "Video",
  Operation: "Transcode",
  Description: "This plugin removes title metadata from subtitle streams.\n\n",
  Version: "1.0",
  Tags: "pre-processing,ffmpeg",
});

// eslint-disable-next-line @typescript-eslint/no-unused-vars
const plugin = (file, librarySettings, inputs, otherArguments) => {
  const lib = require("../methods/lib")();
  const response = {
    processFile: false,
    preset: "",
    container: `.${file.container}`,
    handBrakeMode: false,
    FFmpegMode: true,
    reQueueAfter: false,
    infoLog: "",
  };

  let ffmpegCommandInsert = "";
  let subtitleIdx = 0;
  let convert = false;

  // Check if the file is a video
  if (file.fileMedium !== "video") {
    console.log("File is not video");
    response.infoLog += "☒ File is not video\n";
    return response;
  }

  // Iterate through all streams in the file
  for (let i = 0; i < file.ffProbeData.streams.length; i += 1) {
    const stream = file.ffProbeData.streams[i];

    // Target subtitle streams
    if (stream.codec_type.toLowerCase() === "subtitle") {
      try {
        const title = stream.tags?.title;

        if (title && title !== '""' && title !== "") {
          // Remove the title metadata from the subtitle stream
          response.infoLog += `☒ Removing title from subtitle stream ${i}.\n`;
          ffmpegCommandInsert += ` -metadata:s:s:${subtitleIdx} title= `;
          convert = true;
        }
        subtitleIdx += 1;
      } catch (err) {
        console.error(`Error processing subtitle title metadata: ${err}`);
      }
    }
  }

  // If any subtitle titles were removed, set up the FFmpeg command
  if (convert === true) {
    response.infoLog += "☒ Subtitle title metadata detected and removed.\n";
    response.preset = `,${ffmpegCommandInsert} -c copy -map 0 -max_muxing_queue_size 9999`;
    response.reQueueAfter = true;
    response.processFile = true;
  } else {
    response.infoLog += "☑ No subtitle title metadata to remove.\n";
  }

  return response;
};

module.exports.details = details;
module.exports.plugin = plugin;
