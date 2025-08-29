const details = () => ({
  id: "Tdarr_Plugin_Arrbit_Audio_KeepBestCodec",
  Stage: "Pre-processing",
  Name: "Arrbit - Keep Best Audio Codec",
  Type: "Audio",
  Operation: "Transcode",
  Description: `
    This plugin detects multiple audio streams in the same language and keeps only the best one according to a priority list of codecs.
    For example, if a file has English TrueHD, English DTS, English FLAC, and English AC3, it will keep the TrueHD stream over the others since it's higher in the priority list.
    The priority list of codecs is as follows:
    1. TrueHD
    2. DTS
    3. FLAC
    4. AC3
    If the highest priority codec is not found, the next one is chosen, and so on.
    This process is repeated for each language present in the file.
    `,
  Version: "1.0",
  Tags: "audio,ffmpeg",
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

  // Check if file is a video
  if (file.fileMedium !== "video") {
    response.infoLog += "☒ File is not a video. Skipping.\n";
    return response;
  }

  const codecPriority = ["truehd", "dts", "flac", "ac3"];

  const streams = file.ffProbeData.streams;
  const audioStreams = [];
  const audioStreamIndices = [];

  // Build audio stream list and mapping
  streams.forEach((stream, idx) => {
    if (stream.codec_type.toLowerCase() === "audio") {
      audioStreams.push(stream);
      audioStreamIndices.push(idx);
    }
  });

  if (audioStreams.length === 0) {
    response.infoLog += "☒ No audio streams found.\n";
    return response;
  }

  // Map stream index to audio index (for FFmpeg mapping)
  const streamIndexToAudioIdx = {};
  audioStreams.forEach((stream, idx) => {
    streamIndexToAudioIdx[stream.index] = idx;
  });

  // Group audio streams by language
  const audioStreamsByLanguage = {};

  audioStreams.forEach((stream) => {
    let lang = "und";
    if (stream.tags && (stream.tags.language || stream.tags.LANGUAGE)) {
      lang = (stream.tags.language || stream.tags.LANGUAGE).toLowerCase();
    }
    if (!audioStreamsByLanguage[lang]) {
      audioStreamsByLanguage[lang] = [];
    }
    audioStreamsByLanguage[lang].push(stream);
  });

  const streamsToKeep = [];
  const streamsToRemove = [];

  // Select best audio stream per language
  Object.keys(audioStreamsByLanguage).forEach((lang) => {
    const streams = audioStreamsByLanguage[lang];
    let bestStream = null;
    for (const codec of codecPriority) {
      const stream = streams.find((s) => s.codec_name.toLowerCase() === codec);
      if (stream) {
        bestStream = stream;
        break;
      }
    }
    if (!bestStream) {
      // Keep the first stream if no preferred codec is found
      bestStream = streams[0];
    }
    streamsToKeep.push(bestStream);
    response.infoLog += `☑ Keeping ${lang} audio stream with codec ${bestStream.codec_name} (stream index ${bestStream.index}).\n`;

    // Mark other streams in this language for removal
    streams.forEach((stream) => {
      if (stream.index !== bestStream.index) {
        streamsToRemove.push(stream);
        response.infoLog += `☒ Removing ${lang} audio stream with codec ${stream.codec_name} (stream index ${stream.index}).\n`;
      }
    });
  });

  if (streamsToRemove.length === 0) {
    response.infoLog += "☑ No unnecessary audio streams to remove.\n";
    return response;
  }

  // Build FFmpeg command
  let ffmpegCommand = ", -map 0";

  // Remove unwanted audio streams
  streamsToRemove.forEach((stream) => {
    const audioIdx = streamIndexToAudioIdx[stream.index];
    ffmpegCommand += ` -map -0:a:${audioIdx}`;
  });

  // Copy all streams
  ffmpegCommand += " -c copy";

  response.processFile = true;
  response.preset = ffmpegCommand;
  response.container = `.${file.container}`;
  response.reQueueAfter = true;

  return response;
};

module.exports.details = details;
module.exports.plugin = plugin;
