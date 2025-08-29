/* eslint no-plusplus: ["error", { "allowForLoopAfterthoughts": true }] */
const details = () => ({
  id: "Tdarr_Plugin_Arrbit_Anime_Subtitle_Cleaner",
  Stage: "Pre-processing",
  Name: "Arrbit - Anime Subtitle Cleaner",
  Type: "Subtitle",
  Operation: "Transcode",
  Description:
    "This plugin keeps only specified language subtitle tracks, prioritizes forced subtitles for English, deletes flagged subtitles (hearing impaired, visual impaired, text descriptions, commentary) based on configuration, allows filtering based on title keywords, and deletes subtitles containing undesired title tags. It also removes closed captions (XDS, 608, 708).",
  Version: "3.4",
  Tags: "pre-processing,ffmpeg,subtitle only,configurable",
  Inputs: [
    {
      name: "desired_languages",
      type: "string",
      defaultValue: "eng",
      inputUI: {
        type: "text",
      },
      tooltip: `Specify the language codes for the subtitle tracks you'd like to keep.
                \\nMust follow ISO-639-2 3-letter format. https://en.wikipedia.org/wiki/List_of_ISO_639-2_codes
                \\nSeparate multiple languages with commas. Spaces are acceptable.
                \\nExample:\\n
                eng, spa`,
    },
    {
      name: "undesired_titles",
      type: "string",
      defaultValue: "sdh, commentary, honorifics, honorific",
      inputUI: {
        type: "text",
      },
      tooltip: `Specify the undesired title words.
                \\nSubtitles with titles containing these words will be removed.
                \\nSeparate multiple words with commas. Spaces are acceptable.
                \\nExample:\\n
                SDH, Commentary, Closed caption
                \\nNote:\\n
                This feature will be ignored if the "force_flagged_removal" option is set to true.`,
    },
    {
      name: "desired_titles",
      type: "string",
      defaultValue:
        "full subtitles, full subs, full, dialogue, english, signs & songs, signs and songs, signs & song, sign and song, sign/song, signs, songs, sign, song ",
      inputUI: {
        type: "text",
      },
      tooltip: `Specify desired title keywords to keep specific subtitles.
                \\nSubtitles containing these keywords in their title will be kept.
                \\nSeparate multiple keywords with commas. Spaces are acceptable.
                \\nExample:\\n
                latinoamericano, director's cut`,
    },
    {
      name: "force_flagged_removal",
      type: "boolean",
      defaultValue: true,
      inputUI: {
        type: "checkbox",
      },
      tooltip: `Enable this option to automatically remove subtitles flagged as:
                \\n- Hearing impaired (SDH)
                \\n- Visual impaired
                \\n- Commentary
                \\n- Text descriptions
                \\nThese subtitles will be deleted regardless of whether these terms are included in the "undesired_titles" field.
                \\nIf disabled, flagged subtitles will only be removed if they match the terms specified in the "undesired_titles" field.`,
    },
  ],
});

const plugin = (file, librarySettings, inputs, otherArguments) => {
  const lib = require("../methods/lib")();
  // Load default values for inputs
  inputs = lib.loadDefaultValues(inputs, details);

  // Normalize inputs for case-insensitive matching
  const desired_languages = (inputs.desired_languages || "")
    .split(",")
    .map((lang) => lang.trim().toLowerCase())
    .filter((lang) => lang !== "");

  const desiredTitles = (inputs.desired_titles || "")
    .split(",")
    .map((word) => word.trim().toLowerCase())
    .filter((word) => word !== "");

  const undesiredTitles = (inputs.undesired_titles || "")
    .split(",")
    .map((word) => word.trim().toLowerCase())
    .filter((word) => word !== "");

  const forceFlaggedRemoval = inputs.force_flagged_removal === true;

  const response = {
    processFile: false,
    preset: ',-map 0 -codec copy -bsf:v "filter_units=remove_types=6"', // CC removal preset
    container: `.${file.container}`,
    handBrakeMode: false,
    FFmpegMode: true,
    reQueueAfter: false,
    infoLog: "",
  };

  // Check if file is a video. If it isn't, then exit plugin.
  if (file.fileMedium !== "video") {
    response.infoLog += "☒ File is not a video. Skipping plugin.\n";
    return response;
  }

  // Check if desired_languages has been configured. If it hasn't, then exit plugin.
  if (desired_languages.length === 0) {
    response.infoLog +=
      '☒ No languages specified to keep. Please configure the "desired_languages" input in the plugin settings. Skipping plugin.\n';
    return response;
  }

  // Set up required variables.
  let ffmpegCommandInsert = "";
  let subtitleIdx = 0;
  let convert = false;

  // Function to check if the subtitle is forced
  const isSubtitleForced = (stream) => {
    if (stream.disposition && stream.disposition.forced === 1) {
      return true;
    }
    let streamTitle = "";
    if (stream.tags && stream.tags.title) {
      streamTitle = stream.tags.title.toLowerCase();
      if (streamTitle.includes("forced")) {
        return true;
      }
    }
    return false;
  };

  // Function to check if the subtitle matches desired title keywords
  const matchesDesiredTitles = (stream) => {
    if (desiredTitles.length === 0) {
      return true; // If no desired titles specified, consider all titles as desired
    }
    let streamTitle = "";
    if (stream.tags && stream.tags.title) {
      streamTitle = stream.tags.title.toLowerCase();
      for (const keyword of desiredTitles) {
        if (streamTitle.includes(keyword)) {
          return true;
        }
      }
    }
    return false;
  };

  // Function to check if the subtitle matches undesired title keywords or flags
  const matchesUndesiredTitlesOrFlags = (stream) => {
    let streamTitle = "";
    if (stream.tags && stream.tags.title) {
      streamTitle = stream.tags.title.toLowerCase();
      for (const undesired of undesiredTitles) {
        if (streamTitle.includes(undesired)) {
          return true;
        }
      }
    }
    if (forceFlaggedRemoval) {
      return hasImpairmentFlag(stream);
    }
    return false;
  };

  // Function to check if the subtitle is flagged as Hearing impaired, Visual impaired, Text descriptions, or commentary
  const hasImpairmentFlag = (stream) => {
    const impairments = [
      "hearing impaired",
      "visual impaired",
      "text descriptions",
      "commentary",
    ];
    if (stream.tags && stream.tags.title) {
      const streamTitle = stream.tags.title.toLowerCase();
      for (const impairment of impairments) {
        if (streamTitle.includes(impairment)) {
          return true;
        }
      }
    }
    return false;
  };

  // Iterate through each stream in the file.
  for (let i = 0; i < file.ffProbeData.streams.length; i++) {
    const stream = file.ffProbeData.streams[i];
    const codecType = stream.codec_type.toLowerCase();

    if (codecType === "subtitle") {
      let streamLang = "und"; // Default to 'und' if language tag is missing

      // Attempt to get the language tag
      if (stream.tags && stream.tags.language) {
        streamLang = stream.tags.language.toLowerCase();
      }

      // Check if we should force delete flagged subtitles or those with unknown language
      if (
        (forceFlaggedRemoval && hasImpairmentFlag(stream)) ||
        streamLang === "und" ||
        streamLang === "undefined" ||
        streamLang === "unknown"
      ) {
        ffmpegCommandInsert += `-map -0:s:${subtitleIdx} `;
        response.infoLog += `☒ Removing subtitle stream 0:s:${subtitleIdx} with impairment flag or unknown language.\n`;
        convert = true;
      } else {
        // Determine if the subtitle language is in the list to keep
        const isDesiredLanguage = desired_languages.includes(streamLang);

        const subtitleIsForced = isSubtitleForced(stream);

        let keepSubtitle = false;
        let setForcedFlag = false;

        if (isDesiredLanguage) {
          if (subtitleIsForced) {
            keepSubtitle = true;
            setForcedFlag = true;
          } else if (matchesDesiredTitles(stream)) {
            keepSubtitle = true;
          } else if (!matchesUndesiredTitlesOrFlags(stream)) {
            keepSubtitle = true; // Keep subtitles not matching undesired titles or flags
          }
        }

        if (!keepSubtitle) {
          ffmpegCommandInsert += `-map -0:s:${subtitleIdx} `;
          response.infoLog += `☒ Removing subtitle stream 0:s:${subtitleIdx} with language "${streamLang}".\n`;
          convert = true;
        } else {
          // Check for undesired titles in the subtitle
          if (matchesUndesiredTitlesOrFlags(stream)) {
            ffmpegCommandInsert += `-map -0:s:${subtitleIdx} `;
            response.infoLog += `☒ Removing subtitle stream 0:s:${subtitleIdx} due to undesired title or flag.\n`;
            convert = true;
          } else {
            if (setForcedFlag) {
              if (!(stream.disposition && stream.disposition.forced === 1)) {
                ffmpegCommandInsert += `-disposition:s:${subtitleIdx} +forced `;
                response.infoLog += `☒ Setting forced flag on subtitle stream 0:s:${subtitleIdx}.\n`;
                convert = true;
              }
            }
          }
        }
      }

      subtitleIdx += 1; // Increment subtitle index
    }
  }

  // Check and process for closed captions removal at the file level
  if (file.hasClosedCaptions) {
    response.processFile = true;
    response.infoLog += "☒ This file has closed captions.\n";
  } else {
    file.ffProbeData.streams.forEach((stream) => {
      if (stream.closed_captions) {
        response.processFile = true;
        response.infoLog += "☒ This file has burnt closed captions.\n";
      }
    });
  }

  // Build the final FFmpeg preset if any changes are made
  if (convert === true || response.processFile === true) {
    response.processFile = true;
    // Prepend a comma and include -map 0 to map all streams
    response.preset = `, -map 0 ${ffmpegCommandInsert} -c copy -max_muxing_queue_size 9999`;
    response.container = `.${file.container}`;
    response.reQueueAfter = true;
    response.infoLog += "☒ Audio and subtitle flags have been set.\n";
  } else {
    response.infoLog +=
      "☑ No subtitles or closed captions needed to be removed or tagged.\n";
  }

  return response;
};

module.exports.details = details;
module.exports.plugin = plugin;
