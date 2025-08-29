/* eslint no-plusplus: ["error", { "allowForLoopAfterthoughts": true }] */
const details = () => ({
  id: "Tdarr_Plugin_Arrbit_Subtitle_Cleaner",
  Stage: "Pre-processing",
  Name: "Arrbit - Subtitle Cleaner",
  Type: "Subtitle",
  Operation: "Transcode",
  Description:
    "This plugin manages subtitle tracks by keeping only specified languages, prioritizing forced subtitles for English, and removing flagged subtitles based on configuration. It allows filtering based on title keywords, limits the number of subtitles per language, and removes closed captions (XDS, 608, 708).",
  Version: "3.8",
  Tags: "pre-processing,ffmpeg,subtitle only,configurable",
  Inputs: [
    {
      name: "language",
      type: "string",
      defaultValue: "eng, spa",
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
      defaultValue: "sdh, unknown, undefined, und, spain",
      inputUI: {
        type: "text",
      },
      tooltip: `Specify the undesired title words.
                \\nSubtitles with titles containing these words will be removed.
                \\nSeparate multiple words with commas. Spaces are acceptable.
                \\nExample:\\n
                sdh, unknown, undefined, und, spain`,
    },
    {
      name: "desired_titles",
      type: "string",
      defaultValue: "latinoamericano, dialogue, english",
      inputUI: {
        type: "text",
      },
      tooltip: `Specify desired title keywords to keep specific subtitles.
                \\nSubtitles containing these keywords in their title will be kept.
                \\nSeparate multiple keywords with commas. Spaces are acceptable.
                \\nExample:\\n
                latinoamericano, dialogue, english`,
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
                \\nThese subtitles will be deleted regardless of whether these terms are included in the "undesired_titles" field.`,
    },
    {
      name: "max_subtitles_per_language",
      type: "number",
      defaultValue: 1,
      inputUI: {
        type: "number",
      },
      tooltip: `Specify the maximum number of subtitles to keep per language (excluding English).
                \\nEnglish subtitles will always keep up to 2 subtitles (main and forced).
                \\nExample:\\n
                1`,
    },
  ],
});

// eslint-disable-next-line @typescript-eslint/no-unused-vars
const plugin = (file, librarySettings, inputs, otherArguments) => {
  const lib = require("../methods/lib")();
  inputs = lib.loadDefaultValues(inputs, details);

  const languagesToKeep = (inputs.language || "")
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
  const maxSubsPerLanguage =
    parseInt(inputs.max_subtitles_per_language, 10) || 1;

  const response = {
    processFile: false,
    preset: ',-map 0 -codec copy -bsf:v "filter_units=remove_types=6"', // CC removal preset
    container: `.${file.container}`,
    handBrakeMode: false,
    FFmpegMode: true,
    reQueueAfter: false,
    infoLog: "",
  };

  if (file.fileMedium !== "video") {
    response.infoLog += "☒ File is not a video. Skipping plugin.\n";
    return response;
  }

  let ffmpegCommandInsert = "";
  let subtitleIdx = 0; // Index within subtitle streams (s:0, s:1, etc.)
  let convert = false;

  const subtitlesKeptPerLanguage = {};

  // Define impairment keywords and corresponding tag checks
  const impairmentKeywords = [
    "hearing impaired",
    "visual impaired",
    "text descriptions",
    "commentary",
    "sdh",
    "hard of hearing",
    "deaf",
    // Add more as necessary
  ];

  const impairmentTagAliases = [
    "hearing_impaired",
    "hearing_imp",
    "visual_impaired",
    "visual_imp",
    // Add more aliases if necessary
  ];

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

  const matchesDesiredTitles = (stream) => {
    if (desiredTitles.length === 0) {
      return false;
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

  const matchesUndesiredTitlesOrFlags = (stream) => {
    // First, check for impairment flags if forceFlaggedRemoval is enabled
    if (forceFlaggedRemoval && hasImpairmentFlag(stream)) {
      return true;
    }

    let streamTitle = "";
    if (stream.tags && stream.tags.title) {
      streamTitle = stream.tags.title.toLowerCase();
      for (const undesired of undesiredTitles) {
        if (streamTitle.includes(undesired)) {
          return true;
        }
      }
    }

    return false;
  };

  const hasImpairmentFlag = (stream) => {
    // Check both 'title' and specific tags/dispositions for impairment keywords
    let streamTitle = "";
    if (stream.tags && stream.tags.title) {
      streamTitle = stream.tags.title.toLowerCase();
      for (const impairment of impairmentKeywords) {
        if (streamTitle.includes(impairment)) {
          return true;
        }
      }
    }

    // Additionally, check specific tags for impairment flags
    if (stream.tags) {
      // Some containers use specific tags like HEARING_IMPAIRED=1
      for (const tag in stream.tags) {
        if (Object.prototype.hasOwnProperty.call(stream.tags, tag)) {
          const tagValue = stream.tags[tag];
          if (
            impairmentTagAliases.includes(tag.toLowerCase()) &&
            (tagValue === 1 ||
              tagValue === "1" ||
              tagValue === true ||
              tagValue === "true")
          ) {
            return true;
          }
        }
      }
    }

    // Additionally, check disposition flags for impairment
    if (stream.disposition) {
      // Check for 'commentary', 'hearing_impaired', 'visual_impaired'
      if (stream.disposition.commentary === 1) {
        return true;
      }
      if (stream.disposition.hearing_impaired === 1) {
        return true;
      }
      if (stream.disposition.visual_impaired === 1) {
        return true;
      }
      // Add more disposition flags if necessary
    }

    return false;
  };

  // Enhanced Logging: Log all subtitle streams' tags and disposition
  const logSubtitleStreamDetails = (stream, index) => {
    response.infoLog += `🔍 Subtitle Stream 0:s:${index} Details:\n`;
    response.infoLog += `    Codec Type: ${stream.codec_type}\n`;
    response.infoLog += `    Language: ${
      stream.tags && stream.tags.language ? stream.tags.language : "und"
    }\n`;
    response.infoLog += `    Title: ${
      stream.tags && stream.tags.title ? stream.tags.title : "N/A"
    }\n`;
    response.infoLog += `    Tags: ${
      stream.tags ? JSON.stringify(stream.tags) : "N/A"
    }\n`;
    response.infoLog += `    Disposition: ${
      stream.disposition ? JSON.stringify(stream.disposition) : "N/A"
    }\n`;
  };

  // First pass: Determine which subtitles to remove
  for (let i = 0; i < file.ffProbeData.streams.length; i++) {
    const stream = file.ffProbeData.streams[i];
    const codecType = stream.codec_type.toLowerCase();

    // Only process subtitle streams
    if (codecType === "subtitle") {
      // Log subtitle stream details for debugging
      logSubtitleStreamDetails(stream, subtitleIdx);

      let streamLang = "und";

      if (stream.tags && stream.tags.language) {
        streamLang = stream.tags.language.toLowerCase();
      }

      if (!subtitlesKeptPerLanguage[streamLang]) {
        subtitlesKeptPerLanguage[streamLang] = 0;
      }

      const subtitleIsForced = isSubtitleForced(stream);
      let keepSubtitle = false;
      let setForcedFlag = false;

      // **Priority 1:** Remove if matches impairment flags
      if (forceFlaggedRemoval && hasImpairmentFlag(stream)) {
        ffmpegCommandInsert += `-map -0:s:${subtitleIdx} `;
        response.infoLog += `☒ Removing subtitle stream 0:s:${subtitleIdx} due to impairment/commentary flags.\n`;
        convert = true;
      }
      // **Priority 2:** Keep if matches desired titles
      else if (matchesDesiredTitles(stream)) {
        keepSubtitle = true;
      }
      // **Priority 3:** Check undesired titles or flags
      else if (!matchesUndesiredTitlesOrFlags(stream)) {
        if (streamLang === "eng") {
          if (subtitlesKeptPerLanguage[streamLang] < 2) {
            keepSubtitle = true;
            if (subtitleIsForced) {
              setForcedFlag = true;
            }
          }
        } else if (languagesToKeep.includes(streamLang)) {
          if (subtitlesKeptPerLanguage[streamLang] < maxSubsPerLanguage) {
            keepSubtitle = true;
          }
        }
      }

      if (keepSubtitle) {
        // Keep this subtitle stream
        subtitlesKeptPerLanguage[streamLang] += 1;

        if (
          setForcedFlag &&
          !(stream.disposition && stream.disposition.forced === 1)
        ) {
          // Set the forced flag if it's a forced subtitle and not already set
          ffmpegCommandInsert += `-disposition:s:${subtitleIdx} +forced `;
          response.infoLog += `☒ Setting forced flag on subtitle stream 0:s:${subtitleIdx}.\n`;
          convert = true;
        }
      } else if (!forceFlaggedRemoval || !hasImpairmentFlag(stream)) {
        // Remove this subtitle stream
        ffmpegCommandInsert += `-map -0:s:${subtitleIdx} `;
        response.infoLog += `☒ Removing subtitle stream 0:s:${subtitleIdx} with language "${streamLang}".\n`;
        convert = true;
      }

      subtitleIdx += 1;
    }
  }

  // Handle closed captions removal if necessary
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

  // Finalize the preset and re-queue logic
  if (convert === true || response.processFile === true) {
    response.processFile = response.processFile || convert;
    response.preset = `, -map 0 ${ffmpegCommandInsert} -c copy -max_muxing_queue_size 9999`;
    response.container = `.${file.container}`;
    response.reQueueAfter = convert; // Only re-queue if a conversion occurred
    response.infoLog += "☒ Audio and subtitle flags have been set.\n";
  } else {
    response.infoLog +=
      "☑ No subtitles or closed captions needed to be removed or tagged.\n";
  }

  return response;
};

module.exports.details = details;
module.exports.plugin = plugin;
