#!/usr/bin/env bash
# -------------------------------------------------------------------------------------------------------------
# Arrbit - find_release_id.bash
# Version: v1.0.0-gs2.8.3
# Purpose: Given a Lidarr album ID, fetch releases and print the selected release
#          in the form: Title / Disambiguation / Country. Falls back heuristically
#          when no release is marked as monitored.
# -------------------------------------------------------------------------------------------------------------

# MUST start with helpers and purge (per GS v2.8.3)
source /config/arrbit/helpers/logging_utils.bash
source /config/arrbit/helpers/helpers.bash
arrbitPurgeOldLogs

# Script information
SCRIPT_NAME="find_release_id"
SCRIPT_VERSION="v1.0.0-gs2.8.3"
LOG_FILE="/config/logs/arrbit-${SCRIPT_NAME}-$(date +%Y_%m_%d-%H_%M).log"
mkdir -p "$(dirname "$LOG_FILE")" 2>/dev/null || true
touch "$LOG_FILE" 2>/dev/null || true
chmod 777 "$(dirname "$LOG_FILE")" "$LOG_FILE" 2>/dev/null || true

# Redirect all stdout/stderr to log; keep original stdout on FD 3 for final prints
exec 3>&1
exec 1>>"$LOG_FILE" 2>&1

# Banner (log-only)
echo -e "[Arrbit] Starting ${SCRIPT_NAME} ${SCRIPT_VERSION} ..." >> "$LOG_FILE"

# Connect to ARR bridge
if [ ! -f "/config/arrbit/connectors/arr_bridge.bash" ]; then
  log_error "arr_bridge.bash not found at /config/arrbit/connectors/arr_bridge.bash (see log at /config/logs)"
  {
    echo "[WHY]: The universal arr_bridge connector is missing or not mounted at the canonical path";
    echo "[FIX]: Ensure /config/arrbit/connectors/arr_bridge.bash exists and is readable by this container";
  } >> "$LOG_FILE"
  exit 11
fi
# shellcheck source=/dev/null
source "/config/arrbit/connectors/arr_bridge.bash"

# Args
ALBUM_ID="${lidarr_album_id:-}"
LIST_ALL=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    -a|--album-id)
      ALBUM_ID="$2"; shift ;;
    -l|--list-all)
      LIST_ALL=1 ;;
    -h|--help)
  echo -e "${CYAN}[Arrbit]${NC} Usage: $0 --album-id <ID> [--list-all]"; exit 0 ;;
    *)
      # First non-flag argument as album id
      if [[ -z "$ALBUM_ID" ]]; then ALBUM_ID="$1"; else :; fi ;;
  esac
  shift
done

if [[ -z "$ALBUM_ID" ]]; then
  log_error "No album ID provided (see log at /config/logs)"
  {
    echo "[WHY]: The script needs a Lidarr album ID to fetch releases";
    echo "[FIX]: Pass an album ID via --album-id <ID> or set lidarr_album_id env";
  } >> "$LOG_FILE"
  exit 1
fi

# Fetch album JSON
printf "[Arrbit] Fetching album information for ID: %s\n" "$ALBUM_ID" >> "$LOG_FILE"
album_json=$(arr_api "${arrUrl}/api/${arrApiVersion}/album/$ALBUM_ID")
if [[ -z "$album_json" ]] || ! echo "$album_json" | jq -e '.id' >/dev/null 2>&1; then
  log_error "Failed to fetch album information (see log at /config/logs)"
  {
    echo "[WHY]: The Lidarr API returned an empty or invalid response";
    echo "[FIX]: Verify arr URL/API key via arr_bridge.bash and connectivity";
  } >> "$LOG_FILE"
  exit 1
fi

album_title=$(echo "$album_json" | jq -r '.title // empty')
releases_json=$(echo "$album_json" | jq -c '.releases // []')
if [[ -z "$releases_json" ]] || [[ "$(echo "$releases_json" | jq 'length')" -eq 0 ]]; then
  log_error "No releases array found for album $ALBUM_ID (see log at /config/logs)"
  {
    echo "[WHY]: The album payload has no 'releases' list";
    echo "[FIX]: Confirm the album exists in Lidarr with populated releases";
  } >> "$LOG_FILE"
  exit 1
fi

if [[ $LIST_ALL -eq 1 ]]; then
  echo "[Arrbit] Listing all releases for album: ${album_title} (ID: ${ALBUM_ID})" >> "$LOG_FILE"
  echo "$releases_json" | jq -r '.[] | 
    .country as $c | 
    (if ($c|type=="array") then $c else (if ($c==null) then [] else [$c] end) end) as $ca |
    ( .trackCount // (if (.tracks!=null) then (.tracks|length) elif (.media!=null) then (.media | map(.trackCount // 0) | add) else 0 end) ) as $tc |
    "\(.id) | mon=\(.monitored // false) | title=\(.title // \"\") | disambig=\(.disambiguation // \"\") | country=\($ca | join(\", ")) | tracks=\($tc) | media=\(.format // (.media[0].mediumFormat // \"\")) | mbid=\(.foreignReleaseId // \"\")"'
fi

# Select the release: prefer monitored==true; else prefer Digital Media, then Worldwide/US; else first
selected=$(echo "$album_json" | jq -c '
  def fmt_media: (.format // (.media[0].mediumFormat // ""));
  def is_digital: (fmt_media | test("Digital Media|File"; "i"));
  def to_ca: (.country as $c | (if ($c|type=="array") then $c else (if ($c==null) then [] else [$c] end) end));
  def is_global: (to_ca | join(";") | test("\\[Worldwide\\]|United States"));
  .releases as $r | if ($r | map(select(.monitored == true)) | length) > 0 then
    $r | map(select(.monitored == true)) | .[0]
  else
    $r | sort_by(
      (if (is_digital) then 0 else 1 end),
      (if (is_global) then 0 else 1 end)
    ) | .[0]
  end')

if [[ -z "$selected" ]] || [[ "$selected" == "null" ]]; then
  log_error "Failed to select a release for album $ALBUM_ID (see log at /config/logs)"
  {
    echo "[WHY]: No release matched monitored flag or fallback heuristics";
    echo "[FIX]: Check album's releases in Lidarr UI and ensure one is monitored";
  } >> "$LOG_FILE"
  exit 1
fi

sel_title=$(echo "$selected" | jq -r '.title // ""')
sel_disambig=$(echo "$selected" | jq -r '.disambiguation // ""')
sel_country=$(echo "$selected" | jq -r '.country as $c | (if ($c|type=="array") then $c else (if ($c==null) then [] else [$c] end) end) | join(", ")')
sel_tracks=$(echo "$selected" | jq -r '.trackCount // (if (.tracks!=null) then (.tracks|length) elif (.media!=null) then (.media | map(.trackCount // 0) | add) else 0 end)')
sel_id=$(echo "$selected" | jq -r '.id')
sel_mbid=$(echo "$selected" | jq -r '.foreignReleaseId // ""')

echo "[Arrbit] Selected release for ${album_title} (albumId: ${ALBUM_ID}):" >> "$LOG_FILE"
echo "[Arrbit] ${sel_title} | ${sel_disambig} | ${sel_country} | ${sel_tracks} tracks" >> "$LOG_FILE"
printf "[Arrbit] Details: releaseId=%s mbid=%s\n" "$sel_id" "$sel_mbid" >> "$LOG_FILE"

# Print only two lines to original STDOUT: header and data
printf "%b\n" "${GREEN}release title | version | country | tracks${NC}" >&3
printf "%s | %s | %s | %s tracks\n" "$sel_title" "$sel_disambig" "$sel_country" "$sel_tracks" >&3
exit 0
