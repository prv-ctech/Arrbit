#!/usr/bin/env bash
# -------------------------------------------------------------------------------------------------------------
# Arrbit - tagger.bash - Automatic music tagging script for Lidarr using beets
# Version: v1.0.8-gs2.8.3
# Purpose: This script processes music albums from Lidarr using beets to improve metadata tagging.
# -------------------------------------------------------------------------------------------------------------

# MUST start with helpers and purge (per GS v2.8.3)
source /config/arrbit/helpers/logging_utils.bash
source /config/arrbit/helpers/helpers.bash
arrbitPurgeOldLogs

# Script information (reset for GS update)
SCRIPT_NAME="tagger"
SCRIPT_VERSION="v1.0.8-gs2.8.3"
LOG_FILE="/config/logs/arrbit-tagger-$(date +%Y_%m_%d-%H_%M).log"

# Create log directory/file and set permissive permissions (per GS)
mkdir -p "$(dirname "$LOG_FILE")" 2>/dev/null || true
touch "$LOG_FILE" 2>/dev/null || true
chmod 777 "$(dirname "$LOG_FILE")" "$LOG_FILE" 2>/dev/null || true

# Ensure temporary directory exists
TMP_DIR="/config/arrbit/tmp"
mkdir -p "$TMP_DIR" 2>/dev/null || true
chmod 777 "$TMP_DIR" 2>/dev/null || true
# Ensure ultra-permissive defaults for any new files/dirs we create during the run
umask 000
# Remove any legacy override file from older versions
rm -f "$TMP_DIR/beets-import-override.yaml" 2>/dev/null || true

# Cleanup temp artifacts on any exit (success or failure)
cleanup() {
    rm -f "$TMP_DIR/beets-lidarr-match" \
          "$TMP_DIR/library-lidarr.blb" 2>/dev/null || true
    rm -rf "$TMP_DIR/beets" 2>/dev/null || true
    # Also purge legacy override file if present
    rm -f "$TMP_DIR/beets-import-override.yaml" 2>/dev/null || true
}
trap cleanup EXIT

# Banner (only line allowed with echo -e; colors from logging_utils)
echo -e "${CYAN}[Arrbit]${NC} ${GREEN}Starting ${SCRIPT_NAME} script${NC} ${SCRIPT_VERSION} ..."

# Check if this is a test event from Lidarr (avoid connecting to ARR in test mode)
if [ "$lidarr_eventtype" == "Test" ]; then
    log_info "Tested Successfully"
    exit 0
fi

# Ignore rename-type events (case-insensitive) which often lack album ID and can cause loops
_evt_lower=$(printf '%s' "${lidarr_eventtype:-}" | tr '[:upper:]' '[:lower:]')
if [[ "${_evt_lower}" == *rename* || "${_evt_lower}" == *retag* ]]; then
    log_info "Ignoring ${lidarr_eventtype:-} event"
    exit 0
fi

# Connect to ARR only for real runs
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

# Get album ID from Lidarr
# This can come from environment variable or command line argument
if [ -z "$lidarr_album_id" ]; then
    lidarr_album_id="$1"
fi

# Verify we have an album ID
if [ -z "$lidarr_album_id" ]; then
    # If invoked by Lidarr but no album ID was provided for this event, skip quietly
    if [ -n "${lidarr_eventtype:-}" ]; then
        log_info "Ignoring ${lidarr_eventtype} event without album ID"
        exit 0
    fi
    # Otherwise (manual run without parameter), treat as an error
    log_error "No album ID provided (see log at /config/logs)"
    {
        echo "[WHY]: The script needs a Lidarr album ID to fetch album and track file info";
        echo "[FIX]: Run from Lidarr with a proper event payload or pass an album ID as the first argument";
    } >> "$LOG_FILE"
    exit 1
fi

# Retag events can fire per-track; wait briefly to let Lidarr finish batch retagging
_evt_lower=$(printf '%s' "${lidarr_eventtype:-}" | tr '[:upper:]' '[:lower:]')
 

# Fetch album information from Lidarr API
printf "[Arrbit] Fetching album information for ID: %s\n" "$lidarr_album_id" >> "$LOG_FILE"
# Build full URL for arr_bridge v1.1.0 wrapper (expects URL, not method)
album_info=$(arr_api "${arrUrl}/api/${arrApiVersion}/album/$lidarr_album_id")

if [ -z "$album_info" ]; then
    log_error "Failed to fetch album information (see log at /config/logs)"
    {
        echo "[WHY]: The Lidarr API returned an empty response or the request failed";
        echo "[FIX]: Confirm arr_url/api version/key via arr_bridge.bash and network reachability from this container";
    } >> "$LOG_FILE"
    exit 1
fi

# Extract album artist information
album_artist=$(echo "$album_info" | jq -r '.artist.artistName // empty')
album_artist_path=$(echo "$album_info" | jq -r '.artist.path // empty')

# Validate album metadata
if [ -z "$album_artist" ] || [ -z "$album_artist_path" ]; then
    log_error "Album metadata missing (artist or artist path) (see log at /config/logs)"
    {
        echo "[WHY]: The album payload from Lidarr did not include the artist name or path";
        echo "[FIX]: Verify the album exists in Lidarr and the artist has a valid root folder/path";
    } >> "$LOG_FILE"
    exit 1
fi

# Determine selected release and MBID (prefer monitored)
selected_release=$(echo "$album_info" | jq -c '
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
SELECTED_RELEASE_MBID=$(echo "$selected_release" | jq -r '.foreignReleaseId // empty')
SELECTED_RELEASE_ID=$(echo "$selected_release" | jq -r '.id // empty')
if [ -n "$SELECTED_RELEASE_MBID" ]; then
        printf "[Arrbit] Selected releaseId=%s mbid=%s\n" "$SELECTED_RELEASE_ID" "$SELECTED_RELEASE_MBID" >> "$LOG_FILE"
else
        printf "[Arrbit] No selected release MBID found; proceeding without MBID stamp.\n" >> "$LOG_FILE"
fi

# Get track path information
# Use original endpoint casing compatible with Lidarr and prior script
track_files=$(arr_api "${arrUrl}/api/${arrApiVersion}/trackFile?albumId=$lidarr_album_id")
# Avoid Broken pipe from head(1) by selecting within jq
track_path=$(echo "$track_files" | jq -r 'map(select((.path // "") != "")) | (.[0].path // empty)')
folder_path=$(dirname "$track_path")
album_folder_name=$(basename "$folder_path")

printf "[Arrbit] Processing :: %s :: Processing Files...\n" "$album_folder_name" >> "$LOG_FILE"
log_info "Processing album folder: $album_folder_name"
# Validate paths
if [ -z "$track_path" ] || [ -z "$folder_path" ] || [ "$folder_path" = "." ]; then
    log_error "Unable to determine album folder path from track files (see log at /config/logs)"
    {
        echo "[WHY]: No valid track file paths were returned for the album";
        echo "[FIX]: Ensure the album has imported tracks and Lidarr reports their paths";
    } >> "$LOG_FILE"
    exit 1
fi

if ! echo "$folder_path" | grep -Fq "$album_artist_path"; then
    log_error "ERROR :: $album_artist_path not found within \"$folder_path\" (see log at /config/logs)"
    {
        echo "[WHY]: The album folder resolved from track files does not match the expected artist path";
        echo "[FIX]: Verify Lidarr root folders and that the album's tracks are under the artist path";
    } >> "$LOG_FILE"
    exit 1
fi

if [ ! -d "$folder_path" ]; then
    log_error "ERROR :: \"$folder_path\" Folder is missing (see log at /config/logs)"
    {
        echo "[WHY]: The resolved album directory does not exist on disk";
        echo "[FIX]: Check your mounts and that media is present at the expected path";
    } >> "$LOG_FILE"
    exit 1
fi

 

# Check required tools
for bin in beet ffprobe metaflac jq find dirname basename; do
    if ! command -v "$bin" >/dev/null 2>&1; then
        log_error "Missing required tool: $bin (see log at /config/logs)"
        {
            echo "[WHY]: The executable '$bin' was not found in PATH inside the container";
            echo "[FIX]: Install or enable '$bin' in the image, or adjust PATH accordingly";
        } >> "$LOG_FILE"
        exit 1
    fi
done

# Process with Beets function
# This function handles the actual tagging process using beets
process_with_beets() {
    local process_folder="$1"
    
    printf "[Arrbit] %s :: Start Processing...\n" "$process_folder" >> "$LOG_FILE"
    
    # Check if folder contains FLAC files
    if ! find "$process_folder" -type f -iname "*.flac" | grep -q .; then
        log_error "$process_folder :: ERROR :: Only supports flac files, exiting... (see log at /config/logs)"
        # Add detailed error information to log
        echo "[WHY]: No FLAC files were found in the album folder" >> "$LOG_FILE"
        echo "[FIX]: This script only processes FLAC files. Convert your audio files to FLAC format" >> "$LOG_FILE"
        return 1
    fi
    
    # Start processing timer
    SECONDS=0
    
    # Clean up previous files if they exist
    if [ -f /config/arrbit/tmp/library-lidarr.blb ]; then
        rm /config/arrbit/tmp/library-lidarr.blb
        sleep 0.5
    fi
    
    # Do not remove external beets logs; rely on beets-config.yaml for its log path/rotation
    
    if [ -f "/config/arrbit/tmp/beets-lidarr-match" ]; then
        rm "/config/arrbit/tmp/beets-lidarr-match"
        sleep 0.5
    fi
    
    touch "/config/arrbit/tmp/beets-lidarr-match"
    sleep 0.5
    
    printf "[Arrbit] %s :: Begin matching with beets!\n" "$process_folder" >> "$LOG_FILE"

    # Optionally stamp album MBID into FLACs for deterministic matching
    if [ "${TAGGER_STAMP_MBID:-1}" != "0" ] && [ -n "$SELECTED_RELEASE_MBID" ]; then
        # Count FLACs first
        flac_count=$(find "$process_folder" -type f -iname "*.flac" | wc -l | tr -d '[:space:]')
        if [ "${flac_count:-0}" -gt 0 ]; then            
            printf "[Arrbit] %s :: Stamping MBID %s into %s FLAC(s)\n" "$process_folder" "$SELECTED_RELEASE_MBID" "$flac_count" >> "$LOG_FILE"
            find "$process_folder" -type f -iname "*.flac" -print0 | while IFS= read -r -d '' file; do
                metaflac --remove-tag=MUSICBRAINZ_ALBUMID "$file" 2>/dev/null || true
                metaflac --set-tag=MUSICBRAINZ_ALBUMID="$SELECTED_RELEASE_MBID" "$file" 2>/dev/null || true
            done
        else
            printf "[Arrbit] %s :: No FLAC files found for MBID stamping\n" "$process_folder" >> "$LOG_FILE"
        fi
    fi

    # Optionally stamp per-track MBIDs for perfect mapping
    if [ "${TAGGER_STAMP_TRACK_MBIDS:-1}" != "0" ]; then
        printf "[Arrbit] %s :: Fetching track and trackFile mappings for MB track IDs...\n" "$process_folder" >> "$LOG_FILE"
        tracks_json=$(arr_api "${arrUrl}/api/${arrApiVersion}/track?albumId=$lidarr_album_id")
        if [ -n "$tracks_json" ] && echo "$tracks_json" | jq -e '.[0].id' >/dev/null 2>&1; then
            # Build map: trackId -> foreignTrackId
            # Build map: trackId -> file path from pre-fetched track_files
            printf "[Arrbit] %s :: Stamping per-track MBIDs where available...\n" "$process_folder" >> "$LOG_FILE"
            # Iterate trackFiles to pair trackId with path, lookup foreignTrackId from tracks_json
            echo "$track_files" | jq -c '.[] | select((.path // "") != "") | {trackId: .trackId, path: .path}' | while IFS= read -r tf; do
                tid=$(echo "$tf" | jq -r '.trackId')
                fpath=$(echo "$tf" | jq -r '.path')
                # Lookup foreignTrackId for this trackId
                t_mbid=$(echo "$tracks_json" | jq -r ".[] | select(.id == $tid) | .foreignTrackId // empty")
                if [ -n "$t_mbid" ] && [ -f "$fpath" ]; then
                    metaflac --remove-tag=MUSICBRAINZ_TRACKID "$fpath" 2>/dev/null || true
                    metaflac --set-tag=MUSICBRAINZ_TRACKID="$t_mbid" "$fpath" 2>/dev/null || true
                fi
            done
        else
            printf "[Arrbit] %s :: No track data available; skipping per-track MBID stamping\n" "$process_folder" >> "$LOG_FILE"
        fi
    fi
    
    # Ensure beets configuration uses a writable directory
    export XDG_CONFIG_HOME="/config/arrbit/tmp"
    export BEETSDIR="/config/arrbit/tmp/beets"
    mkdir -p "$XDG_CONFIG_HOME" "$BEETSDIR" 2>/dev/null || true
    # Pre-create artresizer temp subdir to avoid permission issues
    mkdir -p "$TMP_DIR/beets/util_artresizer" 2>/dev/null || true

    # Ensure beets and its plugins (e.g., fetchart/artresizer) use a writable temp directory
    export TMPDIR="$TMP_DIR"
    export TEMP="$TMP_DIR"
    export TMP="$TMP_DIR"
    export MAGICK_TEMPORARY_PATH="$TMP_DIR"

    # Make sure everything we touch is world-writable and owned by current user
    chmod -R 777 "$TMP_DIR" "$BEETSDIR" "$XDG_CONFIG_HOME" 2>/dev/null || true
    uid_curr=$(id -u 2>/dev/null || echo 0)
    gid_curr=$(id -g 2>/dev/null || echo 0)
    chown -R "$uid_curr":"$gid_curr" "$TMP_DIR" "$BEETSDIR" "$XDG_CONFIG_HOME" 2>/dev/null || true
    # Also ensure the target album folder is writable for artwork/embedding
    chmod -R 777 "$process_folder" 2>/dev/null || true
    chown -R "$uid_curr":"$gid_curr" "$process_folder" 2>/dev/null || true

    # Preflight: ensure we can create temp files where artresizer expects
    tmp_test_file=""
    if ! tmp_test_file=$(mktemp -p "$TMP_DIR/beets/util_artresizer" "resize_IM_XXXXXX.jpg" 2>/dev/null); then
        # Retry after another permissive sweep
        chmod -R 777 "$TMP_DIR/beets" 2>/dev/null || true
        chown -R "$uid_curr":"$gid_curr" "$TMP_DIR/beets" 2>/dev/null || true
        tmp_test_file=$(mktemp -p "$TMP_DIR/beets/util_artresizer" "resize_IM_XXXXXX.jpg" 2>/dev/null || echo "")
    fi
    if [ -z "$tmp_test_file" ]; then
        # As a last resort, fall back to /tmp to avoid aborting the import
        printf "[Arrbit] WARN: %s :: Failed to create temp under %s; falling back to /tmp for artresizer.\n" "$process_folder" "$TMP_DIR/beets/util_artresizer" >> "$LOG_FILE"
        export TMPDIR="/tmp"
        export TEMP="/tmp"
        export TMP="/tmp"
        mkdir -p "/tmp/beets/util_artresizer" 2>/dev/null || true
        chmod -R 777 "/tmp/beets" 2>/dev/null || true
    else
        rm -f "$tmp_test_file" 2>/dev/null || true
    fi

    # Resolve beets config file (single canonical path)
    BEETS_CFG="/config/arrbit/config/beets-config.yaml"
    if [ ! -f "$BEETS_CFG" ]; then
        log_error "$process_folder :: Beets config not found at $BEETS_CFG"
        echo "[WHY]: Beets configuration file is missing" >> "$LOG_FILE"
        echo "[FIX]: Provide a valid beets config at $BEETS_CFG" >> "$LOG_FILE"
        return 1
    fi

    # Removed temporary import override (quiet_fallback); deterministic MBID stamping makes it unnecessary

    # Do not override or pre-create beets log; rely on value in beets-config.yaml (e.g., /config/logs/arrbit-beets.log)

    # Run beets import
    # -c: Config file path
    # -l: Library database path
    # -d: Destination directory
    # -qC: Quiet mode, no confirmation
        set -o pipefail
        BEETS_EXTRA_ARGS=()
        if [ -n "$SELECTED_RELEASE_MBID" ]; then
            BEETS_EXTRA_ARGS+=(--set "mb_albumid=$SELECTED_RELEASE_MBID")
        fi
        if [ "${TAGGER_BEETS_DEBUG:-0}" != "0" ]; then
            # Debug mode (unsanitized, verbose)
            beet -c "$BEETS_CFG" -l /config/arrbit/tmp/library-lidarr.blb -d "$process_folder" import -v -qC "${BEETS_EXTRA_ARGS[@]}" "$process_folder" >> "$LOG_FILE" 2>&1
            beets_status=$?
        else
            # Normal mode: sanitize output
            beet -c "$BEETS_CFG" -l /config/arrbit/tmp/library-lidarr.blb -d "$process_folder" import -qC "${BEETS_EXTRA_ARGS[@]}" "$process_folder" 2>&1 \
                | arrbitLogClean >> "$LOG_FILE"
            beets_status=${PIPESTATUS[0]}
        fi
        set +o pipefail
        # Fallback: if beets ran but reported 'Skipping.', retry as-is import to force tag writes
        if [ ${beets_status:-0} -eq 0 ] && grep -q "Skipping\." "$LOG_FILE"; then
            log_warning "$process_folder :: Beets reported 'Skipping.'; retrying as-is import"
            beet -c "$BEETS_CFG" -l /config/arrbit/tmp/library-lidarr.blb -d "$process_folder" import -qC -A "$process_folder" >> "$LOG_FILE" 2>&1 || true
        fi
        # Detect artresizer PermissionError and retry once under /tmp
        if [ ${beets_status:-0} -ne 0 ] && grep -q "util_artresizer" "$LOG_FILE" && grep -qi "PermissionError" "$LOG_FILE"; then
            log_warning "$process_folder :: Detected artresizer PermissionError; retrying import with TMPDIR=/tmp"
            old_tmpdir="$TMPDIR"; old_temp="$TEMP"; old_tmp="$TMP"
            export TMPDIR="/tmp"; export TEMP="/tmp"; export TMP="/tmp"; mkdir -p /tmp/beets/util_artresizer 2>/dev/null || true; chmod -R 777 /tmp/beets 2>/dev/null || true
            set -o pipefail
            if [ "${TAGGER_BEETS_DEBUG:-0}" != "0" ]; then
                beet -c "$BEETS_CFG" -l /config/arrbit/tmp/library-lidarr.blb -d "$process_folder" import -v -qC "${BEETS_EXTRA_ARGS[@]}" "$process_folder" >> "$LOG_FILE" 2>&1
                beets_status=$?
            else
                beet -c "$BEETS_CFG" -l /config/arrbit/tmp/library-lidarr.blb -d "$process_folder" import -qC "${BEETS_EXTRA_ARGS[@]}" "$process_folder" 2>&1 | arrbitLogClean >> "$LOG_FILE"
                beets_status=${PIPESTATUS[0]}
            fi
            set +o pipefail
            export TMPDIR="$old_tmpdir"; export TEMP="$old_temp"; export TMP="$old_tmp"
        fi
    if [ ${beets_status:-0} -ne 0 ]; then
        log_error "$process_folder :: Beets import failed (exit ${beets_status}) (see log at /config/logs)"
        echo "[WHY]: Beets failed to run or initialize its config directory" >> "$LOG_FILE"
        echo "[FIX]: Verify permissions on $XDG_CONFIG_HOME and the beets config at $BEETS_CFG" >> "$LOG_FILE"
        return ${beets_status}
    fi
    
    # Fix tags
	log_info "Fixing tags..."
    printf "[Arrbit] %s :: Fixing tags...\n" "$process_folder" >> "$LOG_FILE"
    printf "[Arrbit] %s :: Fixing flac tags...\n" "$process_folder" >> "$LOG_FILE"

    # Fix flac tags
    find "$process_folder" -type f -iname "*.flac" -print0 | while IFS= read -r -d '' file; do
        
        # Get artist credit from file
        artist_credit=$(ffprobe -loglevel 0 -print_format json -show_format -show_streams "$file" | jq -r ".format.tags.ARTIST_CREDIT" | sed "s/null//g" | sed "/^$/d")
        
        # Remove existing tags to ensure clean tagging
        metaflac --remove-tag=ARTIST "$file"
        metaflac --remove-tag=ALBUMARTIST "$file"
        metaflac --remove-tag=ALBUMARTIST_CREDIT "$file"
        metaflac --remove-tag=ALBUMARTISTSORT "$file"
        metaflac --remove-tag=ALBUM_ARTIST "$file"
        metaflac --remove-tag="ALBUM ARTIST" "$file"
        metaflac --remove-tag=ARTISTSORT "$file"
        metaflac --remove-tag=COMPOSERSORT "$file"
        
        # Set album artist tag
        metaflac --set-tag=ALBUMARTIST="$album_artist" "$file"
        
    # Set artist tag - use artist_credit if available, otherwise use album_artist
    if [ ! -z "$artist_credit" ]; then
            metaflac --set-tag=ARTIST="$artist_credit" "$file"
        else
            metaflac --set-tag=ARTIST="$album_artist" "$file"
        fi
    done
    
    printf "[Arrbit] %s :: Fixing tags Complete!\n" "$process_folder" >> "$LOG_FILE"
    
    # Clean up temporary files
    rm -f "/config/arrbit/tmp/beets-lidarr-match" \
        "/config/arrbit/tmp/library-lidarr.blb" 2>/dev/null || true
    # Clean up beets temp directories we created
    rm -rf "$TMP_DIR/beets" 2>/dev/null || true
    
    # Calculate and log processing duration
    duration=$SECONDS
    printf "[Arrbit] %s :: Finished in %d minutes and %d seconds!\n" "$process_folder" $(($duration / 60)) $(($duration % 60)) >> "$LOG_FILE"
}

# Process the album folder with beets
if process_with_beets "$folder_path"; then
    # Completion message
    log_info "The script ran successfully."
    log_info "Done."
else
    log_error "Processing failed for folder: $folder_path (see log at /config/logs)"
    {
        echo "[WHY]: An upstream step in the tagging pipeline failed";
        echo "[FIX]: Review errors above in the log; resolve the first failure cause and retry";
    } >> "$LOG_FILE"
    exit 1
fi

exit 0