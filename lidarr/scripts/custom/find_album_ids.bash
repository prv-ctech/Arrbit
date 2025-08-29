#!/usr/bin/env bash
# -------------------------------------------------------------------------------------------------------------
# Arrbit - find_album_ids.bash
# Version: v1.2.1-gs2.8.3
# Purpose: Find album IDs in Lidarr using API data for robust artist/title matching
# -------------------------------------------------------------------------------------------------------------

SCRIPT_NAME="find_album_ids"
SCRIPT_VERSION="v1.2.1-gs2.8.3"
LOG_FILE="/config/logs/arrbit-${SCRIPT_NAME}-$(date +%Y_%m_%d-%H_%M).log"
touch "$LOG_FILE" && chmod 777 "$LOG_FILE"

source /config/arrbit/helpers/logging_utils.bash
source /config/arrbit/helpers/helpers.bash
# Temporarily redirect stdout to hide connection message
exec 3>&1 1>/dev/null
source /config/arrbit/connectors/arr_bridge.bash
exec 1>&3 3>&-
arrbitPurgeOldLogs

# ---- BANNER ----
echo -e "${CYAN}[Arrbit]${NC} ${GREEN}Album ID Finder ${NC}${SCRIPT_VERSION}..."

# Check if jq is installed
if ! command -v jq &> /dev/null; then
    echo -e "${CYAN}[Arrbit]${NC} ${RED}ERROR: jq is required but not installed.${NC}"
    exit 1
fi

# Function to display help
show_help() {    
    log_info "------------------------------------------"
    log_info "Usage: $0 [options] \"search term\""
    log_info ""
    log_info "Options:"
    log_info "  -h, --help         Show this help message"
    log_info "  -a, --artist NAME  Search for albums by artist name"
    log_info "  -t, --title NAME   Search for albums by title"
    log_info "  -f, --full         Show full album details"
    log_info ""
    log_info "Examples:"
    log_info "  $0 --artist \"Sam & Dave\"    # Find albums by Sam & Dave"
    log_info "  $0 --artist \"taylor swift\"      # Find albums by Taylor Swift"
    log_info "  $0 --title \"1989\"               # Find albums with '1989' in title"
    log_info ""
    log_info "Note: Search is case-insensitive and handles special characters automatically"   
}

# Default values
ARTIST_SEARCH=""
TITLE_SEARCH=""
FULL_DETAILS=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help; exit 0 ;;
        -a|--artist)
            ARTIST_SEARCH="${2:-}"; shift ;;
        -t|--title)
            TITLE_SEARCH="${2:-}"; shift ;;
        -f|--full)
            FULL_DETAILS=true ;;
        -*)
            # Unknown dash-prefixed token -> treat remainder (without leading dash) as artist search
            if [[ -z "$ARTIST_SEARCH" && -z "$TITLE_SEARCH" ]]; then
                ARTIST_SEARCH="${1#-}"
            fi ;;
        *)
            if [[ -z "$ARTIST_SEARCH" && -z "$TITLE_SEARCH" ]]; then
                ARTIST_SEARCH="$1"
            fi ;;
    esac
    shift
done

# Validate input
if [[ -z "$ARTIST_SEARCH" && -z "$TITLE_SEARCH" ]]; then
        log_error "No search criteria provided. Use -h for help."
        exit 1
fi

# Normalize accidental leading dash artifacts again (user might pass -Artist without quotes)
if [[ -n "$ARTIST_SEARCH" ]]; then
    ARTIST_SEARCH="$(echo "$ARTIST_SEARCH" | sed 's/^-*//')"
fi

# Get artists data from Lidarr API
if [[ -n "$ARTIST_SEARCH" ]]; then
    log_info "Fetching artists from Lidarr..."
    artists_url="${arrUrl}/api/${arrApiVersion}/artist?apikey=${arrApiKey}"
    artists_json=$(arr_api "$artists_url")
    
    if ! echo "$artists_json" | jq -e '.[0]' >/dev/null 2>&1; then
        log_error "Failed to retrieve artists from Lidarr API"
        exit 1
    fi
    
    # Search artists by cleanName and sortName (case insensitive)
    search_lower=$(echo "$ARTIST_SEARCH" | tr '[:upper:]' '[:lower:]')
    log_info "Searching for artist: $ARTIST_SEARCH"
    
    matching_artists=$(echo "$artists_json" | jq -r --arg search "$search_lower" '
        [.[] | select(
            (.cleanName | ascii_downcase | contains($search)) or
            (.sortName | ascii_downcase | contains($search)) or
            (.artistName | ascii_downcase | contains($search))
        )]
    ')
    
    artist_count=$(echo "$matching_artists" | jq 'length')
    if [[ "$artist_count" -eq 0 ]]; then
        log_info "No artists found matching: $ARTIST_SEARCH"
        exit 0
    fi
    
    log_info "Found $artist_count matching artists, fetching their albums..."
    
    # Get all albums for matching artists
    albums_url="${arrUrl}/api/${arrApiVersion}/album?apikey=${arrApiKey}"
    albums_json=$(arr_api "$albums_url")
    
    if ! echo "$albums_json" | jq -e '.[0]' >/dev/null 2>&1; then
        log_error "Failed to retrieve albums from Lidarr API"
        exit 1
    fi
    
    # Filter albums by matching artist IDs
    artist_ids=$(echo "$matching_artists" | jq -r '.[].id' | tr '\n' ',' | sed 's/,$//')
    filtered_albums=$(echo "$albums_json" | jq -r --arg ids "$artist_ids" '
        [$ids | split(",") | .[] | tonumber] as $artist_ids |
        [.[] | select(.artist.id as $aid | $artist_ids | index($aid))]
    ')
fi

# Search by album title
if [[ -n "$TITLE_SEARCH" ]]; then
    log_info "Fetching albums from Lidarr..."
    albums_url="${arrUrl}/api/${arrApiVersion}/album?apikey=${arrApiKey}"
    albums_json=$(arr_api "$albums_url")
    
    if ! echo "$albums_json" | jq -e '.[0]' >/dev/null 2>&1; then
        log_error "Failed to retrieve albums from Lidarr API"
        exit 1
    fi
    
    # Search albums by title (case insensitive)
    search_lower=$(echo "$TITLE_SEARCH" | tr '[:upper:]' '[:lower:]')
    log_info "Searching for album title: $TITLE_SEARCH"
    
    filtered_albums=$(echo "$albums_json" | jq -r --arg search "$search_lower" '
        [.[] | select(.title | ascii_downcase | contains($search))]
    ')
fi

# Check results
album_count=$(echo "$filtered_albums" | jq -r 'length')
if [[ "$album_count" -eq 0 ]]; then
    log_info "No matching albums found."
    exit 0
fi

log_info "Found $album_count albums:"
echo -e "${CYAN}[Arrbit]${NC} ${GREEN}------------------------------------------${NC}"

# Display results
if [[ "$FULL_DETAILS" == "true" ]]; then
    echo "$filtered_albums" | jq -r '.[] | "ID: \(.id)\nTitle: \(.title)\nArtist: \(.artist.artistName)\nRelease Date: \(.releaseDate)\nPath: \(.artist.path)/\(.title)\n------------------------------------------"'
else
    echo -e "${CYAN}[Arrbit]${NC} ${GREEN}ID | Title | Artist | Release Date${NC}"
    echo "$filtered_albums" | jq -r '.[] | "\(.id) | \(.title) | \(.artist.artistName) | \(.releaseDate)"' | while read -r line; do
        echo -e "${CYAN}[Arrbit]${NC} $line"
    done
fi

echo -e "${CYAN}[Arrbit]${NC} ${GREEN}------------------------------------------${NC}"
echo -e "${CYAN}[Arrbit]${NC} ${GREEN}To use an album ID with the tagger script:${NC}"
echo -e "${CYAN}[Arrbit]${NC} ${GREEN}bash enhanced_tagger_v3.bash <ALBUM_ID>${NC}"
echo -e "${CYAN}[Arrbit]${NC} ${GREEN}------------------------------------------${NC}"
