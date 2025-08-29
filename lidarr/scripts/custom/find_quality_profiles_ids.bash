#!/usr/bin/env bash
# -------------------------------------------------------------------------------------------------------------
# Arrbit - find_quality_profiles_ids.bash
# Version: v1.0.11-gs2.8.3
# Purpose: List all Lidarr Quality Profile names and IDs (using find_album_ids pattern)
# -------------------------------------------------------------------------------------------------------------

SCRIPT_NAME="find_quality_profiles_ids"
SCRIPT_VERSION="v1.0.11-gs2.8.3"
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
echo -e "${CYAN}[Arrbit]${NC} ${GREEN}Quality Profile ID Finder ${NC}${SCRIPT_VERSION}..."

# Check if jq is installed
if ! command -v jq &> /dev/null; then
    echo -e "${CYAN}[Arrbit]${NC} ${RED}ERROR: jq is required but not installed.${NC}"
    exit 1
fi

# Get quality profiles from Lidarr (exactly like find_album_ids.bash)
profiles_url="${arrUrl}/api/${arrApiVersion}/qualityprofile?apikey=${arrApiKey}"
profiles_json=$(arr_api "$profiles_url")

# Check if API call was successful
if ! echo "$profiles_json" | jq -e '.[0]' >/dev/null 2>&1; then
    echo -e "${CYAN}[Arrbit]${NC} ${RED}ERROR: Failed to retrieve quality profiles from Lidarr API${NC}"
    exit 1
fi

# Count profiles
profile_count=$(echo "$profiles_json" | jq -r 'length')
log_info Found $profile_count quality profiles.

echo -e "${CYAN}[Arrbit]${NC} ${GREEN}------------------------------------------${NC}"

# Display the results
echo -e "${CYAN}[Arrbit]${NC} ${GREEN}quality profile name | id${NC}"
echo "$profiles_json" | jq -r '.[] | "\(.name) | \(.id)"' | while read -r line; do
    echo -e "${CYAN}[Arrbit]${NC} $line"
done

echo -e "${CYAN}[Arrbit]${NC} ${GREEN}------------------------------------------${NC}"
