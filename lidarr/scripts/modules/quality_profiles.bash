#!/usr/bin/env bash
# -------------------------------------------------------------------------------------------------------------
# Arrbit - quality_profiles.bash
# Version: v1.0.4-gs2.8.2
# Purpose: Import new quality profiles, always map live custom format IDs, skip duplicates, no deletion logic.
# -------------------------------------------------------------------------------------------------------------

source /config/arrbit/helpers/logging_utils.bash
source /config/arrbit/helpers/helpers.bash

arrbitPurgeOldLogs

SCRIPT_NAME="quality_profiles"
SCRIPT_VERSION="v1.0.4-gs2.8.2"
LOG_FILE="/config/logs/arrbit-${SCRIPT_NAME}-$(date +%Y_%m_%d-%H_%M).log"

mkdir -p /config/logs && touch "$LOG_FILE" && chmod 777 "$LOG_FILE"

echo -e "${CYAN}[Arrbit]${NC} ${GREEN}Starting ${SCRIPT_NAME} module${NC} ${SCRIPT_VERSION} ..."

if ! source /config/arrbit/connectors/arr_bridge.bash; then
	log_error "Could not source arr_bridge.bash (Required for API access, check Arrbit setup) (see log at /config/logs)"
	cat <<EOF | arrbitLogClean >>"$LOG_FILE"
[Arrbit] ERROR Could not source arr_bridge.bash
[WHY]: arr_bridge.bash is missing or failed to source.
[FIX]: Verify /config/arrbit/connectors/arr_bridge.bash exists and is correct. See log for details.
EOF
	exit 1
fi

# Import predefined settings
log_info "Importing predefined settings..."

CUSTOM_FORMATS_ENABLED=$(getFlag "CONFIGURE_CUSTOM_FORMATS")
if [[ ${CUSTOM_FORMATS_ENABLED,,} == "true" ]]; then
	# Payloads WITH custom formats
	REPLACE_JSON="/config/arrbit/data/payload-quality_profiles-with_custom_formats.json"
	# Ensure payload exists
	if [[ ! -f $REPLACE_JSON ]]; then
		log_error "File not found: ${REPLACE_JSON} (see log at /config/logs)"
		cat <<EOF | arrbitLogClean >>"$LOG_FILE"
[Arrbit] ERROR File not found: $REPLACE_JSON
[WHY]: The file does not exist at the specified path.
[FIX]: Place a valid payload-quality_profiles-with_custom_formats.json in $(dirname "$REPLACE_JSON").
EOF
		exit 1
	fi
	if [[ ! -f $REPLACE_JSON ]]; then
		log_error "File not found: ${REPLACE_JSON} (see log at /config/logs)"
		cat <<EOF | arrbitLogClean >>"$LOG_FILE"
[Arrbit] ERROR File not found: $REPLACE_JSON
[WHY]: The file does not exist at the specified path.
[FIX]: Place a valid payload-quality_profiles-with_custom_formats.json in $(dirname "$REPLACE_JSON").
EOF
		exit 1
	fi

	# Get custom formats from Lidarr
	log_info "Fetching live custom format IDs from Lidarr..."
	CUSTOM_FORMATS_RAW=$(arr_api "${arrUrl}/api/${arrApiVersion}/customformat")
	if [[ -z $CUSTOM_FORMATS_RAW ]]; then
		log_error "Could not retrieve custom formats from Lidarr. (see log at /config/logs)"
		cat <<EOF | arrbitLogClean >>"$LOG_FILE"
[Arrbit] ERROR Could not retrieve custom formats from Lidarr.
[WHY]: API call for custom formats failed.
[FIX]: Ensure Lidarr is running and API is accessible. See log for details.
EOF
		exit 1
	fi

	# Build a jq map for name → id
	CF_JQ_MAP=$(echo "$CUSTOM_FORMATS_RAW" | jq -r 'map({(.name): .id}) | add')

	# Remap payload formatItems names → live ids
	REMAPPED_JSON=$(jq --argjson cfmap "$CF_JQ_MAP" '
    map(
      .formatItems |= map(
        .format = ($cfmap[.name] // -1)
      )
    )
  ' "$REPLACE_JSON")

	# Validate: if any .format == -1, error!
	if echo "$REMAPPED_JSON" | jq '.[]|.formatItems[]|select(.format == -1)' | grep -q .; then
		log_error "Mismatch: One or more custom formats in your profile payload do not exist in Lidarr. Check for typos or import order. (see log at /config/logs)"
		cat <<EOF | arrbitLogClean >>"$LOG_FILE"
[Arrbit] ERROR Mismatched custom formats in payload
[WHY]: One or more custom formats in your profile payload do not exist in Lidarr.
[FIX]: Check for typos in your payload or verify that all custom formats are present in Lidarr before running this script.
EOF
		exit 1
	fi

	REPLACEMENTS_JSON="$REMAPPED_JSON"
else
	# Payloads WITHOUT custom formats
	REPLACE_JSON="/config/arrbit/data/payload-quality_profiles-no_custom_formats.json"
	if [[ ! -f $REPLACE_JSON ]]; then
		log_error "File not found: ${REPLACE_JSON} (see log at /config/logs)"
		cat <<EOF | arrbitLogClean >>"$LOG_FILE"
[Arrbit] ERROR File not found: $REPLACE_JSON
[WHY]: The file does not exist at the specified path.
[FIX]: Place a valid payload-quality_profiles-no_custom_formats.json in $(dirname "$REPLACE_JSON").
EOF
		exit 1
	fi
	REPLACEMENTS_JSON=$(cat "$REPLACE_JSON")
fi

existing_profiles=$(arr_api "${arrUrl}/api/${arrApiVersion}/qualityprofile")
if [[ -z $existing_profiles ]]; then
	log_error "Could not retrieve existing quality profiles from Lidarr. (see log at /config/logs)"
	cat <<EOF | arrbitLogClean >>"$LOG_FILE"
[Arrbit] ERROR Could not retrieve existing quality profiles from Lidarr.
[WHY]: API call for quality profiles failed.
[FIX]: Ensure Lidarr is running and API is accessible. See log for details.
EOF
	exit 1
fi

mapfile -t EXISTING_NAMES < <(echo "$existing_profiles" | jq -r '.[].name' | tr '[:upper:]' '[:lower:]')
mapfile -t REPLACEMENTS < <(echo "$REPLACEMENTS_JSON" | jq -c '.[]')

# If everything already exists, standard skip message
all_exist=true
for profile in "${REPLACEMENTS[@]}"; do
	name=$(echo "$profile" | jq -r '.name')
	lname=$(echo "$name" | tr '[:upper:]' '[:lower:]')
	if ! printf '%s\n' "${EXISTING_NAMES[@]}" | grep -Fxq "$lname"; then
		all_exist=false
		break
	fi
done
if $all_exist; then
	log_info "Predefined settings already present. Skipping..."
	log_info "Done."
	exit 0
fi

skipped_any=false

for profile in "${REPLACEMENTS[@]}"; do
	name=$(echo "$profile" | jq -r '.name')
	lname=$(echo "$name" | tr '[:upper:]' '[:lower:]')

	if printf '%s\n' "${EXISTING_NAMES[@]}" | grep -Fxq "$lname"; then
		skipped_any=true
		continue
	fi

	payload=$(echo "$profile" | jq 'del(.id)')

	log_info "Importing replacement quality profile: $name"
	printf '[Arrbit] Importing replacement profile: %s\n[Payload]\n%s\n[/Payload]\n' "$name" "$payload" | arrbitLogClean >>"$LOG_FILE"
	response=$(arr_api -X POST --data-raw "$payload" "${arrUrl}/api/${arrApiVersion}/qualityprofile")
	printf '[Arrbit] API Response:\n%s\n' "$response" | arrbitLogClean >>"$LOG_FILE"
	if echo "$response" | jq -e '.id' >/dev/null 2>&1; then
		printf '[Arrbit] SUCCESS Replacement profile created: %s\n' "$name" | arrbitLogClean >>"$LOG_FILE"
		EXISTING_NAMES+=("$lname")
	else
		log_error "Failed to create replacement profile: $name (see log at /config/logs)"
		cat <<EOF | arrbitLogClean >>"$LOG_FILE"
[Arrbit] ERROR Failed to create replacement profile: $name
[WHY]: API POST request failed or invalid response.
[FIX]: Check payload and Lidarr server status. See [API Response] section below.
[API Response]
$response
[/API Response]
EOF
	fi
done

if $skipped_any; then
	log_info "Quality profiles already exist - skipping."
fi

log_info "The module was configured successfully."
log_info "Done."
exit 0
