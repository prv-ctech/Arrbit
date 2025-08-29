#!/usr/bin/env bash
# -------------------------------------------------------------------------------------------------------------
# Arrbit - custom_formats.bash
# Version: v1.0.3-gs2.8.2
# Purpose: Import custom formats from JSON into Lidarr (Golden Standard v2.8.2 enforced)
# -------------------------------------------------------------------------------------------------------------

source /config/arrbit/helpers/logging_utils.bash
source /config/arrbit/helpers/helpers.bash

arrbitPurgeOldLogs

SCRIPT_NAME="custom_formats"
SCRIPT_VERSION="v1.0.3-gs2.8.2"
LOG_FILE="/config/logs/arrbit-${SCRIPT_NAME}-$(date +%Y_%m_%d-%H_%M).log"
JSON_PATH="/config/arrbit/data/payload-custom_formats.json"

mkdir -p /config/logs && touch "$LOG_FILE" && chmod 777 "$LOG_FILE"

echo -e "${CYAN}[Arrbit]${NC} ${GREEN}Starting ${SCRIPT_NAME} module${NC} ${SCRIPT_VERSION} ..."

# Source arr_bridge for API variables and arr_api wrapper
if ! source /config/arrbit/connectors/arr_bridge.bash; then
	log_error "Could not source arr_bridge.bash (Required for API access, check Arrbit setup) (see log at /config/logs)"
	cat <<EOF | arrbitLogClean >>"$LOG_FILE"
[Arrbit] ERROR Could not source arr_bridge.bash
[WHY]: arr_bridge.bash is missing or failed to source.
[FIX]: Verify /config/arrbit/connectors/arr_bridge.bash exists and is correct. See log for details.
EOF
	exit 1
fi

# Check that payload JSON exists
if [[ ! -f $JSON_PATH ]]; then
	log_error "File not found: ${JSON_PATH} (see log at /config/logs)"
	cat <<EOF | arrbitLogClean >>"$LOG_FILE"
[Arrbit] ERROR File not found: $JSON_PATH
[WHY]: The file does not exist at the specified path.
[FIX]: Place a valid payload-custom_formats.json in $(dirname "$JSON_PATH").
EOF
	exit 1
fi

# Import predefined settings
log_info "Importing predefined settings..."

# Query custom formats API, robust error detection
api_response=$(arr_api "${arrUrl}/api/${arrApiVersion}/customformat")
if ! echo "$api_response" | jq . >/dev/null 2>&1; then
	log_error "Failed to parse custom format list from API (see log at /config/logs)"
	cat <<EOF | arrbitLogClean >>"$LOG_FILE"
[Arrbit] ERROR Invalid response from Lidarr API for customformat
[WHY]: API unreachable, misconfigured, or returned invalid data.
[FIX]: Check your Lidarr API status, config, or permissions.
[Response]
$api_response
[/Response]
EOF
	exit 1
fi

# Extract names (can be empty if no custom formats exist)
existing_names=$(echo "$api_response" | jq -r '.[].name' | tr '[:upper:]' '[:lower:]')

# Read all custom formats from JSON
mapfile -t JSON_FORMATS < <(jq -c '.[]' "$JSON_PATH")

# Check if all exist already
all_exist=true
for format in "${JSON_FORMATS[@]}"; do
	format_name=$(echo "$format" | jq -r '.name')
	lowercase_name=$(echo "$format_name" | tr '[:upper:]' '[:lower:]')
	if ! echo "$existing_names" | grep -Fxq "$lowercase_name"; then
		all_exist=false
		break
	fi
done

if $all_exist; then
	log_info "Predefined settings already present. Skipping..."
	log_info "Done."
	exit 0
fi

# Import only missing formats
for format in "${JSON_FORMATS[@]}"; do
	format_name=$(echo "$format" | jq -r '.name')
	lowercase_name=$(echo "$format_name" | tr '[:upper:]' '[:lower:]')
	payload=$(echo "$format" | jq 'del(.id)')
	if echo "$existing_names" | grep -Fxq "$lowercase_name"; then
		continue
	fi
	log_info "Importing custom format: ${format_name}"
	response=$(arr_api -X POST --data-raw "$payload" "${arrUrl}/api/${arrApiVersion}/customformat")
	printf '[Arrbit] API Response:\n%s\n' "$response" | arrbitLogClean >>"$LOG_FILE"
	if echo "$response" | jq -e '.id' >/dev/null 2>&1; then
		printf '[Arrbit] SUCCESS Custom format created: %s\n' "$format_name" | arrbitLogClean >>"$LOG_FILE"
	else
		log_error "Failed to import format: ${format_name} (see log at /config/logs)"
		cat <<EOF | arrbitLogClean >>"$LOG_FILE"
[Arrbit] ERROR Failed to create custom format: $format_name
[WHY]: API failed to return an id. Likely cause: payload invalid or API/server error.
[FIX]: Check payload JSON fields for correctness, or see [Response] section below for more info.
[Response]
$response
[/Response]
EOF
	fi
done

log_info "The module was configured successfully."
log_info "Done."
exit 0
