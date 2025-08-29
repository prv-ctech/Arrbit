#!/usr/bin/env bash
# -------------------------------------------------------------------------------------------------------------
# Arrbit - metadata_profiles.bash
# Version: v1.0.2-gs2.8.2
# Purpose: Overwrite Lidarr metadata profiles with payload from metadata_profiles.json (Golden Standard v2.8.2 enforced)
# -------------------------------------------------------------------------------------------------------------

source /config/arrbit/helpers/logging_utils.bash
source /config/arrbit/helpers/helpers.bash

arrbitPurgeOldLogs

SCRIPT_NAME="metadata_profiles"
SCRIPT_VERSION="v1.0.2-gs2.8.2"
LOG_FILE="/config/logs/arrbit-${SCRIPT_NAME}-$(date +%Y_%m_%d-%H_%M).log"
JSON_PATH="/config/arrbit/data/payload-metadata_profiles.json"

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

if [[ ! -f $JSON_PATH ]]; then
	log_error "File not found: ${JSON_PATH} (see log at /config/logs)"
	cat <<EOF | arrbitLogClean >>"$LOG_FILE"
[Arrbit] ERROR File not found: $JSON_PATH
[WHY]: The file does not exist at the specified path.
[FIX]: Place a valid payload-metadata_profiles.json in $(dirname "$JSON_PATH").
EOF
	exit 1
fi

# Check if payload already exists and compare with current settings
existing_defs=$(arr_api "${arrUrl}/api/${arrApiVersion}/metadataprofile")
if [[ -z $existing_defs ]]; then
	log_error "Could not retrieve existing metadata profiles from Lidarr. (see log at /config/logs)"
	cat <<EOF | arrbitLogClean >>"$LOG_FILE"
[Arrbit] ERROR Could not retrieve existing metadata profiles from Lidarr.
[WHY]: API call for current metadata profiles failed.
[FIX]: Check ARR is running and accessible. See API response/logs for details.
EOF
	exit 1
fi

mapfile -t EXISTING_IDS < <(echo "$existing_defs" | jq -r '.[] | "\(.id)|\(.name|ascii_downcase)"')

# --- Pass 1: Compare all entries; only proceed if at least one differs ---
all_match=true
mapfile -t JSON_DEFS < <(jq -c '.[]' "$JSON_PATH")

for definition in "${JSON_DEFS[@]}"; do
	name=$(echo "$definition" | jq -r '.name')
	l_name=$(echo "$name" | tr '[:upper:]' '[:lower:]')
	payload=$(echo "$definition" | jq 'del(.id)')
	match_id=""
	match_payload=""

	for entry in "${EXISTING_IDS[@]}"; do
		IFS="|" read -r eid ename <<<"$entry"
		if [[ $ename == "$l_name" ]]; then
			match_id="$eid"
			match_payload=$(echo "$existing_defs" | jq --arg eid "$eid" '.[] | select(.id == ($eid|tonumber)) | del(.id)')
			break
		fi
	done

	if [[ -n $match_payload ]]; then
		if [[ "$(echo "$payload" | jq -S .)" != "$(echo "$match_payload" | jq -S .)" ]]; then
			all_match=false
			break
		fi
	else
		all_match=false
		break
	fi
done

if $all_match; then
	log_info "Predefined settings already present. Skipping..."
	echo "[Arrbit] Done."
	exit 0
fi

# Import predefined settings
log_info "Importing predefined settings..."

# --- Pass 2: Only update the ones that are different ---
for definition in "${JSON_DEFS[@]}"; do
	name=$(echo "$definition" | jq -r '.name')
	l_name=$(echo "$name" | tr '[:upper:]' '[:lower:]')
	payload=$(echo "$definition" | jq 'del(.id)')
	match_id=""
	match_payload=""

	for entry in "${EXISTING_IDS[@]}"; do
		IFS="|" read -r eid ename <<<"$entry"
		if [[ $ename == "$l_name" ]]; then
			match_id="$eid"
			match_payload=$(echo "$existing_defs" | jq --arg eid "$eid" '.[] | select(.id == ($eid|tonumber)) | del(.id)')
			break
		fi
	done

	# Only update if different or missing
	if [[ -n $match_payload ]]; then
		if [[ "$(echo "$payload" | jq -S .)" != "$(echo "$match_payload" | jq -S .)" ]]; then
			log_info "Importing metadata profile: ${name}"
			printf '[Arrbit] Updating Metadata Profile: %s (Lidarr ID: %s)\n[Payload]\n%s\n[/Payload]\n' "$name" "$match_id" "$payload" | arrbitLogClean >>"$LOG_FILE"
			response=$(arr_api -X PUT --data-raw "$payload" "${arrUrl}/api/${arrApiVersion}/metadataprofile/$match_id")
			printf '[Response]\n%s\n[/Response]\n' "$response" | arrbitLogClean >>"$LOG_FILE"

			if echo "$response" | jq -e '.id' >/dev/null 2>&1; then
				printf '[Arrbit] SUCCESS: Metadata profile processed: %s\n' "$name" | arrbitLogClean >>"$LOG_FILE"
			else
				log_error "Failed to import metadata profile: ${name} (see log at /config/logs)"
				cat <<EOF | arrbitLogClean >>"$LOG_FILE"
[Arrbit] ERROR Failed to import metadata profile: $name
[WHY]: API PUT request failed or invalid response.
[FIX]: Check payload and Lidarr server status. See [Response] section below.
[Response]
$response
[/Response]
EOF
			fi
		fi
	else
		log_info "Importing metadata profile: ${name}"
		printf '[Arrbit] Creating NEW Metadata Profile: %s\n[Payload]\n%s\n[/Payload]\n' "$name" "$payload" | arrbitLogClean >>"$LOG_FILE"
		response=$(arr_api -X POST --data-raw "$payload" "${arrUrl}/api/${arrApiVersion}/metadataprofile")
		printf '[Response]\n%s\n[/Response]\n' "$response" | arrbitLogClean >>"$LOG_FILE"

		if echo "$response" | jq -e '.id' >/dev/null 2>&1; then
			printf '[Arrbit] SUCCESS: Metadata profile created: %s\n' "$name" | arrbitLogClean >>"$LOG_FILE"
		else
			log_error "Failed to import metadata profile: ${name} (see log at /config/logs)"
			cat <<EOF | arrbitLogClean >>"$LOG_FILE"
[Arrbit] ERROR Failed to import metadata profile: $name
[WHY]: API POST request failed or invalid response.
[FIX]: Check payload and Lidarr server status. See [Response] section below.
[Response]
$response
[/Response]
EOF
		fi
	fi
done

log_info "The module was configured successfully."
echo "[Arrbit] Done."
exit 0
