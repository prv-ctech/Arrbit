#!/usr/bin/env bash
# -------------------------------------------------------------------------------------------------------------
# Arrbit - custom_scripts.bash
# Version: v1.0.2-gs2.8.2
# Purpose: Registers all custom scripts found in /config/arrbit/data/custom_script_*.json (modular, bulletproof, Golden Standard v2.8.2 enforced)
# -------------------------------------------------------------------------------------------------------------

source /config/arrbit/helpers/logging_utils.bash
source /config/arrbit/helpers/helpers.bash
arrbitPurgeOldLogs

SCRIPT_NAME="custom_scripts"
SCRIPT_VERSION="v1.0.2-gs2.8.2"
LOG_FILE="/config/logs/arrbit-${SCRIPT_NAME}-$(date +%Y_%m_%d-%H_%M).log"
PAYLOAD_DIR="/config/arrbit/data"
PATTERN="custom_script_*.json"

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

files_found=0
scripts_registered=0

for file in "$PAYLOAD_DIR"/$PATTERN; do
	if [[ ! -f $file ]]; then
		continue
	fi
	files_found=1

	jq -c '. as $in | (if type=="array" then $in[] else $in end)' "$file" | while read -r raw; do
		# Validate: Only process objects
		is_obj=$(echo "$raw" | jq -er 'type == "object"' 2>/dev/null || echo "false")
		if [[ $is_obj != "true" ]]; then
			log_error "Payload in $file is not a valid object (see log at /config/logs)"
			cat <<EOF | arrbitLogClean >>"$LOG_FILE"
[Arrbit] ERROR Invalid payload in $file
[WHY]: JSON value is not a valid object.
[FIX]: Ensure each entry in $file is a JSON object.
[Payload]
$raw
[/Payload]
EOF
			continue
		fi

		payload=$(echo "$raw" | jq 'del(.id)')
		name=$(echo "$payload" | jq -r '.name // empty')

		# Validate: Name is required
		if [[ -z $name ]]; then
			log_error "Payload missing .name property in $file (see log at /config/logs)"
			cat <<EOF | arrbitLogClean >>"$LOG_FILE"
[Arrbit] ERROR Payload missing .name property in $file
[WHY]: JSON object has no "name" field.
[FIX]: Add a "name" property to each custom script object.
[Payload]
$payload
[/Payload]
EOF
			continue
		fi

		# Skip if already registered (info only in log)
		if arr_api "${arrUrl}/api/${arrApiVersion}/notification" | jq -e --arg n "$name" '.[] | select(.name==$n)' >/dev/null; then
			log_info "Custom script '$name' already registered; skipping."
			printf '[Arrbit] SKIP custom script "%s" already exists\n' "$name" | arrbitLogClean >>"$LOG_FILE"
			continue
		fi

		log_info "Importing custom script: $name"
		printf '[Arrbit] Registering custom script "%s"\n[Payload]\n%s\n[/Payload]\n' "$name" "$payload" | arrbitLogClean >>"$LOG_FILE"

		# Do not leak apikey in URL!
		response=$(arr_api -X POST --data-raw "$payload" "${arrUrl}/api/${arrApiVersion}/notification")
		printf '[Response]\n%s\n[/Response]\n' "$response" | arrbitLogClean >>"$LOG_FILE"

		if echo "$response" | jq -e '.id' >/dev/null 2>&1; then
			log_info "SUCCESS: Custom script '$name' registered."
			printf '[Arrbit] SUCCESS custom script "%s" registered\n' "$name" | arrbitLogClean >>"$LOG_FILE"
			scripts_registered=$((scripts_registered + 1))
		else
			log_error "Failed to import custom script: $name (see log at /config/logs)"
			cat <<EOF | arrbitLogClean >>"$LOG_FILE"
[Arrbit] ERROR Failed to import custom script: $name
[WHY]: API did not return an id; likely cause: payload invalid or server/API error.
[FIX]: Check the payload JSON fields for correctness or see [Response] below.
[Response]
$response
[/Response]
EOF
		fi
	done
done

if [[ $files_found -eq 0 ]]; then
	log_info "No custom script payload files found."
	log_info "Done."
	exit 0
fi

log_info "The module was configured successfully."
log_info "Done."
exit 0
