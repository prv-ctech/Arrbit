#!/usr/bin/env bash
set -euo pipefail
# -------------------------------------------------------------------------------------------------------------
# Arrbit - autoconfig.bash
# Version: v1.0.2-gs2.8.2
# Purpose: Orchestrates Arrbit modules based on config flags in arrbit-config.conf (Golden Standard v2.8.2 enforced)
# -------------------------------------------------------------------------------------------------------------

source /config/arrbit/helpers/helpers.bash
source /config/arrbit/helpers/logging_utils.bash

arrbitPurgeOldLogs

SCRIPT_NAME="autoconfig"
# shellcheck disable=SC2034 # SCRIPT_VERSION for tooling
SCRIPT_VERSION="v1.0.2-gs2.8.2"
ARRBIT_ROOT="/config/arrbit"
export CONFIG_FILE="$ARRBIT_ROOT/config/arrbit-config.conf" # address SC2034 by exporting
MODULES_DIR="$ARRBIT_ROOT/modules"
LOG_DIR="/config/logs"
LOG_FILE="$LOG_DIR/arrbit-${SCRIPT_NAME}-$(date +%Y_%m_%d-%H_%M).log"

mkdir -p "$LOG_DIR" && touch "$LOG_FILE" && chmod 777 "$LOG_FILE"

# Banner (Golden Standard v2.8.2: colored banner required)
echo -e "${CYAN}[Arrbit]${NC} ${GREEN}Starting ${SCRIPT_NAME} service${NC} ${SCRIPT_VERSION} ..."

# --- 1. Check ENABLE_AUTOCONFIG flag first, fail fast with warning if not true ---
ENABLE_AUTOCONFIG=$(getFlag ENABLE_AUTOCONFIG)
if [[ ${ENABLE_AUTOCONFIG,,} != "true" ]]; then
	log_warning "Autoconfig service is OFF. Update ENABLE_AUTOCONFIG to 'true' in arrbit-config.conf."
	exit 0
fi

# --- 2. MODULES LIST (Add/remove modules here as required) ---
MODULES=(
	custom_formats
	custom_scripts
	media_management
	metadata_consumer
	metadata_profiles
	metadata_write
	quality_definitions # new
	quality_profiles
	track_naming
	ui_settings
)

# --- 3. CHECK IF ANY MODULES ARE ENABLED (error if all are disabled) ---
ENABLED_COUNT=0
for NAME in "${MODULES[@]}"; do
	FLAG="CONFIGURE_$(echo "$NAME" | tr '[:lower:]' '[:upper:]')"
	VAL=$(getFlag "$FLAG")
	if [[ -n ${VAL} && ${VAL,,} == "true" ]]; then
		((ENABLED_COUNT++))
	fi
done
if ((ENABLED_COUNT == 0)); then
	log_error "Autoconfig stopped: no CONFIGURE_* modules enabled. Update your configuration. (see log at /config/logs)"
	cat <<EOF | arrbitLogClean >>"$LOG_FILE"
[Arrbit] ERROR No modules enabled
[WHY]: All CONFIGURE_* flags are set to false in arrbit-config.conf
[FIX]: Edit /config/arrbit/config/arrbit-config.conf and set desired CONFIGURE_* flags to true
EOF
	exit 1
fi

# --- 4. RUN ENABLED MODULES ONLY (no internal flag logic in modules) ---
log_info "Starting modules..."

for NAME in "${MODULES[@]}"; do
	FLAG="CONFIGURE_$(echo "$NAME" | tr '[:lower:]' '[:upper:]')"
	VAL=$(getFlag "$FLAG")
	[[ -z ${VAL} || ${VAL,,} != "true" ]] && continue

	SCRIPT="$MODULES_DIR/${NAME}.bash"
	if [ -f "$SCRIPT" ]; then
		# Log orchestration details only (modules write to their own logs)
		printf '[Arrbit] Executing module: %s\n' "$NAME" | arrbitLogClean >>"$LOG_FILE"
		if bash "$SCRIPT"; then
			printf '[Arrbit] Module finished: %s (ok)\n' "$NAME" | arrbitLogClean >>"$LOG_FILE"
		else
			printf '[Arrbit] Module finished: %s (failed)\n' "$NAME" | arrbitLogClean >>"$LOG_FILE"
		fi
	else
		printf '[Arrbit] SKIP module not found: %s\n' "$SCRIPT" | arrbitLogClean >>"$LOG_FILE"
	fi
done

# --- 5. WRAP UP (Golden Standard v2.8.2: exactly 4 messages required) ---
log_info "Finished running all modules"
echo "[Arrbit] Done."

exit 0
