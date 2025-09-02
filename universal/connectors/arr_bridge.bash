#!/usr/bin/env bash
# -------------------------------------------------------------------------------------------------------------
# Arrbit - arr_bridge.bash
# Version: v3.0.0-gs3.2.0 (Security-first minimal ARR API connector, no fallbacks)
# Purpose: Minimal secure ARR API connector with fail-fast error handling
# -------------------------------------------------------------------------------------------------------------

# Fixed base path and security-first initialization
ARRBIT_BASE="${ARRBIT_BASE:-/app/arrbit}"
source "$ARRBIT_BASE/universal/helpers/logging_utils.bash" || exit 1
source "$ARRBIT_BASE/universal/helpers/helpers.bash" || exit 1

arrbitPurgeOldLogs

SCRIPT_NAME="arr_bridge"
SCRIPT_VERSION="v3.0.0-gs3.2.0"

# Initialize logging (fail fast)
LOG_FILE="$ARRBIT_LOGS_DIR/arrbit-${SCRIPT_NAME}-$(date +%Y_%m_%d-%H_%M).log"
arrbitInitLog "$LOG_FILE" || { log_error "Could not initialize log file"; exit 1; }
arrbitBanner "$SCRIPT_NAME" "$SCRIPT_VERSION"

# Configuration validation (fail fast, no fallbacks)
CONFIG_XML="/config/config.xml"
[[ -f "$CONFIG_XML" ]] || { log_error "ARR config.xml not found"; exit 1; }

# Extract and validate ARR configuration
arr_url_base="$(cat "$CONFIG_XML" | xq | jq -r .Config.UrlBase)"
[[ "$arr_url_base" == "null" ]] && arr_url_base=""
[[ -n "$arr_url_base" ]] && arr_url_base="/$(printf '%s' "$arr_url_base" | sed 's|^/||;s|/$||')"

arr_api_key="$(cat "$CONFIG_XML" | xq | jq -r .Config.ApiKey)"
[[ -z "$arr_api_key" || "$arr_api_key" == "null" ]] && { log_error "API key not found"; exit 1; }

arr_instance_name="$(cat "$CONFIG_XML" | xq | jq -r .Config.InstanceName)"
[[ "$arr_instance_name" == "null" ]] && { log_error "Instance name not found"; exit 1; }

arr_port="$(cat "$CONFIG_XML" | xq | jq -r .Config.Port)"
[[ -z "$arr_port" || "$arr_port" == "null" ]] && { log_error "API port not found"; exit 1; }

# Build API URL (no overrides, fail fast)
arrUrl="http://127.0.0.1:${arr_port}${arr_url_base}"
arrApiKey="$arr_api_key"

# API version detection (try v3 first, then v1 for Lidarr compatibility)
arrApiVersion="v3"
response="$(curl -s --fail -H "X-Api-Key: $arrApiKey" "${arrUrl}/api/v3/system/status" 2>/dev/null)" || {
  # Fallback to v1 for Lidarr compatibility
  arrApiVersion="v1"
  response="$(curl -s --fail -H "X-Api-Key: $arrApiKey" "${arrUrl}/api/v1/system/status" 2>/dev/null)" || \
  { log_error "Could not connect to ARR API (neither v3 nor v1)"; exit 1; }
}

echo "$response" | jq -e '.instanceName' >/dev/null 2>&1 || { log_error "Invalid API response"; exit 1; }

# Export validated configuration
export arrApiKey arrUrl arrApiVersion

# Minimal API availability check (fail fast)
curl -s --fail -H "X-Api-Key: $arrApiKey" "${arrUrl}/api/${arrApiVersion}/system/status" >/dev/null 2>&1 || \
{ log_error "ARR API not available"; exit 1; }

# Minimal API call wrapper
arr_api() {
  curl -s --fail -H "X-Api-Key: $arrApiKey" -H "Content-Type: application/json" "$@"
}
export -f arr_api

log_info "Connected to ${arr_instance_name}"
# If sourced, return; if executed directly, exit
return 0 2>/dev/null || exit 0