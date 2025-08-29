#!/usr/bin/env bash
set -euo pipefail
# -------------------------------------------------------------------------------------------------------------
# Arrbit - plugins
# Version: v1.0.1-gs2.8.2
# Purpose: Modular installer for Lidarr plug-ins (Golden Standard v2.8.2 compliant).
# -------------------------------------------------------------------------------------------------------------

source /config/arrbit/helpers/logging_utils.bash
source /config/arrbit/helpers/helpers.bash

arrbitPurgeOldLogs

chmod -R 777 /config/arrbit/
chmod -R 777 /config/plugins/
chmod -R 777 /config/logs/

PLUGINS_DIR="/config/plugins"
SCRIPT_NAME="plugins"
# shellcheck disable=SC2034 # SCRIPT_VERSION exposed for external tooling
SCRIPT_VERSION="v1.0.1-gs2.8.2"
LOG_FILE="/config/logs/arrbit-${SCRIPT_NAME}-$(date +%Y_%m_%d-%H_%M).log"
export CONFIG_FILE="/config/arrbit/config/arrbit-config.conf" # previously SC2034; exported for external consumers

touch "$LOG_FILE" && chmod 777 "$LOG_FILE"

# --- Banner (color allowed on first line) ---
echo -e "${CYAN}[Arrbit]${NC} ${GREEN}Starting ${SCRIPT_NAME} service${NC} ${SCRIPT_VERSION}"

# --- Get config flags (robust: trims whitespace, lowercase for test) ---
ENABLE_PLUGINS=$(getFlag ENABLE_PLUGINS | tr -d '[:space:]' | tr '[:upper:]' '[:lower:]')
INSTALL_PLUGIN_DEEZER=$(getFlag INSTALL_PLUGIN_DEEZER | tr -d '[:space:]' | tr '[:upper:]' '[:lower:]')
INSTALL_PLUGIN_TIDAL=$(getFlag INSTALL_PLUGIN_TIDAL | tr -d '[:space:]' | tr '[:upper:]' '[:lower:]')
INSTALL_PLUGIN_TUBIFARRY=$(getFlag INSTALL_PLUGIN_TUBIFARRY | tr -d '[:space:]' | tr '[:upper:]' '[:lower:]')

# --- Master enable check (fail fast) ---
if [[ ${ENABLE_PLUGINS} != "true" ]]; then
	log_warning "Plugins service is OFF. Set ENABLE_PLUGINS to 'true' in arrbit-config.conf."
	log_info "Log saved to $LOG_FILE"
	exit 0
fi

# --- If all plugin flags are false, warn and exit ---
if [[ ${INSTALL_PLUGIN_DEEZER} != "true" && ${INSTALL_PLUGIN_TIDAL} != "true" && ${INSTALL_PLUGIN_TUBIFARRY} != "true" ]]; then
	log_warning "Plugins service is ON, but all plugin install flags are disabled. Enable one or more plugins in arrbit-config.conf."
	log_info "Log saved to $LOG_FILE"
	exit 0
fi

# --- FUNCTIONS ---
has_dll() {
	shopt -s nullglob
	local dll_files=("$1"/*.dll)
	((${#dll_files[@]}))
}

install_plugin() { # $1 = plugin_name, $2 = plugin_dir, $3 = plugin_url
	local plugin_name="$1"
	local plugin_dir="$2"
	local plugin_url="$3"

	# Determine presence without invoking function in a conditional (avoids SC2310)
	has_dll "${plugin_dir}"
	local dll_status=$?
	if ((dll_status == 0)); then
		log_info "$plugin_name already present – skipping"
		return
	fi

	log_info "Downloading $plugin_name …"
	local tmp_dir
	tmp_dir=$(mktemp -d)

	if curl -fsSL -o "$tmp_dir/plugin.zip" "$plugin_url" >>"$LOG_FILE" 2>&1; then
		if unzip -q "$tmp_dir/plugin.zip" -d "$tmp_dir" >>"$LOG_FILE" 2>&1; then
			mkdir -p "$plugin_dir"
			find "$tmp_dir" -type f \( -iname "*.dll" -o -iname "*.json" -o -iname "*.pdb" \) -exec mv {} "$plugin_dir/" \;
			chmod -R 777 "$plugin_dir"
			log_info "$plugin_name installed"
		else
			log_warning "Failed to unzip $plugin_name – skipped"
		fi
	else
		log_warning "Failed to download $plugin_name – skipped"
	fi

	rm -rf "$tmp_dir"
}

# --- PLUGIN INSTALLATION (conditional) ---
if [[ ${INSTALL_PLUGIN_DEEZER} == "true" ]]; then
	install_plugin "Deezer" "$PLUGINS_DIR/TrevTV/Lidarr.Plugin.Deezer" \
		"https://github.com/TrevTV/Lidarr.Plugin.Deezer/releases/latest/download/Lidarr.Plugin.Deezer.net6.0.zip"
fi

if [[ ${INSTALL_PLUGIN_TIDAL} == "true" ]]; then
	install_plugin "Tidal" "$PLUGINS_DIR/TrevTV/Lidarr.Plugin.Tidal" \
		"https://github.com/TrevTV/Lidarr.Plugin.Tidal/releases/latest/download/Lidarr.Plugin.Tidal.net6.0.zip"
fi

if [[ ${INSTALL_PLUGIN_TUBIFARRY} == "true" ]]; then
	install_plugin "Tubifarry" "$PLUGINS_DIR/TypNull/Tubifarry" \
		"https://github.com/TypNull/Tubifarry/releases/download/v1.8.1.1/Tubifarry-v1.8.1.1.net6.0-develop.zip"
fi
log_info "All plugins installed"
log_info "Done"

exit 0
