#!/usr/bin/env bash
# -------------------------------------------------------------------------------------------------------------
# Arrbit - setup
# Version: v3.2.0-gs3.2.0
# Purpose: Bootstraps Arrbit: downloads, installs, and initializes everything into /app/arrbit. SILENT except fatal error.
# -------------------------------------------------------------------------------------------------------------

set -euo pipefail

# --- Fixed base path and variables ---
ARRBIT_BASE="/app/arrbit"
TMP_DIR="/app/arrbit/data/temp/arrbit_dl_$$"
ZIP_URL="https://github.com/prvctech/Arrbit/archive/refs/heads/main.zip"
REPO_MAIN="$TMP_DIR/Arrbit-main/lidarr"
REPO_UNIVERSAL="$TMP_DIR/Arrbit-main/universal"

# --- Minimal bootstrap logging (no helpers yet) ---
LOG_DIR="/app/arrbit/logs"
LOG_FILE="$LOG_DIR/arrbit-setup-info-$(date -u +%Y_%m_%d-%H_%M).log"

log_info() { printf '%s [INFO] %s\n' "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" "$*" >>"${LOG_FILE}"; }
log_warning() { printf '%s [WARN] %s\n' "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" "$*" >>"${LOG_FILE}"; }
log_error() { printf '%s [ERROR] %s\n' "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" "$*" >>"${LOG_FILE}"; }

# --- Ensure Arrbit base and tmp dir exist ---
mkdir -p "$ARRBIT_BASE" "$ARRBIT_BASE/data/temp" "$LOG_DIR"
mkdir -p "$TMP_DIR"

# --- Download and extract repo with validation ---
cd "$TMP_DIR"
if ! curl -fsSL "$ZIP_URL" -o arrbit.zip; then
    log_error "Failed to download repository. Check network and URL."
    exit 1
fi

# --- Validate ZIP integrity ---
if ! unzip -tq arrbit.zip >/dev/null 2>&1; then
    log_error "Downloaded ZIP file is corrupted or invalid."
    exit 1
fi

unzip -qqo arrbit.zip

# --- Verify repository structure ---
if [[ ! -d "$REPO_UNIVERSAL/helpers" ]] || [[ ! -d "$REPO_UNIVERSAL/connectors" ]]; then
    log_error "Repository structure incomplete. Missing required directories."
    exit 1
fi

# --- Copy helpers and connectors from universal ---
cp -r "$REPO_UNIVERSAL/helpers" "$ARRBIT_BASE/universal/"
cp -r "$REPO_UNIVERSAL/connectors" "$ARRBIT_BASE/universal/"

# --- Switch to Golden Standard logging ---
HELPERS_DIR="$ARRBIT_BASE/universal/helpers"
source "$HELPERS_DIR/logging_utils.bash"
source "$HELPERS_DIR/helpers.bash"
arrbitPurgeOldLogs 3

# --- Copy modules, services, and data ---
mkdir -p "$ARRBIT_BASE/scripts/modules"
cp -rf "$REPO_MAIN/process_scripts/modules/." "$ARRBIT_BASE/scripts/modules/"

mkdir -p "$ARRBIT_BASE/data"
cp -rf "$REPO_MAIN/data/." "$ARRBIT_BASE/data/"

mkdir -p "$ARRBIT_BASE/scripts/services"
cp -rf "$REPO_MAIN/process_scripts/services/." "$ARRBIT_BASE/scripts/services/"

# --- Copy custom process scripts if they exist ---
if [[ -d "$REPO_MAIN/process_scripts/custom" ]]; then
    cp -rf "$REPO_MAIN/process_scripts/custom" "$ARRBIT_BASE/scripts/"
fi

# --- Copy setup scripts except setup.bash and run ---
mkdir -p "$ARRBIT_BASE/scripts/setup"
find "$REPO_MAIN/setup" -type f ! -name "setup.bash" ! -name "run" -exec cp -f {} "$ARRBIT_BASE/scripts/setup/" \;

# --- Ensure config directory exists ---
mkdir -p "$ARRBIT_BASE/config"

# --- Copy each config file ONLY if it does NOT already exist ---
for src_file in "$REPO_MAIN/config/"*; do
    if [[ -f "$src_file" ]]; then
        filename="$(basename "$src_file")"
        dest_file="$ARRBIT_BASE/config/$filename"
        if [[ ! -f "$dest_file" ]]; then
            cp -f "$src_file" "$dest_file"
        fi
    fi
done

# --- Create environments directory ---
mkdir -p "$ARRBIT_BASE/environments"

# --- Cleanup temporary files ---
rm -rf "$TMP_DIR"

log_info "Setup completed successfully"
exit 0
