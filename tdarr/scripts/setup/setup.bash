#!/usr/bin/env bash
# -------------------------------------------------------------------------------------------------------------
# Arrbit - Tdarr Setup
# Version: v3.2.0-gs3.2.0
# Purpose: Bootstraps Arrbit for Tdarr: downloads, installs, and initializes everything into /app/arrbit. SILENT except fatal error.
# -------------------------------------------------------------------------------------------------------------

set -euo pipefail

# --- Fixed base path and readonly variables ---
readonly ARRBIT_BASE="/app/arrbit"
readonly TMP_DIR="/app/arrbit/data/temp/arrbit_dl_$$"
readonly ZIP_URL="https://github.com/prv-ctech/Arrbit/archive/refs/heads/development.zip"
readonly REPO_MAIN="$TMP_DIR/Arrbit-development/tdarr"
readonly REPO_UNIVERSAL="$TMP_DIR/Arrbit-development/universal"

# --- Minimal bootstrap logging (no helpers yet) ---
readonly LOG_DIR="/app/arrbit/logs"
readonly LOG_FILE="$LOG_DIR/arrbit-setup-info-$(date -u +%Y_%m_%d-%H_%M).log"

log_info() { printf '%s [INFO] %s\n' "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" "$*" >>"${LOG_FILE}"; }
log_warning() { printf '%s [WARN] %s\n' "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" "$*" >>"${LOG_FILE}"; }
log_error() { printf '%s [ERROR] %s\n' "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" "$*" >>"${LOG_FILE}"; }

# --- Create full core structure ---
mkdir -p "$ARRBIT_BASE" \
         "$ARRBIT_BASE/data" \
         "$ARRBIT_BASE/data/temp" \
         "$ARRBIT_BASE/logs" \
         "$ARRBIT_BASE/scripts" \
         "$ARRBIT_BASE/scripts/setup" \
         "$ARRBIT_BASE/scripts/plugins" \
         "$ARRBIT_BASE/universal" \
         "$ARRBIT_BASE/universal/helpers" \
         "$ARRBIT_BASE/config" \
         "$ARRBIT_BASE/environments" \
         "$LOG_DIR"

# --- Set 777 permissions for all created directories ---
chmod -R 777 "$ARRBIT_BASE"

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
if [[ ! -d "$REPO_UNIVERSAL/helpers" ]]; then
    log_error "Repository structure incomplete. Missing required helpers directory."
    exit 1
fi

# --- Copy universal helpers only ---
cp -r "$REPO_UNIVERSAL/helpers" "$ARRBIT_BASE/universal/"

# --- Set 777 permissions for copied helpers ---
chmod -R 777 "$ARRBIT_BASE/universal/helpers"

# --- Copy Tdarr-specific plugins ---
mkdir -p "$ARRBIT_BASE/scripts/plugins"
cp -rf "$REPO_MAIN/scripts/plugins/." "$ARRBIT_BASE/scripts/plugins/"

# --- Set 777 permissions for copied plugins ---
chmod -R 777 "$ARRBIT_BASE/scripts/plugins"

# --- Copy setup scripts ---
cp -f "$REPO_MAIN/scripts/setup/dependencies.bash" "$ARRBIT_BASE/scripts/setup/"
cp -f "$REPO_MAIN/scripts/setup/run" "$ARRBIT_BASE/scripts/setup/"

# --- Set 777 permissions for copied setup scripts ---
chmod -R 777 "$ARRBIT_BASE/scripts/setup"

# --- Copy data files ---
mkdir -p "$ARRBIT_BASE/data"
if [[ -d "$REPO_MAIN/data" ]]; then
    cp -rf "$REPO_MAIN/data/." "$ARRBIT_BASE/data/"
fi

# --- Set 777 permissions for copied data files ---
chmod -R 777 "$ARRBIT_BASE/data"

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

# --- Set 777 permissions for copied config files ---
chmod -R 777 "$ARRBIT_BASE/config"

# --- Create Tdarr-specific directories ---
mkdir -p "$ARRBIT_BASE/environments/tdarr"

# --- Cleanup temporary files ---
rm -rf "$TMP_DIR"

log_info "Tdarr setup completed successfully"
exit 0