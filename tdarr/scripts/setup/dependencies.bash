#!/usr/bin/env bash
# -------------------------------------------------------------------------------------------------------------
# Arrbit - Tdarr Dependencies Setup
# Version: v1.0.0-gs3.2.1
# Purpose: Installs AI Language Detection dependencies in isolated Python environment
# -------------------------------------------------------------------------------------------------------------

set -euo pipefail

# Script metadata (REQUIRED)
SCRIPT_NAME="dependencies"
SCRIPT_VERSION="v1.0.0-gs3.2.1"

# Source helpers (fixed base path model)
ARRBIT_BASE="/app/arrbit"
source "$ARRBIT_BASE/universal/helpers/logging_utils.bash"
source "$ARRBIT_BASE/universal/helpers/helpers.bash"
arrbitPurgeOldLogs # Always run first (retains newest 3 per script prefix)

# Initialize logging
LOG_FILE="$ARRBIT_LOGS_DIR/arrbit-${SCRIPT_NAME}-$(date +%Y_%m_%d-%H_%M).log"
arrbitInitLog "$LOG_FILE" || exit 1
arrbitBanner "$SCRIPT_NAME" "$SCRIPT_VERSION"

log_info "Starting AI Language Detection dependencies installation"

# Environment paths
AI_ENV_DIR="$ARRBIT_ENVIRONMENTS_DIR/ai-language-detection"
PYTHON_BIN="$AI_ENV_DIR/bin/python"
PIP_BIN="$AI_ENV_DIR/bin/pip"

# Check if already installed
if [[ -f "$PYTHON_BIN" ]] && [[ -f "$PIP_BIN" ]]; then
    log_info "AI Language Detection environment already exists"
    # Verify key packages are installed
    if "$PIP_BIN" show whisperx lingua-language-detector >/dev/null 2>&1; then
        log_info "All required packages already installed"
        log_info "Dependencies installation completed (already exists)"
        exit 0
    fi
fi

log_info "Installing system dependencies"

# Update package list and install system dependencies
apt-get update || { log_error "Failed to update package list"; exit 1; }

# Install Python 3, pip, venv, and FFmpeg
apt-get install -y python3 python3-pip python3-venv ffmpeg || { 
    log_error "Failed to install system dependencies"; 
    exit 1; 
}

log_info "System dependencies installed successfully"

# Create isolated Python environment
log_info "Creating isolated Python environment at $AI_ENV_DIR"
mkdir -p "$AI_ENV_DIR" || { log_error "Failed to create environment directory"; exit 1; }

python3 -m venv "$AI_ENV_DIR" || { 
    log_error "Failed to create Python virtual environment"; 
    exit 1; 
}

log_info "Python virtual environment created successfully"

# Set 777 permissions for environment directory
chmod -R 777 "$AI_ENV_DIR" || { log_error "Failed to set permissions"; exit 1; }

log_info "Installing Python packages in isolated environment"

# Install core AI packages
"$PIP_BIN" install --upgrade pip || { log_error "Failed to upgrade pip"; exit 1; }

# Install WhisperX and dependencies
"$PIP_BIN" install whisperx || { log_error "Failed to install WhisperX"; exit 1; }

# Install Lingua language detector
"$PIP_BIN" install lingua-language-detector || { 
    log_error "Failed to install lingua-language-detector"; 
    exit 1; 
}

# Install faster-whisper for CPU optimization
"$PIP_BIN" install faster-whisper || { 
    log_error "Failed to install faster-whisper"; 
    exit 1; 
}

log_info "All Python packages installed successfully"

# Verify installation
log_info "Verifying package installation"

if ! "$PIP_BIN" show whisperx lingua-language-detector faster-whisper >/dev/null 2>&1; then
    log_error "Package verification failed"
    exit 1
fi

log_info "Package verification completed successfully"

# Set final permissions
chmod -R 777 "$AI_ENV_DIR" || { log_error "Failed to set final permissions"; exit 1; }

log_info "AI Language Detection dependencies installation completed successfully"
arrbitBanner "Dependencies Setup Complete" "$SCRIPT_VERSION"
exit 0