#!/usr/bin/env bash
# -------------------------------------------------------------------------------------------------------------
# Arrbit - Tdarr Dependencies Setup
# Version: v1.3.2-gs3.4.0
# Purpose: Installs AI Language Detection (WhisperX 3.4.2) and MKVToolNix CLI in separate environments
# Compliance: WhisperX 3.4.2 with PyTorch 2.5.1+ ecosystem, minimal MKVToolNix CLI installation
# -------------------------------------------------------------------------------------------------------------

set -euo pipefail

# Script metadata (REQUIRED)
SCRIPT_NAME="dependencies"
SCRIPT_VERSION="v1.3.2-gs3.4.0"

# Set base path before sourcing helpers
ARRBIT_BASE="/app/arrbit"

# Source helpers (ARRBIT_BASE is now defined)
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
MKVTOOLNIX_ENV_DIR="$ARRBIT_ENVIRONMENTS_DIR/mkvtoolnix"
PYTHON_BIN="$AI_ENV_DIR/bin/python"
PIP_BIN="$AI_ENV_DIR/bin/pip"

# Check if already installed
if [[ -f "$PYTHON_BIN" ]] && [[ -f "$PIP_BIN" ]]; then
    log_info "AI Language Detection environment already exists"
    # Verify WhisperX 3.4.2 is installed
    if "$PIP_BIN" show whisperx lingua-language-detector >/dev/null 2>&1; then
        WHISPERX_VERSION=$("$PYTHON_BIN" -c "import whisperx; print(whisperx.__version__)" 2>/dev/null || echo "unknown")
        if [[ "$WHISPERX_VERSION" == "3.4.2" ]]; then
            log_info "WhisperX 3.4.2 already installed correctly"
            log_info "Dependencies installation completed (already exists)"
            exit 0
        else
            log_info "WhisperX version mismatch (found: $WHISPERX_VERSION, expected: 3.4.2) - reinstalling"
        fi
    fi
fi

log_info "Installing system dependencies"

# Update package list and install minimal system dependencies
apt-get update || { log_error "Failed to update package list"; exit 1; }

# Install only essential system packages for CPU-based WhisperX
apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    python3-dev \
    ffmpeg \
    git \
    build-essential || { 
    log_error "Failed to install system dependencies"; 
    exit 1; 
}

log_info "System dependencies installed successfully"

# Install MKVToolNix CLI tools in separate environment
log_info "Installing MKVToolNix CLI tools"
MKVTOOLNIX_ENV_DIR="$ARRBIT_ENVIRONMENTS_DIR/mkvtoolnix"

# Create MKVToolNix environment directory
mkdir -p "$MKVTOOLNIX_ENV_DIR" || { log_error "Failed to create MKVToolNix environment directory"; exit 1; }

# Install only mkvtoolnix (CLI tools only, no GUI)
apt-get install -y \
    mkvtoolnix || { 
    log_error "Failed to install MKVToolNix CLI tools"; 
    exit 1; 
}

# Verify MKVToolNix installation
log_info "Verifying MKVToolNix CLI tools installation"
if command -v mkvpropedit >/dev/null 2>&1; then
    MKVPROPEDIT_VERSION=$(mkvpropedit --version | head -n1)
    log_info "MKVToolNix CLI installed: $MKVPROPEDIT_VERSION"
    log_info "Primary tool: mkvpropedit (for metadata editing)"
    
    # Create symlinks in MKVToolNix environment for consistency
    ln -sf "$(which mkvpropedit)" "$MKVTOOLNIX_ENV_DIR/mkvpropedit" || true
    ln -sf "$(which mkvmerge)" "$MKVTOOLNIX_ENV_DIR/mkvmerge" || true
    ln -sf "$(which mkvextract)" "$MKVTOOLNIX_ENV_DIR/mkvextract" || true
    ln -sf "$(which mkvinfo)" "$MKVTOOLNIX_ENV_DIR/mkvinfo" || true
    
    log_info "MKVToolNix tools linked to: $MKVTOOLNIX_ENV_DIR"
else
    log_error "mkvpropedit not found after installation"
    exit 1
fi

# Set permissions for MKVToolNix environment
chmod -R 755 "$MKVTOOLNIX_ENV_DIR" || { log_error "Failed to set MKVToolNix permissions"; exit 1; }

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

# Install core AI packages with PyTorch 2.5.1+ CPU ecosystem
"$PIP_BIN" install --upgrade pip || { log_error "Failed to upgrade pip"; exit 1; }

# Install PyTorch 2.5.1+ CPU-only ecosystem (WhisperX 3.4.2 requirement)
"$PIP_BIN" install torch>=2.5.1+cpu torchaudio>=2.5.1+cpu --index-url https://download.pytorch.org/whl/cpu || { 
    log_error "Failed to install PyTorch 2.5.1+ CPU ecosystem"; 
    exit 1; 
}

# Create constraints file with exact WhisperX 3.4.2 requirements (no hallucinations)
cat > /tmp/constraints.txt << 'EOF'
# WhisperX 3.4.2 exact dependency requirements (from PyPI JSON API)
torch>=2.5.1
torchaudio>=2.5.1
pyannote-audio>=3.3.2
ctranslate2<4.5.0
faster-whisper>=1.1.1
nltk>=3.9.1
numpy>=2.0.2
onnxruntime>=1.19
pandas>=2.2.3
transformers>=4.48.0
# Target WhisperX version
whisperx==3.4.2
EOF

export PIP_CONSTRAINT=/tmp/constraints.txt

# Set environment variables to prevent CUDA package installation
export CUDA_VISIBLE_DEVICES=""
export TORCH_ONLY_CPU=1

# Install WhisperX 3.4.2 with all required dependencies (single installation call)
"$PIP_BIN" install \
    --constraint /tmp/constraints.txt \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    "whisperx==3.4.2" \
    "pyannote-audio>=3.3.2" \
    "ctranslate2<4.5.0" \
    "faster-whisper>=1.1.1" \
    "nltk>=3.9.1" \
    "numpy>=2.0.2" \
    "onnxruntime>=1.19" \
    "pandas>=2.2.3" \
    "transformers>=4.48.0" \
    scipy \
    librosa \
    accelerate \
    datasets || { 
    log_error "Failed to install WhisperX 3.4.2 and dependencies"; 
    exit 1; 
}

# Install Lingua language detector (with updated constraints)
"$PIP_BIN" install \
    --constraint /tmp/constraints.txt \
    lingua-language-detector || { 
    log_error "Failed to install lingua-language-detector"; 
    exit 1; 
}

log_info "All Python packages installed successfully"

# Verify installation with WhisperX 3.4.2 and all dependencies
log_info "Verifying WhisperX 3.4.2 and MKVToolNix package installation"

if ! "$PIP_BIN" show whisperx lingua-language-detector torch faster-whisper pyannote-audio >/dev/null 2>&1; then
    log_error "Python package verification failed"
    exit 1
fi

# Verify MKVToolNix CLI availability
if ! command -v mkvpropedit >/dev/null 2>&1; then
    log_error "mkvpropedit CLI tool verification failed"
    exit 1
fi

# Verify MKVToolNix environment setup
if [[ ! -L "$MKVTOOLNIX_ENV_DIR/mkvpropedit" ]]; then
    log_error "MKVToolNix environment setup verification failed"
    exit 1
fi

# Verify WhisperX version (try multiple methods)
WHISPERX_VERSION=$("$PYTHON_BIN" -c "
import whisperx
try:
    print(whisperx.__version__)
except AttributeError:
    import pkg_resources
    print(pkg_resources.get_distribution('whisperx').version)
" 2>/dev/null || echo "unknown")

log_info "WhisperX version installed: $WHISPERX_VERSION"

if [[ "$WHISPERX_VERSION" == "3.4.2" ]]; then
    log_info "WhisperX 3.4.2 installation verified successfully"
elif [[ "$WHISPERX_VERSION" != "unknown" ]]; then
    log_warning "WhisperX version found: $WHISPERX_VERSION (expected 3.4.2)"
else
    log_warning "WhisperX version could not be determined, but package is installed"
fi

# Set final permissions
chmod -R 777 "$AI_ENV_DIR" || { log_error "Failed to set final permissions"; exit 1; }

# Clean up constraints file
rm -f /tmp/constraints.txt || true

log_info "AI Language Detection and MKVToolNix dependencies installation completed successfully"
log_info "Environment paths:"
log_info "  ✓ AI Language Detection: $AI_ENV_DIR"
log_info "  ✓ MKVToolNix CLI: $MKVTOOLNIX_ENV_DIR"
log_info "Installed components:"
log_info "  ✓ WhisperX 3.4.2 with PyTorch 2.5.1+ ecosystem"
log_info "  ✓ lingua-language-detector for mixed-language detection"  
log_info "  ✓ MKVToolNix CLI (mkvpropedit) for metadata editing without transcoding"
arrbitBanner "Dependencies Setup Complete" "$SCRIPT_VERSION"
exit 0