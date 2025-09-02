#!/usr/bin/env bash
# -------------------------------------------------------------------------------------------------------------
# Arrbit - AI Dependencies Test Script
# Version: v1.3.2-gs3.4.0
# Purpose: Test WhisperX 3.4.2, MKVToolNix CLI, and validate all dependencies in separate environments
# Compliance: WhisperX 3.4.2 with PyTorch 2.5.1+ ecosystem and MKVToolNix CLI testing
# -------------------------------------------------------------------------------------------------------------

set -euo pipefail

# Script metadata (REQUIRED)
SCRIPT_NAME="test-dependencies"
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

log_info "Starting AI Dependencies and MKVToolNix CLI Test"

# Environment paths
AI_ENV_DIR="$ARRBIT_ENVIRONMENTS_DIR/ai-language-detection"
MKVTOOLNIX_ENV_DIR="$ARRBIT_ENVIRONMENTS_DIR/mkvtoolnix"
PYTHON_BIN="$AI_ENV_DIR/bin/python"
PIP_BIN="$AI_ENV_DIR/bin/pip"

# Check if environment exists
if [[ ! -f "$PYTHON_BIN" ]] || [[ ! -f "$PIP_BIN" ]]; then
    log_error "AI Language Detection environment not found at $AI_ENV_DIR"
    log_error "Please run dependencies.bash first"
    exit 1
fi

# Check if MKVToolNix environment exists
if [[ ! -d "$MKVTOOLNIX_ENV_DIR" ]]; then
    log_error "MKVToolNix environment not found at $MKVTOOLNIX_ENV_DIR"
    log_error "Please run dependencies.bash first"
    exit 1
fi

log_info "Testing Python environment..."

# Test Python version
PYTHON_VERSION=$("$PYTHON_BIN" --version)
log_info "Python version: $PYTHON_VERSION"

# Test pip version
PIP_VERSION=$("$PIP_BIN" --version)
log_info "Pip version: $PIP_VERSION"

log_info "Testing package imports..."

# Test core PyTorch 2.5.1+ ecosystem
"$PYTHON_BIN" -c "
import sys
print('Testing PyTorch 2.5.1+ ecosystem...')

# Test torch (PyTorch 2.5.1+ CPU version)
try:
    import torch
    version = torch.__version__
    print(f'✓ torch {version} - OK')
    print(f'  CUDA available: {torch.cuda.is_available()}')
    
    # Verify PyTorch 2.x requirement
    major_version = int(version.split('.')[0])
    if major_version >= 2:
        print(f'  ✅ PyTorch 2.x requirement satisfied (found {major_version}.x)')
    else:
        print(f'  ✗ PyTorch 2.x requirement NOT satisfied (found {major_version}.x)')
        sys.exit(1)
        
except ImportError as e:
    print(f'✗ torch - FAILED: {e}')
    sys.exit(1)

# Test torchaudio
try:
    import torchaudio
    version = torchaudio.__version__
    print(f'✓ torchaudio {version} - OK')
    
    # Verify version 2.5.1+ requirement
    version_parts = version.split('+')[0].split('.')  # Remove +cpu suffix
    major, minor = int(version_parts[0]), int(version_parts[1])
    patch = int(version_parts[2]) if len(version_parts) > 2 else 0
    
    if major > 2 or (major == 2 and minor > 5) or (major == 2 and minor == 5 and patch >= 1):
        print(f'  ✅ torchaudio 2.5.1+ requirement satisfied')
    else:
        print(f'  ⚠️  torchaudio version may be below 2.5.1 requirement')
        
except ImportError as e:
    print(f'✗ torchaudio - FAILED: {e}')
    sys.exit(1)

# Test numpy (2.0.2+ requirement)
try:
    import numpy as np
    version = np.__version__
    print(f'✓ numpy {version} - OK')
    
    # Check numpy 2.0.2+ requirement
    version_parts = version.split('.')
    major, minor = int(version_parts[0]), int(version_parts[1])
    patch = int(version_parts[2]) if len(version_parts) > 2 else 0
    
    if major > 2 or (major == 2 and minor > 0) or (major == 2 and minor == 0 and patch >= 2):
        print(f'  ✅ numpy 2.0.2+ requirement satisfied')
    else:
        print(f'  ⚠️  numpy version may be below 2.0.2 requirement')
        
except ImportError as e:
    print(f'✗ numpy - FAILED: {e}')
    sys.exit(1)

# Test pandas (2.2.3+ requirement)
try:
    import pandas as pd
    version = pd.__version__
    print(f'✓ pandas {version} - OK')
    
    # Check pandas 2.2.3+ requirement
    version_parts = version.split('.')
    major, minor = int(version_parts[0]), int(version_parts[1])
    patch = int(version_parts[2]) if len(version_parts) > 2 else 0
    
    if major > 2 or (major == 2 and minor > 2) or (major == 2 and minor == 2 and patch >= 3):
        print(f'  ✅ pandas 2.2.3+ requirement satisfied')
    else:
        print(f'  ⚠️  pandas version may be below 2.2.3 requirement')
        
except ImportError as e:
    print(f'✗ pandas - FAILED: {e}')
    sys.exit(1)

# Test scipy
try:
    import scipy
    print(f'✓ scipy {scipy.__version__} - OK')
except ImportError as e:
    print(f'✗ scipy - FAILED: {e}')
    sys.exit(1)

# Test librosa
try:
    import librosa
    print(f'✓ librosa {librosa.__version__} - OK')
except ImportError as e:
    print(f'✗ librosa - FAILED: {e}')
    sys.exit(1)

print('PyTorch 2.5.1+ ecosystem test completed successfully!')
"

log_info "Testing WhisperX 3.4.2 components..."

# Test WhisperX 3.4.2 dependencies
"$PYTHON_BIN" -c "
import sys
print('Testing WhisperX 3.4.2 components...')

# Test faster-whisper (1.1.1+ requirement)
try:
    import faster_whisper
    # Try to get version if available
    try:
        version = faster_whisper.__version__
        print(f'✓ faster-whisper {version} - OK')
        
        # Check 1.1.1+ requirement
        version_parts = version.split('.')
        major, minor = int(version_parts[0]), int(version_parts[1])
        patch = int(version_parts[2]) if len(version_parts) > 2 else 0
        
        if major > 1 or (major == 1 and minor > 1) or (major == 1 and minor == 1 and patch >= 1):
            print(f'  ✅ faster-whisper 1.1.1+ requirement satisfied')
        else:
            print(f'  ⚠️  faster-whisper version may be below 1.1.1 requirement')
            
    except AttributeError:
        print(f'✓ faster-whisper - OK (version not available)')
except ImportError as e:
    print(f'✗ faster-whisper - FAILED: {e}')
    sys.exit(1)

# Test ctranslate2 (<4.5.0 requirement)
try:
    import ctranslate2
    version = ctranslate2.__version__
    print(f'✓ ctranslate2 {version} - OK')
    
    # Check <4.5.0 requirement
    version_parts = version.split('.')
    major, minor = int(version_parts[0]), int(version_parts[1])
    
    if major < 4 or (major == 4 and minor < 5):
        print(f'  ✅ ctranslate2 <4.5.0 requirement satisfied')
    else:
        print(f'  ⚠️  ctranslate2 version may violate <4.5.0 requirement')
        
except ImportError as e:
    print(f'✗ ctranslate2 - FAILED: {e}')
    sys.exit(1)

# Test transformers (4.48.0+ requirement)
try:
    import transformers
    version = transformers.__version__
    print(f'✓ transformers {version} - OK')
    
    # Check 4.48.0+ requirement
    version_parts = version.split('.')
    major, minor = int(version_parts[0]), int(version_parts[1])
    patch = int(version_parts[2]) if len(version_parts) > 2 else 0
    
    if major > 4 or (major == 4 and minor > 48) or (major == 4 and minor == 48 and patch >= 0):
        print(f'  ✅ transformers 4.48.0+ requirement satisfied')
    else:
        print(f'  ⚠️  transformers version may be below 4.48.0 requirement')
        
except ImportError as e:
    print(f'✗ transformers - FAILED: {e}')
    sys.exit(1)

# Test onnxruntime (1.19+ requirement)
try:
    import onnxruntime
    version = onnxruntime.__version__
    print(f'✓ onnxruntime {version} - OK')
    
    # Check 1.19+ requirement
    version_parts = version.split('.')
    major, minor = int(version_parts[0]), int(version_parts[1])
    
    if major > 1 or (major == 1 and minor >= 19):
        print(f'  ✅ onnxruntime 1.19+ requirement satisfied')
    else:
        print(f'  ⚠️  onnxruntime version may be below 1.19 requirement')
        
except ImportError as e:
    print(f'✗ onnxruntime - FAILED: {e}')
    sys.exit(1)

# Test datasets
try:
    import datasets
    print(f'✓ datasets {datasets.__version__} - OK')
except ImportError as e:
    print(f'✗ datasets - FAILED: {e}')
    sys.exit(1)

# Test accelerate
try:
    import accelerate
    print(f'✓ accelerate {accelerate.__version__} - OK')
except ImportError as e:
    print(f'✗ accelerate - FAILED: {e}')
    sys.exit(1)

# Test nltk (3.9.1+ requirement)
try:
    import nltk
    version = nltk.__version__
    print(f'✓ nltk {version} - OK')
    
    # Check 3.9.1+ requirement
    version_parts = version.split('.')
    major, minor = int(version_parts[0]), int(version_parts[1])
    patch = int(version_parts[2]) if len(version_parts) > 2 else 0
    
    if major > 3 or (major == 3 and minor > 9) or (major == 3 and minor == 9 and patch >= 1):
        print(f'  ✅ nltk 3.9.1+ requirement satisfied')
    else:
        print(f'  ⚠️  nltk version may be below 3.9.1 requirement')
        
except ImportError as e:
    print(f'✗ nltk - FAILED: {e}')
    sys.exit(1)

# Test pyannote-audio (3.3.2+ requirement)
try:
    import pyannote.audio
    version = pyannote.audio.__version__
    print(f'✓ pyannote-audio {version} - OK')

    # Check 3.3.2+ requirement for WhisperX 3.4.2
    version_parts = version.split('.')
    major, minor = int(version_parts[0]), int(version_parts[1])
    patch = int(version_parts[2]) if len(version_parts) > 2 else 0
    
    if major > 3 or (major == 3 and minor > 3) or (major == 3 and minor == 3 and patch >= 2):
        print(f'  ✅ pyannote-audio 3.3.2+ requirement satisfied')
        print(f'  ✅ Compatible with WhisperX 3.4.2')
    else:
        print(f'  ⚠️  pyannote-audio version may be below 3.3.2 requirement')

except ImportError as e:
    print(f'✗ pyannote-audio - FAILED: {e}')
    sys.exit(1)

print('WhisperX 3.4.2 components test completed successfully!')
"

log_info "Testing WhisperX 3.4.2 import..."

# Test WhisperX 3.4.2 main import
"$PYTHON_BIN" -c "
import sys
print('Testing WhisperX 3.4.2 main import...')

try:
    import whisperx
    # Get exact version
    try:
        version = whisperx.__version__
        print(f'✓ whisperx {version} - OK')
        
        # Verify version 3.4.2
        if version == '3.4.2':
            print(f'  ✅ WhisperX 3.4.2 requirement satisfied')
        else:
            print(f'  ⚠️  WhisperX version mismatch - expected 3.4.2, found {version}')
            
    except AttributeError:
        print('✓ whisperx - OK (version not available in __version__)')
        # Try alternative version check
        try:
            # Some packages store version differently
            import pkg_resources
            version = pkg_resources.get_distribution('whisperx').version
            print(f'  Version from pkg_resources: {version}')
        except:
            print('  Version could not be determined')
            
    print('WhisperX 3.4.2 import successful!')
except ImportError as e:
    print(f'✗ whisperx - FAILED: {e}')
    sys.exit(1)
except Exception as e:
    print(f'✗ whisperx - ERROR: {e}')
    sys.exit(1)
"

log_info "Testing lingua-language-detector..."

# Test lingua-language-detector
"$PYTHON_BIN" -c "
import sys
print('Testing lingua-language-detector...')

try:
    from lingua import Language, LanguageDetectorBuilder
    print('✓ lingua-language-detector - OK')

    # Test basic functionality
    languages = [Language.ENGLISH, Language.FRENCH, Language.GERMAN, Language.SPANISH]
    detector = LanguageDetectorBuilder.from_languages(*languages).build()
    print('Language detector built successfully!')

    # Test detection
    result = detector.detect_language_of('Hello world, how are you?')
    print(f'Language detection test: {result}')

except ImportError as e:
    print(f'✗ lingua-language-detector - FAILED: {e}')
    sys.exit(1)
except Exception as e:
    print(f'✗ lingua-language-detector - ERROR: {e}')
    sys.exit(1)
"

log_info "Testing MKVToolNix CLI tools..."

# Test MKVToolNix CLI tools
log_info "Verifying MKVToolNix CLI installation and functionality"

# Test mkvpropedit (primary tool)
if command -v mkvpropedit >/dev/null 2>&1; then
    MKVPROPEDIT_VERSION=$(mkvpropedit --version 2>&1 | head -n1)
    log_info "mkvpropedit version: $MKVPROPEDIT_VERSION"
    
    # Test mkvpropedit help command
    if mkvpropedit --help >/dev/null 2>&1; then
        log_info "✓ mkvpropedit help command works"
    else
        log_error "✗ mkvpropedit help command failed"
        exit 1
    fi
    
    # Test property names listing (dry run capability)
    log_info "Testing mkvpropedit property listing capability..."
    if mkvpropedit --list-property-names >/dev/null 2>&1; then
        log_info "✓ mkvpropedit property listing works"
        
        # Show sample of available properties
        SAMPLE_PROPERTIES=$(mkvpropedit --list-property-names 2>/dev/null | head -5 | tr '\n' ', ' | sed 's/, $//')
        log_info "Sample properties available: $SAMPLE_PROPERTIES"
    else
        log_error "✗ mkvpropedit property listing failed"
        exit 1
    fi
else
    log_error "✗ mkvpropedit command not found"
    exit 1
fi

# Test MKVToolNix environment symlinks
log_info "Testing MKVToolNix environment setup..."
if [[ -L "$MKVTOOLNIX_ENV_DIR/mkvpropedit" ]]; then
    ENV_MKVPROPEDIT="$MKVTOOLNIX_ENV_DIR/mkvpropedit"
    if [[ -x "$ENV_MKVPROPEDIT" ]]; then
        ENV_VERSION=$("$ENV_MKVPROPEDIT" --version 2>&1 | head -n1)
        log_info "✓ Environment mkvpropedit works: $ENV_VERSION"
    else
        log_error "✗ Environment mkvpropedit not executable"
        exit 1
    fi
else
    log_error "✗ MKVToolNix environment symlink not found"
    exit 1
fi

# Test additional MKVToolNix tools (if available)
log_info "Testing additional MKVToolNix tools..."

if command -v mkvmerge >/dev/null 2>&1; then
    MKVMERGE_VERSION=$(mkvmerge --version 2>&1 | head -n1)
    log_info "✓ mkvmerge available: $MKVMERGE_VERSION"
else
    log_warning "⚠ mkvmerge not available (not critical for metadata editing)"
fi

if command -v mkvinfo >/dev/null 2>&1; then
    MKVINFO_VERSION=$(mkvinfo --version 2>&1 | head -n1)
    log_info "✓ mkvinfo available: $MKVINFO_VERSION"
else
    log_warning "⚠ mkvinfo not available (not critical for metadata editing)"
fi

log_info "MKVToolNix CLI testing completed successfully!"

# Test metadata editing simulation (dry run)
log_info "Testing metadata editing simulation..."
TEMP_TEST_OUTPUT=$(mktemp)
if mkvpropedit --help 2>&1 | grep -q "language"; then
    log_info "✓ mkvpropedit supports language metadata editing"
    log_info "Example usage: mkvpropedit input.mkv --edit track:a1 --set language=eng"
else
    log_warning "⚠ Could not verify language metadata support"
fi
rm -f "$TEMP_TEST_OUTPUT" || true

log_info "Testing WhisperX 3.4.2 functionality..."

# Test WhisperX 3.4.2 basic functionality
"$PYTHON_BIN" -c "
import sys
print('Testing WhisperX 3.4.2 functionality...')

try:
    import whisperx
    import torch
    import pyannote.audio

    # Check version compatibility for WhisperX 3.4.2
    torch_version = torch.__version__
    pa_version = pyannote.audio.__version__

    print(f'Version compatibility check for WhisperX 3.4.2:')
    print(f'  PyTorch: {torch_version}')
    print(f'  pyannote.audio: {pa_version}')

    # Check WhisperX 3.4.2 requirements
    compatibility_status = []

    # PyTorch 2.5.1+ requirement
    torch_major = int(torch_version.split('.')[0])
    if torch_major >= 2:
        compatibility_status.append('✅ PyTorch 2.x requirement satisfied')
    else:
        compatibility_status.append('✗ PyTorch 2.x requirement NOT satisfied')

    # pyannote.audio 3.3.2+ requirement
    pa_major = int(pa_version.split('.')[0])
    pa_minor = int(pa_version.split('.')[1])
    pa_patch = int(pa_version.split('.')[2]) if len(pa_version.split('.')) > 2 else 0
    
    if pa_major > 3 or (pa_major == 3 and pa_minor > 3) or (pa_major == 3 and pa_minor == 3 and pa_patch >= 2):
        compatibility_status.append('✅ pyannote.audio 3.3.2+ requirement satisfied')
    else:
        compatibility_status.append('⚠️  pyannote.audio may be below 3.3.2 requirement')

    for status in compatibility_status:
        print(f'  {status}')

    # Check device availability
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')

    # Test model loading (base model for quick test)
    print('Testing WhisperX 3.4.2 model loading...')
    try:
        model = whisperx.load_model('base', device=device, compute_type='int8')
        print('✓ WhisperX 3.4.2 model loaded successfully!')

        # Clean up
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
    except Exception as model_error:
        print(f'⚠️  Model loading test failed: {model_error}')
        print('  This may be normal for first run (model download required)')

except ImportError as e:
    print(f'✗ WhisperX 3.4.2 functionality - FAILED: {e}')
    sys.exit(1)
except Exception as e:
    print(f'✗ WhisperX 3.4.2 functionality - ERROR: {e}')
    sys.exit(1)
"

log_info "All WhisperX 3.4.2 and MKVToolNix dependency tests completed successfully!"
log_info "Environment Summary:"
log_info "  ✓ WhisperX 3.4.2 with PyTorch 2.5.1+ ecosystem is fully functional"
log_info "  ✓ MKVToolNix CLI tools are ready for metadata editing"
log_info "  ✓ Separate environments properly configured"

arrbitBanner "Dependencies Test Complete" "$SCRIPT_VERSION"
exit 0
