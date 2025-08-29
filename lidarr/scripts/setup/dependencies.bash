#!/usr/bin/env bash
# -------------------------------------------------------------------------------------------------------------
# Arrbit - dependencies.bash
# Version: v1.0.15-gs2.8.2
# Purpose: Silent dependency installer for Arrbit (Golden Standard v2.8.2 compliant)
# -------------------------------------------------------------------------------------------------------------

source /config/arrbit/helpers/logging_utils.bash
source /config/arrbit/helpers/helpers.bash
arrbitPurgeOldLogs

SCRIPT_NAME="dependencies"
SCRIPT_VERSION="v1.0.15-gs2.8.2"
LOG_FILE="/config/logs/arrbit-${SCRIPT_NAME}-$(date +%Y_%m_%d-%H_%M).log"

mkdir -p /config/logs && touch "$LOG_FILE" && chmod 777 "$LOG_FILE"

# --- Setup version tracking (prevent re-installation) ---
SETUP_VERSION_FILE="/config/setup_version.txt"
CURRENT_VERSION="1.0.15"

# Virtual env (to isolate tidal-dl-ng/python-ffmpeg from ffmpeg-python)
# Location moved per user preference
ARRBIT_VENV="/config/arrbit/custom/tidal-dl-ng"
VENV_PY="$ARRBIT_VENV/bin/python3"
VENV_PIP="$ARRBIT_VENV/bin/pip"
mkdir -p "$(dirname "$ARRBIT_VENV")" >>"$LOG_FILE" 2>&1 || true

# --- What we consider required (tools + python modules) ---
REQUIRED_CMDS="beet atomicparsley python3 uv eyeD3 yq xq jq ffmpeg \
vorbiscomment metaflac opustags aria2c rg convert tidy lame"
REQUIRED_PY_MODS=(
  eyed3 yq mutagen beautifulsoup4 jellyfish pyacoustid requests
  yt_dlp pyxDamerauLevenshtein colorama pylast r128gain
  cryptography requests_oauthlib plexapi
)

list_missing_cmds() {
  local missing=""
  for cmd in $REQUIRED_CMDS; do
    if ! command -v "$cmd" >/dev/null 2>&1; then
      missing+=" $cmd"
    fi
  done
  echo "$missing"
}

list_missing_py() {
  if ! command -v python3 >/dev/null 2>&1; then
    # python3 not available yet; assume all required Python modules are missing
    printf '%s ' ${REQUIRED_PY_MODS[@]}
    echo
    return 0
  fi
  python3 - <<'PY'
name_map = {
  "beautifulsoup4": "bs4",
  "pyacoustid": "acoustid",
  "yt_dlp": "yt_dlp",
  "pyxDamerauLevenshtein": "pyxdameraulevenshtein",
  "requests_oauthlib": "requests_oauthlib",
  "tidal-dl-ng": "tidal_dl_ng",
}
mods = [
  "eyed3","yq","mutagen","beautifulsoup4","jellyfish","pyacoustid","requests",
  "yt_dlp","pyxDamerauLevenshtein","colorama","pylast","r128gain",
  "cryptography","requests_oauthlib","plexapi"
]
missing = []
for pkg in mods:
    mod = name_map.get(pkg, pkg.replace("-","_").replace(".","_"))
    try:
        __import__(mod)
    except Exception:
        missing.append(pkg)
print(" ".join(missing))
PY
}

# Early-skip logic
if [[ -f "$SETUP_VERSION_FILE" ]]; then
  source "$SETUP_VERSION_FILE"
  if [[ "$CURRENT_VERSION" == "${setupversion:-}" ]]; then
    m_cmds="$(list_missing_cmds)"; m_py="$(list_missing_py)"
    if [[ -z "$m_cmds" && -z "$m_py" && -x "$VENV_PY" ]]; then
      printf '[Arrbit] Dependencies already installed. Skipping.\n' | arrbitLogClean >> "$LOG_FILE"
      exit 0
    fi
  fi
fi

# --- Install missing dependencies (silent, log-only) ---
missing_cmds="$(list_missing_cmds)"
missing_py_mods="$(list_missing_py)"

if [[ -z "$missing_cmds" && -z "$missing_py_mods" && -x "$VENV_PY" ]]; then
  echo "setupversion=$CURRENT_VERSION" > "$SETUP_VERSION_FILE"
  exit 0
fi

printf '[Arrbit] Installing missing commands: %s\n' "${missing_cmds:-none}" | arrbitLogClean >> "$LOG_FILE"
printf '[Arrbit] Installing missing python modules: %s\n' "${missing_py_mods:-none}" | arrbitLogClean >> "$LOG_FILE"

# If nothing is missing, mark version and exit quickly to avoid re-downloading
if [[ -z "$missing_cmds" && -z "$missing_py_mods" && -x "$VENV_PY" ]]; then
  echo "setupversion=$CURRENT_VERSION" > "$SETUP_VERSION_FILE"
  printf '[Arrbit] Dependencies installation completed successfully.\n' | arrbitLogClean >> "$LOG_FILE"
  exit 0
fi

# System-level packages
apk add --no-cache uv >>"$LOG_FILE" 2>&1
apk add --no-cache \
  build-base \
  python3 python3-dev py3-pip py3-virtualenv \
  musl-dev libffi-dev openssl-dev \
  cargo rust \
  cmake >>"$LOG_FILE" 2>&1 || true
apk add --no-cache \
  tidyhtml musl-locales musl-locales-lang \
  flac jq git ffmpeg imagemagick \
  opus-tools opustags vorbis-tools \
  parallel npm ripgrep lame aria2 >>"$LOG_FILE" 2>&1
apk add --no-cache -X http://dl-cdn.alpinelinux.org/alpine/edge/community beets >>"$LOG_FILE" 2>&1
apk add --no-cache -X http://dl-cdn.alpinelinux.org/alpine/edge/testing atomicparsley faac >>"$LOG_FILE" 2>&1 || true
apk add --no-cache chromaprint >>"$LOG_FILE" 2>&1 || true

# Python system site-packages (respect PEP 668 with override)
python3 -m pip install -U --no-cache-dir --break-system-packages \
  "Cython<3" setuptools wheel >>"$LOG_FILE" 2>&1
python3 -m pip install -U --no-cache-dir --break-system-packages \
  eyed3 yq mutagen beautifulsoup4 jellyfish requests \
  yt-dlp pyxDamerauLevenshtein colorama \
  pylast r128gain cryptography requests-oauthlib \
  plexapi pyacoustid >>"$LOG_FILE" 2>&1

# Create isolated venv for tidal-dl-ng + python-ffmpeg to avoid ffmpeg-python conflict
if [[ ! -x "$VENV_PY" ]]; then
  python3 -m venv "$ARRBIT_VENV" >>"$LOG_FILE" 2>&1 || virtualenv "$ARRBIT_VENV" >>"$LOG_FILE" 2>&1 || true
fi
if [[ -x "$VENV_PIP" ]]; then
  "$VENV_PIP" install -U --no-cache-dir \
    tidal-dl-ng python-ffmpeg >>"$LOG_FILE" 2>&1 || true
  # Wrappers to enforce venv runtime using the console script inside the venv
  cat >/usr/local/bin/tidal-dl-ng <<'SH'
#!/bin/sh
exec "/config/arrbit/custom/tidal-dl-ng/bin/tdn" "$@"
SH
  chmod +x /usr/local/bin/tidal-dl-ng
  cat >/usr/local/bin/tdn <<'SH'
#!/bin/sh
exec "/config/arrbit/custom/tidal-dl-ng/bin/tdn" "$@"
SH
  chmod +x /usr/local/bin/tdn
  # Refresh shell hash for current session
  hash -r 2>/dev/null || true
fi

# Verify ffmpeg wrapper in venv (not system)
ffmpeg_py_missing=""
if [[ -x "$VENV_PY" ]]; then
  "$VENV_PY" - <<'PY' >>"$LOG_FILE" 2>&1
try:
    from ffmpeg import FFmpeg
    print("[Arrbit] venv python-ffmpeg wrapper OK")
except Exception as e:
    print("[Arrbit] venv python-ffmpeg wrapper FAIL:", e)
    raise
PY
  if [[ $? -ne 0 ]]; then ffmpeg_py_missing="venv python-ffmpeg"; fi
else
  ffmpeg_py_missing="venv missing"
fi

# --- Post-install verification ---
missing_after_cmds="$(list_missing_cmds)"
missing_after_py="$(list_missing_py)"
if [[ -n "$ffmpeg_py_missing" ]]; then
  missing_after_py="$missing_after_py $ffmpeg_py_missing"
fi

if [[ -n "$missing_after_cmds" || -n "$missing_after_py" ]]; then
  log_error "Failed to install required dependencies: cmds:(${missing_after_cmds:-none}) py:(${missing_after_py:-none}) (see log at /config/logs)"
  cat <<EOF | arrbitLogClean >> "$LOG_FILE"
[Arrbit] ERROR Failed to install required dependencies
[WHY]: One or more packages failed to install or are not available in repositories/PyPI
[FIX]: See details below, verify repository availability, or manually install missing items
[Missing Commands] ${missing_after_cmds:-none}
[Missing Python Modules] ${missing_after_py:-none}
[Installation Log]
$(cat "$LOG_FILE")
[/Installation Log]
EOF
  exit 1
fi

echo "setupversion=$CURRENT_VERSION" > "$SETUP_VERSION_FILE"
printf '[Arrbit] Dependencies installation completed successfully.\n' | arrbitLogClean >> "$LOG_FILE"
exit 0
