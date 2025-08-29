# -------------------------------------------------------------------------------------------------------------
# Arrbit - helpers.bash
# Version: v3.0.0-gs3.2.0 (Security-first minimal helpers, no fallbacks)
# Purpose: Essential helper functions for Arrbit scripts (config parsing, validation, file ops)
# -------------------------------------------------------------------------------------------------------------

# Prevent multiple inclusions
[[ -n "${ARRBIT_HELPERS_INCLUDED:-}" ]] && return 0
readonly ARRBIT_HELPERS_INCLUDED=1

# Fixed base path configuration (security-first, no auto-detection)
readonly ARRBIT_BASE="${ARRBIT_BASE:-/app/arrbit}"
readonly ARRBIT_CONFIG_DIR="${ARRBIT_BASE}/config"
readonly ARRBIT_DATA_DIR="${ARRBIT_BASE}/data" 
readonly ARRBIT_LOGS_DIR="${ARRBIT_BASE}/logs"
readonly ARRBIT_HELPERS_DIR="${ARRBIT_BASE}/universal/helpers"
readonly ARRBIT_SCRIPTS_DIR="${ARRBIT_BASE}/scripts"
readonly ARRBIT_ENVIRONMENTS_DIR="${ARRBIT_BASE}/environments"

# Export variables for scripts
export ARRBIT_BASE ARRBIT_CONFIG_DIR ARRBIT_DATA_DIR ARRBIT_LOGS_DIR ARRBIT_HELPERS_DIR ARRBIT_SCRIPTS_DIR ARRBIT_ENVIRONMENTS_DIR

# -------------------------------------------------------
# Configuration access (secure, fail-fast)
# -------------------------------------------------------
getFlag() {
  local flag_name="$1" config_file="${ARRBIT_CONFIG_DIR}/arrbit-config.conf"
  [[ -n "$flag_name" && -f "$config_file" ]] || return 1
  
  local flag_upper
  flag_upper=$(printf '%s' "$flag_name" | tr '[:lower:]' '[:upper:]')
  
  awk -F '=' -v key="$flag_upper" '
    $0 !~ /^[[:space:]]*#/ && NF >= 2 {
      gsub(/[[:space:]]+/, "", $1)
      if (toupper($1) == key) {
        val=$2
        sub(/[#;].*/, "", val)
        gsub(/^[[:space:]]+|[[:space:]]+$/, "", val)
        gsub(/^"+|"+$/, "", val)
        print val
        exit
      }
    }
  ' "$config_file" || return 1
}

# -------------------------------------------------------
# Essential utility functions (minimal, secure)
# -------------------------------------------------------
isReadable() { [[ -n "${1:-}" && -e "${1:-}" && -r "${1:-}" ]]; }
isWritableDir() { [[ -d "${1:-}" && -w "${1:-}" ]]; }
isValidUrl() { [[ "${1:-}" =~ ^https?://[^[:space:]]+$ ]]; }
hasCommand() { command -v "${1:-}" >/dev/null 2>&1; }

# Secure directory creation (fail fast)
ensureDir() {
  local dir="${1:-}"
  [[ -n "$dir" ]] || return 1
  [[ -d "$dir" ]] && return 0
  mkdir -p -- "$dir"
}

# Simple file size (no fallbacks - fail fast if stat unavailable)
getFileSize() {
  local file="${1:-}"
  [[ -f "$file" ]] || { printf '0'; return 1; }
  stat -c '%s' -- "$file" 2>/dev/null || { printf '0'; return 1; }
}

# Array join utility (essential only)
joinBy() {
  local delim="$1"; shift
  local out="" first=1
  for item in "$@"; do
    if [[ $first -eq 1 ]]; then
      out="$item"
      first=0
    else
      out+="${delim}${item}"  
    fi
  done
  printf '%s' "$out"
}

fi
