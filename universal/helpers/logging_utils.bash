# -------------------------------------------------------------------------------------------------------------
# Arrbit - logging_utils.bash  
# Version: v3.0.0-gs3.2.0 (Security-first minimal logging, no fallbacks)
# Purpose: Minimal secure logging system with UTC timestamps and direct system integration
# -------------------------------------------------------------------------------------------------------------

# Fixed base path and required directories
readonly ARRBIT_BASE="${ARRBIT_BASE:-/app/arrbit}"
readonly ARRBIT_LOGS_DIR="${ARRBIT_BASE}/logs"
export ARRBIT_BASE ARRBIT_LOGS_DIR

# Ensure log directory exists (fail fast if cannot create)
mkdir -p "${ARRBIT_LOGS_DIR}" || exit 1

# Simple color constants (terminal only)
readonly CYAN='\033[96m' YELLOW='\033[93m' RED='\033[91m' GREEN='\033[92m' NC='\033[0m'

# ------------------------------------------------
# Core logging functions (RFC 5424 compliant levels)
# ------------------------------------------------
log_info() {
  local msg="$*"
  [[ -z "$msg" ]] && return 1
  local ts
  ts=$(date -u +'%Y-%m-%dT%H:%M:%SZ')
  
  # Terminal output (colored)
  if [[ -t 1 && -z "${ARRBIT_NO_COLOR:-}" ]]; then
    printf '%b %s\n' "${CYAN}[Arrbit]${NC}" "$msg"
  else
    printf '[Arrbit] %s\n' "$msg"
  fi
  
  # File output (structured, no colors)
  [[ -n "${LOG_FILE:-}" && -w "$(dirname "${LOG_FILE}" 2>/dev/null)" ]] && printf '%s [INFO] %s\n' "$ts" "$msg" >> "$LOG_FILE"
}

log_warning() {
  local msg="$*"
  [[ -z "$msg" ]] && return 1
  local ts
  ts=$(date -u +'%Y-%m-%dT%H:%M:%SZ')
  
  # Terminal output (colored)
  if [[ -t 1 && -z "${ARRBIT_NO_COLOR:-}" ]]; then
    printf '%b %bWARNING%b %s\n' "${CYAN}[Arrbit]${NC}" "$YELLOW" "$NC" "$msg"
  else
    printf '[Arrbit] WARNING %s\n' "$msg"
  fi
  
  # File output (structured, no colors)
  [[ -n "${LOG_FILE:-}" && -w "$(dirname "${LOG_FILE}" 2>/dev/null)" ]] && printf '%s [WARN] %s\n' "$ts" "$msg" >> "$LOG_FILE"
}

log_error() {
  local msg="$*"
  [[ -z "$msg" ]] && return 1
  local ts
  ts=$(date -u +'%Y-%m-%dT%H:%M:%SZ')
  
  # Terminal output (colored, to stderr)
  if [[ -t 2 && -z "${ARRBIT_NO_COLOR:-}" ]]; then
    printf '%b %bERROR%b %s\n' "${CYAN}[Arrbit]${NC}" "$RED" "$NC" "$msg" >&2
  else
    printf '[Arrbit] ERROR %s\n' "$msg" >&2
  fi
  
  # File output (structured, no colors)  
  [[ -n "${LOG_FILE:-}" && -w "$(dirname "${LOG_FILE}" 2>/dev/null)" ]] && printf '%s [ERROR] %s\n' "$ts" "$msg" >> "$LOG_FILE"
}

# Trace level logging (minimal implementation)
log_trace() {
  [[ "${ARRBIT_LOG_LEVEL:-}" == "TRACE" ]] || return 0
  log_info "TRACE: $*"
}

# ------------------------------------------------
# Log file initialization and utilities
# ------------------------------------------------
arrbitInitLog() {
  local target="${1:-${LOG_FILE:-}}"
  [[ -z "$target" ]] && return 1
  local parent
  parent=$(dirname "$target")
  mkdir -p "$parent" || return 1
  touch "$target" || return 1
  [[ -z "${LOG_FILE:-}" ]] && { LOG_FILE="$target"; export LOG_FILE; }
  return 0
}

# Standard banner with minimal formatting
arrbitBanner() {
  local name="${1:-Script}" ver="${2:-}"
  if [[ -t 1 && -z "${ARRBIT_NO_COLOR:-}" ]]; then
    printf '%b %b%s%s%b\n' "${CYAN}[Arrbit]${NC}" "$GREEN" "$name" "${ver:+ $ver}" "$NC"
  else
    printf '[Arrbit] %s%s\n' "$name" "${ver:+ $ver}"
  fi
}

# Simplified log retention (keep newest N files)
arrbitPurgeOldLogs() {
  local max="${1:-3}" dir="${2:-$ARRBIT_LOGS_DIR}"
  [[ -d "$dir" ]] || return 0
  
  local files=()
  while IFS= read -r -d '' file; do
    files+=("$file")
  done < <(find "$dir" -name "arrbit-*.log" -type f -print0 2>/dev/null)
  
  [[ ${#files[@]} -le $max ]] && return 0
  
  # Sort by modification time (newest first) and remove excess
  printf '%s\0' "${files[@]}" | xargs -0 ls -t | tail -n +$((max + 1)) | xargs -r rm -f
}
