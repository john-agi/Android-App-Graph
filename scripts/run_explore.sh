#!/usr/bin/env bash
# Auto-restart exploration on crash (e.g. API failures).
# Uses --resume-from auto so it picks up from the least-explored node.
# Logs are saved to logs/ with a timestamp, and also printed to the terminal.
#
# Usage:
#   ./scripts/run_explore.sh                    # defaults
#   ./scripts/run_explore.sh -c configs/explore.yaml --max-steps 200
#   MAX_RETRIES=10 RETRY_DELAY=15 ./scripts/run_explore.sh

set -euo pipefail

MAX_RETRIES="${MAX_RETRIES:-10}"
RETRY_DELAY="${RETRY_DELAY:-60}"  # seconds between retries

# Parse -c / --config from args to find config file
CONFIG_FILE="configs/explore.yaml"
ARGS=("$@")
for i in "${!ARGS[@]}"; do
    if [[ "${ARGS[$i]}" == "-c" || "${ARGS[$i]}" == "--config" ]]; then
        CONFIG_FILE="${ARGS[$((i+1))]}"
        break
    fi
done

# Extract app name from config YAML
APP_NAME=$(uv run python3 -c "
import yaml
with open('$CONFIG_FILE') as f:
    cfg = yaml.safe_load(f)
print(cfg['apps'][0]['name'])
")

# Create logs directory
mkdir -p logs

LOG_FILE="logs/explore_${APP_NAME}.log"

# Helper: format seconds as HH:MM:SS
fmt_duration() {
    local secs=$1
    printf "%02d:%02d:%02d" $((secs/3600)) $(( (secs%3600)/60 )) $((secs%60))
}

START_TIME=$SECONDS
attempt=0

echo "=========================================="
echo "[run_explore] Started at $(date)"
echo "[run_explore] Log file: $LOG_FILE"
echo "==========================================" | tee "$LOG_FILE"

while [ "$attempt" -lt "$MAX_RETRIES" ]; do
    attempt=$((attempt + 1))
    ATTEMPT_START=$SECONDS

    echo "==========================================" | tee -a "$LOG_FILE"
    echo "[run_explore] Attempt $attempt / $MAX_RETRIES  ($(date))" | tee -a "$LOG_FILE"
    echo "==========================================" | tee -a "$LOG_FILE"

    # Run exploration with auto-resume; pass through any extra args
    # Pipe both stdout and stderr to log file AND terminal
    if uv run app-graph --resume-from auto "$@" 2>&1 | tee -a "$LOG_FILE"; then
        TOTAL_ELAPSED=$((SECONDS - START_TIME))
        echo "==========================================" | tee -a "$LOG_FILE"
        echo "[run_explore] Exploration finished successfully." | tee -a "$LOG_FILE"
        echo "[run_explore] Total time: $(fmt_duration $TOTAL_ELAPSED)" | tee -a "$LOG_FILE"
        echo "==========================================" | tee -a "$LOG_FILE"
        exit 0
    fi

    ATTEMPT_ELAPSED=$((SECONDS - ATTEMPT_START))
    echo "[run_explore] Crashed after $(fmt_duration $ATTEMPT_ELAPSED) on attempt $attempt." | tee -a "$LOG_FILE"

    if [ "$attempt" -lt "$MAX_RETRIES" ]; then
        echo "[run_explore] Retrying in ${RETRY_DELAY}s..." | tee -a "$LOG_FILE"
        sleep "$RETRY_DELAY"
    fi
done

TOTAL_ELAPSED=$((SECONDS - START_TIME))
echo "==========================================" | tee -a "$LOG_FILE"
echo "[run_explore] Exhausted all $MAX_RETRIES retries." | tee -a "$LOG_FILE"
echo "[run_explore] Total time: $(fmt_duration $TOTAL_ELAPSED)" | tee -a "$LOG_FILE"
echo "==========================================" | tee -a "$LOG_FILE"
exit 1
