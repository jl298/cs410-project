#!/bin/bash

set -e
set -u
set -o pipefail

LOGDIR="../logs"
TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
LOGFILE="$LOGDIR/pipeline_${TIMESTAMP}.log"

mkdir -p "$LOGDIR"

log() {
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo "[$timestamp] $1" | tee -a "$LOGFILE"
}

handle_error() {
    local exit_code=$?
    local line_number=$1
    log "Error occurred in line $line_number, exit code: $exit_code"
    exit $exit_code
}

trap 'handle_error ${LINENO}' ERR

if ! command -v python3 &> /dev/null; then
    log "ERROR: Python3 is not installed"
    exit 1
fi

clean_datasets() {
    log "Cleaning dataset files..."
    local files=(
        "../dataset/sentiment-dataset.json"
        "../dataset/sns-posts-dataset.json"
        "../dataset/sns-posts-dataset.recommender.json"
        "../dataset/sns-posts-dataset.sentiment.json"
        "../dataset/sns-user-dataset.json"
    )

    local has_error=0
    for file in "${files[@]}"; do
        if [ -f "$file" ]; then
            if rm "$file" 2>/dev/null; then
                log "Deleted: $file"
            else
                log "ERROR: Permission denied when trying to delete $file"
                log "Please run with sufficient permissions (try: sudo $0 ${1:-})"
                has_error=1
            fi
        else
            log "File not found: $file"
        fi
    done

    if [ $has_error -eq 1 ]; then
        return 1
    fi

    log "Dataset cleaning completed"
    return 0
}

run_script() {
    local script=$1
    local description=$2

    log "Starting $description..."
    local start_time=$(date +%s)

    if python3 "$script" 2>&1 | tee -a "$LOGFILE"; then
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        log "Completed $description in $duration seconds"
        return 0
    else
        log "Failed to execute $description"
        return 1
    fi
}

main() {
    if ! clean_datasets; then
        log "Failed to clean datasets. Please check permissions and try again."
        exit 1
    fi

    if [ "${1:-}" = "clean" ]; then
        log "Clean only mode - exiting"
        exit 0
    fi

    log "Starting data processing and model training pipeline"

    run_script "ingestion_backfill.py" "Data Ingestion" || exit 1

    run_script "sentiment_training.py" "Sentiment Model Training" || exit 1

    run_script "sentiment_backfill.py" "Sentiment Backfill" || exit 1

    run_script "recommender_training.py" "Recommender Model Training" || exit 1

    log "Pipeline completed successfully"
}

main "${1:-}"

exit 0