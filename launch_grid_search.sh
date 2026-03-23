#!/bin/bash
# launch_grid_search.sh - Launch overnight grid search in a tmux session.
#
# Usage:
#   ./launch_grid_search.sh              # Tier 1 (default, ~18 configs, ~9h)
#   ./launch_grid_search.sh --tier 2     # Tier 2 (~160 configs, ~80h)
#   ./launch_grid_search.sh --tier 3     # Tier 3 (full sweep)
#   ./launch_grid_search.sh --dry-run    # List configs without running
#   ./launch_grid_search.sh --attach     # Attach to existing session
#   ./launch_grid_search.sh --status     # Show progress from latest run
#
# The script runs inside a tmux session named "k25-grid". Output is logged
# to sglang-cookbook/grid_results/<timestamp>/.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SESSION_NAME="k25-grid"
TIER=1
DRY_RUN=""
ACTION="launch"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --tier)    TIER="$2"; shift 2 ;;
        --dry-run) DRY_RUN="--dry-run"; shift ;;
        --attach)  ACTION="attach"; shift ;;
        --status)  ACTION="status"; shift ;;
        *)         echo "Unknown: $1"; exit 1 ;;
    esac
done

case "$ACTION" in
    attach)
        if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
            tmux attach-session -t "$SESSION_NAME"
        else
            echo "No tmux session '${SESSION_NAME}' found."
            exit 1
        fi
        ;;

    status)
        LATEST_DIR=$(ls -td "${SCRIPT_DIR}/grid_results/"*/ 2>/dev/null | head -1)
        if [ -z "$LATEST_DIR" ]; then
            echo "No grid search results found."
            exit 1
        fi
        echo "=== Latest run: ${LATEST_DIR} ==="
        echo ""

        if [ -f "${LATEST_DIR}/grid_search.log" ]; then
            echo "--- Last 20 log lines ---"
            tail -20 "${LATEST_DIR}/grid_search.log"
            echo ""
        fi

        if [ -f "${LATEST_DIR}/results.csv" ]; then
            local_total=$(wc -l < "${LATEST_DIR}/results.csv")
            echo "--- Results: $((local_total - 1)) benchmark rows ---"
            echo ""

            echo "Top 5 decode throughput (tok/s):"
            grep "bench_one_batch" "${LATEST_DIR}/results.csv" 2>/dev/null \
                | sort -t',' -k7 -rn \
                | head -5 \
                | awk -F',' '{printf "  %-55s decode=%s tok/s  IL=%s OL=%s\n", $1, $7, $3, $4}'
            echo ""

            echo "Top 5 serving throughput (output tok/s):"
            grep "bench_serving" "${LATEST_DIR}/results.csv" 2>/dev/null \
                | sort -t',' -k9 -rn \
                | head -5 \
                | awk -F',' '{printf "  %-55s out_tps=%s  conc=%s\n", $1, $9, $5}'
        fi

        if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
            echo ""
            echo "Session '${SESSION_NAME}' is RUNNING. Attach with:"
            echo "  tmux attach -t ${SESSION_NAME}"
        else
            echo ""
            echo "Session '${SESSION_NAME}' is not running."
        fi
        ;;

    launch)
        if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
            echo "Session '${SESSION_NAME}' already exists!"
            echo "  Attach:  tmux attach -t ${SESSION_NAME}"
            echo "  Kill:    tmux kill-session -t ${SESSION_NAME}"
            exit 1
        fi

        echo "========================================"
        echo "Launching K2.5 Grid Search"
        echo "  Tier:    ${TIER}"
        echo "  Session: ${SESSION_NAME}"
        echo "  DryRun:  ${DRY_RUN:-no}"
        echo "========================================"

        if [ -n "$DRY_RUN" ]; then
            bash "${SCRIPT_DIR}/grid_search.sh" --tier "$TIER" --dry-run
            exit 0
        fi

        tmux new-session -d -s "$SESSION_NAME" -x 200 -y 50

        tmux send-keys -t "$SESSION_NAME" \
            "bash ${SCRIPT_DIR}/grid_search.sh --tier ${TIER} 2>&1 | tee ${SCRIPT_DIR}/grid_search_latest.log; echo '=== GRID SEARCH COMPLETE ===' " Enter

        echo ""
        echo "Grid search launched in tmux session '${SESSION_NAME}'."
        echo ""
        echo "Commands:"
        echo "  Attach:   tmux attach -t ${SESSION_NAME}"
        echo "  Status:   $0 --status"
        echo "  Kill:     tmux kill-session -t ${SESSION_NAME}"
        echo "  Detach:   Ctrl-b d (from inside tmux)"
        ;;
esac
