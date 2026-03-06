#!/bin/bash
# Headless mode: Deep analysis of a specific flow
# Usage: bash scripts/analyze-flow.sh <flow_id>

if [ -z "$1" ]; then
	echo "Usage: $0 <flow_id>"
	exit 1
fi

FLOW_ID="$1"

claude -p "Analyze flow $FLOW_ID in detail:
1. Fetch the flow: curl -sk https://localhost:3000/api/flows/$FLOW_ID
2. Show flow metadata (name, status, created_at)
3. For each experiment: show name, type, status, and key results
4. Show iteration-by-iteration breakdown if available (best CE and accuracy at each iteration)
5. Highlight the best configuration found
6. If the flow failed, show the full error context
Show ACTUAL database values only — never estimate." \
	--allowedTools "Bash,Read,Grep"
