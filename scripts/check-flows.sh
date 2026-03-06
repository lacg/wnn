#!/bin/bash
# Headless mode: Check all flow statuses via Claude Code
# Usage: bash scripts/check-flows.sh

claude -p "Query all flows from the dashboard API (curl -sk https://localhost:3000/api/flows) and display a formatted table showing: ID, Name, Status, Best CE, Best Accuracy, number of experiments. For failed flows show the error. For running flows show which experiment is active. Show ACTUAL database values only." \
	--allowedTools "Bash,Read,Grep"
