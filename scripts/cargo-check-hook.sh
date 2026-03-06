#!/bin/bash
# Post-edit hook: run cargo check when a .rs file is edited
# Called by Claude Code PostToolUse hook for the Edit tool
# Reads tool use context from stdin

INPUT=$(cat)

# Extract file path from the tool input
FILE_PATH=$(echo "$INPUT" | grep -o '"file_path"[[:space:]]*:[[:space:]]*"[^"]*"' | head -1 | sed 's/.*"file_path"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/')

# Only run cargo check for .rs files in the accelerator directory
if echo "$FILE_PATH" | grep -q '\.rs$'; then
	ACCEL_DIR="/Users/lacg/Library/Mobile Documents/com~apple~CloudDocs/Studies/research/wnn/src/wnn/ram/strategies/accelerator"
	if [ -d "$ACCEL_DIR" ]; then
		echo "Running cargo check for edited Rust file..."
		cd "$ACCEL_DIR" && cargo check --message-format=short 2>&1 | head -20
		EXIT_CODE=${PIPESTATUS[0]}
		if [ $EXIT_CODE -ne 0 ]; then
			echo "cargo check FAILED (exit $EXIT_CODE) — fix compilation errors before continuing"
			exit 1
		else
			echo "cargo check passed"
		fi
	fi
fi
