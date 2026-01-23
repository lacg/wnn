#!/bin/bash
# Monitor overnight run every hour
LOG_FILE="/Users/lacg/Library/Mobile Documents/com~apple~CloudDocs/Studies/research/wnn/overnight_monitor.log"
MAIN_PID=$1

echo "=== Overnight Monitor Started ===" > "$LOG_FILE"
echo "Monitoring PID: $MAIN_PID" >> "$LOG_FILE"
echo "Started: $(date)" >> "$LOG_FILE"
echo "" >> "$LOG_FILE"

while true; do
    # Check if main process is still running
    if ! ps -p $MAIN_PID > /dev/null 2>&1; then
        echo "$(date) | Main process $MAIN_PID finished or killed" >> "$LOG_FILE"
        break
    fi
    
    echo "========================================" >> "$LOG_FILE"
    echo "$(date)" >> "$LOG_FILE"
    
    # CPU and Memory
    echo "--- Process Stats ---" >> "$LOG_FILE"
    ps -p $MAIN_PID -o pid,%cpu,%mem,rss,vsz 2>/dev/null >> "$LOG_FILE"
    
    # Physical memory in GB
    RSS_KB=$(ps -p $MAIN_PID -o rss= 2>/dev/null)
    if [ -n "$RSS_KB" ]; then
        RSS_GB=$(echo "scale=2; $RSS_KB / 1024 / 1024" | bc)
        echo "Physical Memory: ${RSS_GB} GB" >> "$LOG_FILE"
    fi
    
    # Virtual memory in GB  
    VSZ_KB=$(ps -p $MAIN_PID -o vsz= 2>/dev/null)
    if [ -n "$VSZ_KB" ]; then
        VSZ_GB=$(echo "scale=2; $VSZ_KB / 1024 / 1024" | bc)
        echo "Virtual Memory: ${VSZ_GB} GB" >> "$LOG_FILE"
    fi
    
    # GPU usage (Metal on macOS)
    echo "--- GPU Stats ---" >> "$LOG_FILE"
    sudo powermetrics --samplers gpu_power -i 1000 -n 1 2>/dev/null | grep -E "GPU|active" | head -5 >> "$LOG_FILE" || echo "GPU stats unavailable" >> "$LOG_FILE"
    
    # Latest generation info
    echo "--- Latest Progress ---" >> "$LOG_FILE"
    tail -5 /Users/lacg/Library/Mobile\ Documents/com~apple~CloudDocs/Studies/research/wnn/nohup.out 2>/dev/null | grep -E "Gen |Phase|Best" >> "$LOG_FILE"
    
    # Calculate genome/second from recent output
    echo "--- Performance ---" >> "$LOG_FILE"
    RECENT=$(tail -100 /Users/lacg/Library/Mobile\ Documents/com~apple~CloudDocs/Studies/research/wnn/nohup.out 2>/dev/null | grep "Genome" | tail -20)
    if [ -n "$RECENT" ]; then
        FIRST_TIME=$(echo "$RECENT" | head -1 | cut -d'|' -f1 | tr -d ' ')
        LAST_TIME=$(echo "$RECENT" | tail -1 | cut -d'|' -f1 | tr -d ' ')
        COUNT=$(echo "$RECENT" | wc -l | tr -d ' ')
        echo "Recent: $COUNT genomes from $FIRST_TIME to $LAST_TIME" >> "$LOG_FILE"
    fi
    
    echo "" >> "$LOG_FILE"
    
    # Sleep 1 hour
    sleep 3600
done

echo "=== Monitor Ended: $(date) ===" >> "$LOG_FILE"
