#!/bin/bash
# Watch for CICIDS batch — queue CIC-IoT sweep when last flow (F849) starts running

DB="/Users/lacg/wnn/db/wnn.db"

while true; do
    REMAINING=$(sqlite3 "$DB" "SELECT COUNT(*) FROM flows WHERE name LIKE 'PUB50-cicids-16b%' AND status = 'queued';")
    RUNNING=$(sqlite3 "$DB" "SELECT id FROM flows WHERE name LIKE 'PUB50-cicids-16b%' AND status = 'running';")
    
    if [ "$REMAINING" -eq 0 ]; then
        echo "$(date): No more queued CICIDS flows. F849 running or done. Queuing CIC-IoT sweep..."
        
        UPDATED=$(sqlite3 "$DB" "UPDATE flows SET status = 'queued' WHERE name LIKE 'SWEEP-ciciot%' AND status = 'pending'; SELECT changes();")
        echo "$(date): Queued $UPDATED CIC-IoT sweep flows"
        
        COMPLETED=$(sqlite3 "$DB" "SELECT COUNT(*) FROM flows WHERE name LIKE 'PUB50-cicids-16b%' AND status = 'completed';")
        echo "$(date): CICIDS completed so far: $COMPLETED, running: $RUNNING"
        
        break
    else
        echo "$(date): CICIDS queued: $REMAINING, running: $RUNNING"
    fi
    
    sleep 1800  # Check every 30 minutes
done
