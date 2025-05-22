#!/bin/bash

# Path to log file
LOGFILE="/var/log/bleeparr_monitor.log"
TMPLOG="/tmp/bleeparr_last_run.log"

# Find first unprocessed video
VIDEO=$(find /mnt/storagepool/CleanVid/ \
  -type f \( -iname "*.mp4" -o -iname "*.mkv" \) \
  ! -iname "*CleanedWithCleanVid*" \
  ! -iname "*edited by Bleeparr*" \
  | sort | head -n 1)

if [[ -n "$VIDEO" ]]; then
  echo "🕒 $(date): Processing '$VIDEO'" > "$TMPLOG"

  # Always quote $VIDEO to avoid shell issues
  if bleeparr --input "$VIDEO" --delete-original >> "$TMPLOG" 2>&1; then
    echo "✅ $(date): Success: $VIDEO" >> "$TMPLOG"
    SUBJECT="✅ Bleeparr completed: $(basename "$VIDEO")"
  else
    echo "❌ $(date): Error processing $VIDEO" >> "$TMPLOG"
    SUBJECT="❌ Bleeparr failed: $(basename "$VIDEO")"
  fi

  # Append to main log
  cat "$TMPLOG" >> "$LOGFILE"

  # Send email only when a video was processed
  (
    echo "Subject: $SUBJECT"
    echo "To: davidpeele@gmail.com"
    echo
    cat "$TMPLOG"
  ) | msmtp davidpeele@gmail.com

else
  echo "🕒 $(date): No matching videos found." >> "$LOGFILE"
fi