#!/usr/bin/env bash
# svc.sh - control the cv-lab live UI through systemd --user.
set -uo pipefail
cd "$(dirname "$0")"
PORT=8099
UNIT=cv-lab-webui.service

case "$1" in
  start)   systemctl --user start "$UNIT" && sleep 7 && systemctl --user is-active "$UNIT" ;;
  stop)    systemctl --user stop "$UNIT" && echo "stopped" ;;
  restart) systemctl --user restart "$UNIT" && sleep 8 && systemctl --user is-active "$UNIT" ;;
  enable)  systemctl --user enable "$UNIT" ;;
  disable) systemctl --user disable "$UNIT" ;;
  logs)    journalctl --user -u "$UNIT" -n 40 --no-pager ;;
  status)
    systemctl --user is-active "$UNIT"
    curl -s -m 3 "http://127.0.0.1:$PORT/events" | head -c 400; echo ;;
  *)
    echo "usage: $0 {start|stop|restart|status|logs|enable|disable}" ;;
esac
