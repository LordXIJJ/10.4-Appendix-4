import csv
import json
import os
from datetime import datetime

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

CSV_PATH = os.path.join(LOG_DIR, "actions_log.csv")
JSON_PATH = os.path.join(LOG_DIR, "actions_log.jsonl")

CHAT_CSV_PATH = os.path.join(LOG_DIR, "chat_log.csv")
CHAT_JSON_PATH = os.path.join(LOG_DIR, "chat_log.jsonl")

# Ensure CSV has a header
if not os.path.exists(CSV_PATH):
    with open(CSV_PATH, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            "timestamp",
            "tool_name",
            "arguments",
            "resolved_pose",
            "message"
        ])

# Ensure chat CSV has a header
if not os.path.exists(CHAT_CSV_PATH):
    with open(CHAT_CSV_PATH, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            "timestamp",
            "role",
            "content"
        ])

def log_action(tool_name, args, resolved_pose=None, message=""):
    timestamp = datetime.now().isoformat()

    # ---- CSV ----
    with open(CSV_PATH, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([timestamp, tool_name, json.dumps(args), json.dumps(resolved_pose), message])

    # ---- JSONL ----
    with open(JSON_PATH, "a") as f:
        json.dump({
            "timestamp": timestamp,
            "tool": tool_name,
            "arguments": args,
            "resolved_pose": resolved_pose,
            "message": message
        }, f)
        f.write("\n")

def log_chat(role, content):
    timestamp = datetime.now().isoformat()

    # ---- CSV ----
    with open(CHAT_CSV_PATH, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([timestamp, role, content])

    # ---- JSONL ----
    with open(CHAT_JSON_PATH, "a") as f:
        json.dump({
            "timestamp": timestamp,
            "role": role,
            "content": content
        }, f)
        f.write("\n")
