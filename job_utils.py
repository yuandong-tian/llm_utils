#!/usr/bin/env python3
"""Job discovery utilities for the LLM router.

Provides squeue-based job listing and log parsing so the router can
auto-discover running backends without hacking sys.path.
"""

import glob
import os
import re
import subprocess
from typing import Dict, List, Optional


def run_cmd(cmd: str, timeout: int = 10) -> str:
    """Run a shell command and return stdout."""
    try:
        r = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=timeout
        )
        return r.stdout.strip()
    except Exception:
        return ""


def get_jobs() -> List[Dict[str, str]]:
    """Get running jobs from squeue."""
    out = run_cmd(
        "squeue -u $USER -o '%.18i %.12P %.12j %.8T %.10M %.6D %R' --noheader"
    )
    jobs = []
    for line in out.splitlines():
        parts = line.split()
        if len(parts) >= 7:
            jobs.append(
                {
                    "job_id": parts[0],
                    "partition": parts[1],
                    "name": parts[2],
                    "state": parts[3],
                    "time": parts[4],
                    "node": parts[6],
                }
            )
    return jobs


def find_job_log(job_id: str, log_dir: Optional[str] = None) -> Optional[str]:
    """Find the .err log file for a given job ID.

    Searches log_dir (defaults to env LLM_LOG_DIR or ~/test_cluster/yuandong).
    """
    if log_dir is None:
        log_dir = os.environ.get(
            "LLM_LOG_DIR",
            os.path.expanduser("~/test_cluster/yuandong"),
        )
    matches = glob.glob(os.path.join(log_dir, f"*_{job_id}.err"))
    return matches[0] if matches else None


def parse_log_info(job: Dict[str, str], log_dir: Optional[str] = None) -> Dict[str, str]:
    """Parse status and port from a job's err log."""
    log_file = find_job_log(job["job_id"], log_dir=log_dir)
    if not log_file:
        return {"status": "Pending", "port": "?"}

    header = run_cmd(f"head -20 {log_file} 2>/dev/null")
    port = "?"
    if header:
        m = re.search(r"\bport=(\d+)", header)
        if m:
            port = int(m.group(1))

    tail = run_cmd(f"tail -100 {log_file} 2>/dev/null")
    if not tail:
        return {"status": "Pending", "port": port}

    ready_check = run_cmd(
        f"grep -c 'ready to roll\\|Uvicorn running on' {log_file} 2>/dev/null"
    )
    is_ready = ready_check.strip() not in ("", "0")

    if is_ready:
        if "Prefill batch" in tail or "Decode batch" in tail:
            status = "Serving"
        else:
            status = "Ready"
    elif "DeepGEMM" in tail.split("\n")[-1] or "DeepGEMM JIT" in tail:
        status = "DeepGEMM JIT compile"
    else:
        m = re.findall(r"Loading safetensors.*?(\d+)%", tail)
        if m:
            status = f"Loading weights {m[-1]}%"
        elif "Installing transformers" in tail:
            status = "Installing deps"
        elif "importing docker" in tail or "imported docker" in tail:
            status = "Docker image setup"
        elif "Load weight begin" in tail:
            status = "Loading weights 0%"
        else:
            status = "Starting..."

    return {"status": status, "port": port}
