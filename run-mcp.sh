#!/bin/bash
cd ~/Desktop/Repo/threads-lens
source .venv/bin/activate
exec python -m threads_lens.mcp_server
