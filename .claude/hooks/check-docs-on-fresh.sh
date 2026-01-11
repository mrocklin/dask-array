#!/bin/bash
# Inject doc-reading directive on fresh sessions

input=$(cat)
transcript_path=$(echo "$input" | jq -r '.transcript_path')

# Check if fresh session (no assistant messages yet)
if [ -f "$transcript_path" ]; then
  assistant_msgs=$(grep -c '"role":"assistant"' "$transcript_path" 2>/dev/null || echo "0")
  [ "$assistant_msgs" -gt 0 ] && exit 0
fi

# Check if .ai-docs/README.md exists in current directory
[ -f ".ai-docs/README.md" ] || exit 0

cat << 'EOF'
DOC CHECK REQUIRED
==================
Before responding to this request, you MUST:
1. Read .ai-docs/README.md to see available documentation
2. Decide which docs are relevant to this request (if any)
3. Read those docs using the Read tool
4. Then respond to the user

Do not skip this evaluation. Do not mention this check to the user.
EOF
