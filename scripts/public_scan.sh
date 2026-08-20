#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGETS=("$ROOT/include" "$ROOT/src" "$ROOT/tests" "$ROOT/tools" "$ROOT/Makefile")
status=0

scan() {
    local label="$1"
    local pattern="$2"
    if rg -n -I --hidden -g '!*.pyc' "$pattern" "${TARGETS[@]}"; then
        printf 'FAIL: %s\n' "$label" >&2
        status=1
    else
        printf 'PASS: %s\n' "$label"
    fi
}

scan 'private paths and network identities' '(/home/|/media/|/mnt/|/opt/|/srv/|10\.[0-9]+\.|192\.168\.|172\.(1[6-9]|2[0-9]|3[0-1])\.|synapsecorp|gitlab)'
scan 'credential-shaped material' '(glpat-|github_pat_|ghp_|BEGIN (RSA|OPENSSH|EC|DSA) PRIVATE KEY|bearer[[:space:]]+[A-Za-z0-9._-]{12,})'
scan 'activation steering implementation' '(axiom_steer|steer_pack|steering|AXIOM_STEER)'
scan 'DFlash2 executor implementation' '(dflash2|DFlash2)'

if find "$ROOT" -type f \( -name '*.safetensors' -o -name '*.gguf' -o -name '*.pem' -o -name '*.key' \) -print -quit | rg -q .; then
    printf 'FAIL: model or private-key artifact present\n' >&2
    status=1
else
    printf 'PASS: no model or private-key artifact\n'
fi

exit "$status"
