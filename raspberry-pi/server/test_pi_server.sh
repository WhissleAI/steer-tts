#!/bin/sh
# steertts-pi end to end: HTTP endpoints (curl) and --stdin mode, against the edge model.
#   sh raspberry-pi/server/test_pi_server.sh [path/to/steertts-pi]      (ctest runs it as test_pi_server; 77 = skipped)
# Model: $STEERTTS_EDGE_MODEL, else c/models/steer-tts-edge.
set -u
here=$(cd "$(dirname "$0")" && pwd)
BIN=${1:-$here/../../c/build/steertts-pi}
MODEL=${STEERTTS_EDGE_MODEL:-$here/../../c/models/steer-tts-edge}
[ -f "$MODEL/config.json" ] || { echo "skip: no edge model at $MODEL"; exit 77; }
[ -x "$BIN" ] || { echo "skip: no $BIN"; exit 77; }
command -v curl >/dev/null || { echo "skip: curl not found"; exit 77; }
TMP=$(mktemp -d "${TMPDIR:-/tmp}/steertts-pi-test.XXXXXX")
PORT=$((20000 + $$ % 20000))
fails=0
ok() { echo "ok   $1"; }
bad() { echo "FAIL $1"; fails=$((fails + 1)); }
check() { if eval "$2"; then ok "$1"; else bad "$1"; fi; }
# a WAV of at least $2 bytes whose header says PCM 16-bit mono 24 kHz
is_wav() { [ -f "$1" ] && [ "$(wc -c < "$1")" -ge "$2" ] && [ "$(head -c 4 "$1")" = RIFF ] &&
           [ "$(od -An -tu4 -j24 -N4 "$1" | tr -d ' ')" = 24000 ] && [ "$(od -An -tu2 -j34 -N2 "$1" | tr -d ' ')" = 16 ]; }

"$BIN" --model "$MODEL" --port "$PORT" --threads 2 --quiet > "$TMP/server.log" 2>&1 &
PID=$!
trap 'kill $PID 2>/dev/null; rm -rf "$TMP"' EXIT INT TERM
B=http://127.0.0.1:$PORT
up=0
for i in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20; do
  curl -sf "$B/health" > "$TMP/health.json" 2>/dev/null && { up=1; break; }
  sleep 0.25
done
[ $up = 1 ] || { cat "$TMP/server.log"; echo "FAIL server did not start"; exit 1; }
check "health" 'grep -q "\"status\": \"ok\"" "$TMP/health.json" && grep -q "steer-tts-edge" "$TMP/health.json"'
check "voices" 'curl -sf "$B/voices" | grep -q "\"gender\": \"male\""'
check "tags" 'curl -sf "$B/tags" | grep -q "\"whisper\""'
check "controls" 'curl -sf "$B/controls" | grep -q "\"f0_end\""'
check "GET /synthesize -> wav" 'curl -sf "$B/synthesize?text=Hello+there.&ctrl=f0_end=2&voice=male" -o "$TMP/get.wav" && is_wav "$TMP/get.wav" 20000'
check "POST /synthesize -> wav" 'curl -sf "$B/synthesize" -d "{\"text\": \"Are you coming with us tonight?\", \"ctrl\": {\"f0_end\": 2, \"f0_iqr\": 1.5}, \"tag\": \"happiness\"}" -o "$TMP/post.wav" && is_wav "$TMP/post.wav" 40000'
check "POST /synthesize?format=pcm" 'curl -sf "$B/synthesize?format=pcm" -d "{\"text\": \"Are you coming with us tonight?\", \"ctrl\": {\"f0_end\": 2, \"f0_iqr\": 1.5}, \"tag\": \"happiness\"}" -o "$TMP/post.pcm" && [ $(($(wc -c < "$TMP/post.wav") - 44)) -eq "$(wc -c < "$TMP/post.pcm")" ]'
check "pcm = wav data" 'tail -c +45 "$TMP/post.wav" | cmp -s - "$TMP/post.pcm"'
T='{"text": "The engineers tested the new design for three weeks before approving it. A second sentence. <whisper>A third.</whisper>"}'
check "POST /stream (chunked wav)" 'curl -sfN "$B/stream" -d "$T" -o "$TMP/stream.wav" && is_wav "$TMP/stream.wav" 100000'
check "stream = synthesize with markup" 'curl -sf "$B/synthesize?format=pcm" -d "{\"markup\": true, ${T#\{}" -o "$TMP/whole.pcm" && tail -c +45 "$TMP/stream.wav" | cmp -s - "$TMP/whole.pcm"'
code=$(curl -s -o "$TMP/e1" -w "%{http_code}" "$B/synthesize" -d '{"text": "x", "ctrl": {"bogus": 1}}')
check "bad control -> 400 + message" '[ "$code" = 400 ] && grep -q "unknown control" "$TMP/e1"'
check "bad JSON -> 400" '[ "$(curl -s -o /dev/null -w "%{http_code}" "$B/synthesize" -d "not json")" = 400 ]'
check "unknown path -> 404" '[ "$(curl -s -o /dev/null -w "%{http_code}" "$B/nope")" = 404 ]'
check "wrong method -> 405" '[ "$(curl -s -o /dev/null -w "%{http_code}" -X DELETE "$B/health")" = 405 ]'
check "client gone mid-stream, server still serves" 'curl -s --max-time 0.2 "$B/stream" -d "{\"text\": \"$(printf "This is a long sentence that keeps going. %.0s" 1 2 3 4 5 6 7 8 9 10)\"}" -o /dev/null; curl -sf "$B/health" >/dev/null'
kill $PID 2>/dev/null
wait $PID 2>/dev/null

printf 'Hello from standard input.\n{"text": "A JSON line.", "ctrl": {"rate": 1}}\n' | "$BIN" --model "$MODEL" --stdin --quiet > "$TMP/stdin.pcm" 2> "$TMP/stdin.err"
check "--stdin -> pcm" '[ "$(wc -c < "$TMP/stdin.pcm")" -gt 40000 ] && [ ! -s "$TMP/stdin.err" ]'
printf 'One.\n{"text": "x", "voice": "nobody"}\nTwo.\n' | "$BIN" --model "$MODEL" --stdin --out-dir "$TMP/wavs" --quiet > "$TMP/stdin.jsonl" 2> "$TMP/stdin2.err"
check "--stdin --out-dir: wavs, errors reported, next line read" 'is_wav "$TMP/wavs/0001.wav" 10000 && is_wav "$TMP/wavs/0003.wav" 10000 && grep -q "unknown voice" "$TMP/stdin2.err" && [ "$(wc -l < "$TMP/stdin.jsonl" | tr -d " ")" = 2 ]'
echo "$fails failed"
[ $fails = 0 ]
