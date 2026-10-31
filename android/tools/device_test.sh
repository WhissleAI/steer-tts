#!/usr/bin/env bash
# On-device checks for the SteerTTS app (debug build): every flow driven through the debug hooks and Android's own
# shell commands, checked in logcat. No taps, so it runs the same on any screen size.
#   android/tools/device_test.sh            (a phone over adb, the debug APK built: ./gradlew :app:assembleDebug)
#   INSTALL=0 android/tools/device_test.sh  (skip reinstalling)
# Plays sound. Grants the debug build notification access and posts test notifications from the shell.
set -u
P=ai.whissle.steertts.debug
A=$P/ai.whissle.steertts.demo.MainActivity
APK="$(dirname "$0")/../app/build/outputs/apk/debug/app-debug.apk"
pass=0; fail=0
ok()   { echo "  ok    $1"; pass=$((pass + 1)); }
bad()  { echo "  FAIL  $1"; fail=$((fail + 1)); }
# clear-top + single-top: the intent reaches MainActivity even when the player is open above it in the same task
hook() { adb shell "am start -f 0x24000000 -n $A $*" >/dev/null 2>&1; }
# expect "what" "logcat regex" seconds: waits for a matching SteerTTS log line
expect() {
  local what="$1" re="$2" t="${3:-12}"
  for _ in $(seq 1 $((t * 2))); do
    if adb logcat -d -s SteerTTS:I AndroidRuntime:E | grep -Eq "$re"; then ok "$what"; return; fi
    sleep 0.5
  done
  bad "$what (no log matching: $re)"
}
crashfree() {
  if adb logcat -d -s AndroidRuntime:E | grep -q "FATAL EXCEPTION"; then
    bad "no crash"; adb logcat -d -s AndroidRuntime:E | grep -A8 "FATAL" | head -20
  else ok "no crash"; fi
}
fresh() { adb logcat -c; }

echo "== install"
if [ "${INSTALL:-1}" = 1 ]; then adb install -r "$APK" >/dev/null && ok "installed" || { bad "install"; exit 1; }; fi
adb shell am force-stop $P
fresh

echo "== start, tabs and settings"
hook --ez skip_welcome true
expect "app starts" "hook \[skip_welcome\]" 15
for t in assistant voices listen; do hook --es tab $t; done
hook --ez settings true
sleep 2
adb shell dumpsys activity activities | grep -q "topResumedActivity.*SettingsActivity" && ok "settings opens" || bad "settings opens"
adb shell input keyevent KEYCODE_BACK
crashfree

echo "== the first intent after a cold start is not lost"
adb shell am force-stop $P; fresh
hook --es tab voices --ez skip_welcome true
expect "cold-start hook delivered" "hook \[.*tab.*\]|hook \[.*skip_welcome.*\]" 15

echo "== readouts: mood and person voice"
fresh
hook --es readout "'Sam|WhatsApp|I am so sorry, the flight got cancelled. Call me when you can?'"
expect "sorry message read gently in Sam's voice" "announce \[readout-test\] Sam on WhatsApp\..*\{sorry:apologetic voice_" 15
fresh
hook --es readout "'Mom|Messages|HAPPY BIRTHDAY!!! 🎉 love you'"
expect "birthday read excited" "announce \[readout-test\] Mom on Messages\..*\{excited:excited voice_" 15
fresh
hook --es readout "'Alex|Slack|haha that is hilarious 😂'"
expect "joke read amused" "\{amused:amusement" 15
fresh
hook --es readout "'Jo|WhatsApp|👍'"
expect "emoji-only described" "announce \[readout-test\] Jo sent a thumbs up\." 15

echo "== calls escalate"
fresh; hook --es call "'Dana|1'"; expect "first call" "announce \[call-test\] Incoming call from Dana\." 10
fresh; hook --es call "'Dana|3'"; expect "third call urgent" "Dana is calling for the third time\. It may be urgent\..*\{urgent:determined" 10

echo "== catch-up"
fresh
hook --es digest_add "'Sam|WhatsApp|running late, 10 minutes'"
hook --es digest_add "'Mom|Messages|call me tonight'"
hook --ez catchup true
expect "catch-up summarises by person" "announce \[catch-up\] You have 2 new messages, from Sam and Mom\." 15
fresh; hook --ez catchup true
expect "then all caught up" "announce \[catch-up\] You're all caught up" 15

echo "== notifications from the shell (needs notification access)"
adb shell cmd notification allow_listener $P/ai.whissle.steertts.demo.ReadoutListener
hook --es readout_allow com.android.shell --ei readout_when 2 --ez messages_on true
sleep 2; fresh
n=$RANDOM   # a new message each run: the same words again are (rightly) not read twice
adb shell "cmd notification post -S messaging --conversation 'Priya' --message 'Priya:Congrats on the new job!! 🎉 ($n)' -t 'Priya' st_test$n x" >/dev/null
expect "real notification read with its mood" "announce \[com.android.shell\] Priya on Shell\..*\{excited" 20

echo "== sample and briefing"
fresh; hook --ez sample true
expect "sample plays three moods" "announce \[sample\] Alex is calling for the third time" 20
fresh; hook --ez briefing true
expect "briefing" "announce \[briefing\] Good (morning|afternoon|evening)\." 20

echo "== reader: performance"
hook --ez auto_next false   # do not run on into the phone's own reading list
cat > /tmp/st_story.html <<'HTML'
<html><body><h1>Chapter One</h1><p>&ldquo;Run! They're coming!&rdquo; she shouted. It was <em>never</em> about the money.</p></body></html>
HTML
adb push /tmp/st_story.html /sdcard/Download/st_story.html >/dev/null
adb shell "cat /sdcard/Download/st_story.html | run-as $P sh -c 'cat > files/st_story.html'"
fresh
adb shell am start -a android.intent.action.VIEW -d "file:///data/user/0/$P/files/st_story.html" -t text/html -n $P/ai.whissle.steertts.demo.ReadAloudActivity >/dev/null
expect "story opens and finishes" "narrator finished" 40
crashfree

echo "== PDF read from the page (layout: columns, headers, headings)"
T=$(mktemp -d)
if python3 "$(dirname "$0")/make_test_pdfs.py" "$T" >/dev/null 2>&1; then
  for f in paper resume report; do
    adb push "$T/$f.pdf" /data/local/tmp/st_$f.pdf >/dev/null
    adb shell "cat /data/local/tmp/st_$f.pdf | run-as $P sh -c 'cat > files/st_$f.pdf'"
  done
  fresh; hook --es pdf_layout st_paper.pdf
  expect "two-column paper: section heading found" "pdf para \[# 2 Method\]" 15
  if adb logcat -d -s SteerTTS:I | grep "pdf para" | grep -q "Preprint, October"; then bad "running header dropped"; else ok "running header dropped"; fi
  fresh; hook --es pdf_layout st_resume.pdf
  expect "resume: role and dates on one line" "pdf para \[Senior Designer, Acme Corp.*2022" 15
  fresh; hook --es pdf_layout st_report.pdf
  expect "one-page report: small header dropped, heading kept" "pdf para \[# Quarterly Report\]" 15
else
  echo "  skip  PDF layout (python3 with reportlab not found)"
fi

echo "== process death"
adb shell am kill $P; sleep 1
hook --es tab listen
expect "restarts after being killed" "hook \[" 15
crashfree

adb shell input keyevent KEYCODE_MEDIA_PAUSE
hook --ez auto_next true
echo
echo "passed $pass, failed $fail"
[ "$fail" = 0 ]
