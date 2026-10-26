#!/bin/bash
# Builds SteerTTS.app (universal: Apple silicon + Intel) with the edge model inside, signed with the Developer ID.
#   ./build.sh               build, sign, put dist/SteerTTS.app and dist/SteerTTS-<version>.dmg
#   ./build.sh --notarize    also notarize and staple (needs: xcrun notarytool store-credentials WHISSLE_NOTARIZE)
#   ./build.sh --dev         arm64 only, ad-hoc signed, no DMG (fast)
set -euo pipefail
cd "$(dirname "$0")"
VERSION=1.0.0
BUILD=1
IDENTITY="Developer ID Application: Whissle LLC (XXK95K57U5)"
PROFILE=WHISSLE_NOTARIZE
MODE=${1:-}

APP=dist/SteerTTS.app
rm -rf "$APP"
if [[ "$MODE" == "--dev" ]]; then
  swift build -c release --product SteerTTS
  BIN=.build/release/SteerTTS
else
  swift build -c release --product SteerTTS --arch arm64 --arch x86_64
  BIN=.build/apple/Products/Release/SteerTTS
fi

mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"
cp "$BIN" "$APP/Contents/MacOS/SteerTTS"
sed -e "s/__VERSION__/$VERSION/" -e "s/__BUILD__/$BUILD/" Resources/Info.plist > "$APP/Contents/Info.plist"
printf 'APPL????' > "$APP/Contents/PkgInfo"
cp Resources/AppIcon.icns "$APP/Contents/Resources/"
# the model: weights, voices and the text front end (the ONNX export is not used by the C engine)
mkdir -p "$APP/Contents/Resources/model"
rsync -a --exclude onnx ../c/models/steer-tts-edge/ "$APP/Contents/Resources/model/"

if [[ "$MODE" == "--dev" ]]; then
  codesign --force --deep --sign - "$APP"
  echo "$APP (ad-hoc signed)"
  exit 0
fi
codesign --force --options runtime --timestamp --entitlements Resources/SteerTTS.entitlements --sign "$IDENTITY" "$APP"
codesign --verify --strict --verbose=2 "$APP"

DMG=dist/SteerTTS-$VERSION.dmg
rm -rf dist/dmg "$DMG"
mkdir -p dist/dmg
cp -R "$APP" dist/dmg/
ln -s /Applications dist/dmg/Applications
hdiutil create -volname SteerTTS -srcfolder dist/dmg -ov -format UDZO "$DMG" >/dev/null
rm -rf dist/dmg
codesign --sign "$IDENTITY" --timestamp "$DMG"

if [[ "$MODE" == "--notarize" ]]; then
  xcrun notarytool submit "$DMG" --keychain-profile "$PROFILE" --wait
  xcrun stapler staple "$DMG"
fi
echo "$APP"
echo "$DMG"
