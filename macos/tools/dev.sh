#!/bin/bash
# Sends a developer command to a SteerTTS started with STEERTTS_DEV=1 (see Sources/SteerTTSApp/DevHooks.swift).
#   tools/dev.sh section:voices      tools/dev.sh snapshot:/tmp/s.png      tools/dev.sh open:/path/book.epub
cat > /tmp/steertts_dev.swift <<SWIFT
import Foundation
DistributedNotificationCenter.default().postNotificationName(Notification.Name("ai.whissle.steertts.dev"),
    object: CommandLine.arguments[1], userInfo: nil, deliverImmediately: true)
SWIFT
[ -x /tmp/steertts_dev ] || swiftc -O /tmp/steertts_dev.swift -o /tmp/steertts_dev
/tmp/steertts_dev "$1"
