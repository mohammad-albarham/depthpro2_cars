#!/usr/bin/env bash
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
swift_package="$repo/demo/apple"

if ! xcode-select -p >/dev/null 2>&1; then
    echo "demo: no Swift toolchain; install Xcode or the command line tools (xcode-select --install)" >&2
    exit 1
fi

echo "demo: building (about 20 s the first time)"
build_log="$(swift build -c release --package-path "$swift_package" 2>&1)" || {
    echo "$build_log" >&2
    exit 1
}
exec "$(swift build -c release --package-path "$swift_package" --show-bin-path)/demo" "$@"
