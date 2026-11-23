// swift-tools-version: 6.0
//
// For licensing see accompanying LICENSE file.
// Copyright (C) 2026 Apple Inc. All Rights Reserved.
//
import Foundation
import PackageDescription

let infoPlist = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
    .appendingPathComponent("Info.plist").path

let package = Package(
    name: "demo",
    platforms: [.macOS(.v15)],
    targets: [
        .executableTarget(
            name: "demo",
            linkerSettings: [
                .unsafeFlags([
                    "-Xlinker", "-sectcreate", "-Xlinker", "__TEXT",
                    "-Xlinker", "__info_plist", "-Xlinker", infoPlist,
                ])
            ]
        )
    ]
)
