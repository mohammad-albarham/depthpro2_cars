//
// For licensing see accompanying LICENSE file.
// Copyright (C) 2026 Apple Inc. All Rights Reserved.
//

import Foundation

struct DemoError: LocalizedError {
    let message: String
    init(_ message: String) { self.message = message }
    var errorDescription: String? { message }
}

func note(_ message: String) {
    FileHandle.standardError.write(Data("demo: \(message)\n".utf8))
}

func fail(_ error: Error) -> Never {
    note(error.localizedDescription)
    CompiledPackage.removeAll()
    exit(1)
}

CompiledPackage.removeOnInterrupt()
CompiledPackage.removeStale()
do {
    try runLive(options: Options(arguments: CommandLine.arguments))
} catch {
    fail(error)
}
