import Foundation

extension Foundation.Bundle {
    static let module: Bundle = {
        let mainPath = Bundle.main.bundleURL.appendingPathComponent("WNNDashboard_WNNDashboard.bundle").path
        let buildPath = "/Users/lacg/Library/Mobile Documents/com~apple~CloudDocs/Studies/research/wnn/dashboard/ios/WNNDashboard/.build/arm64-apple-macosx/debug/WNNDashboard_WNNDashboard.bundle"

        let preferredBundle = Bundle(path: mainPath)

        guard let bundle = preferredBundle ?? Bundle(path: buildPath) else {
            // Users can write a function called fatalError themselves, we should be resilient against that.
            Swift.fatalError("could not load resource bundle: from \(mainPath) or \(buildPath)")
        }

        return bundle
    }()
}