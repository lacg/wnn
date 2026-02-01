// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "WNNDashboard",
    platforms: [
        .iOS(.v16),
        .macOS(.v13)
    ],
    products: [
        .library(
            name: "WNNDashboard",
            targets: ["WNNDashboard"]
        )
    ],
    targets: [
        .target(
            name: "WNNDashboard",
            resources: [
                .process("Resources")
            ]
        )
    ]
)
