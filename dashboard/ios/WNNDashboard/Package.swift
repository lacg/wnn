// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "WNNDashboard",
    platforms: [
        .iOS("16.1")
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
