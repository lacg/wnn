// swift-tools-version: 6.1
import PackageDescription

let package = Package(
	name: "WNNVerifier",
	platforms: [
		.iOS(.v26),
		.macOS(.v26),
	],
	products: [
		.library(
			name: "WNNVerifier",
			targets: ["WNNVerifier"]
		)
	],
	targets: [
		.target(
			name: "WNNVerifier",
			resources: [
				.process("Resources"),
				.process("Metal"),
			]
		)
	]
)
