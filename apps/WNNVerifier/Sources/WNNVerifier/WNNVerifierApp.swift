// WNNVerifierApp - Main app entry point

import SwiftUI

@main
public struct WNNVerifierApp: App {
	@StateObject private var leaderboardVM = LeaderboardViewModel()
	@StateObject private var verifyVM = VerifyViewModel()

	public init() {}

	public var body: some Scene {
		WindowGroup {
			ContentView(leaderboardVM: leaderboardVM, verifyVM: verifyVM)
		}
	}
}
