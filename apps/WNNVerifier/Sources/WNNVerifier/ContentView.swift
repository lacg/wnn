// ContentView - Tab-based navigation (Leaderboard, Verify, About)

import SwiftUI

public struct ContentView: View {
	@ObservedObject var leaderboardVM: LeaderboardViewModel
	@ObservedObject var verifyVM: VerifyViewModel
	@State private var selectedTab: Tab = .leaderboard

	public init(leaderboardVM: LeaderboardViewModel, verifyVM: VerifyViewModel) {
		self.leaderboardVM = leaderboardVM
		self.verifyVM = verifyVM
	}

	enum Tab: Hashable {
		case leaderboard
		case verify
		case about
	}

	public var body: some View {
		TabView(selection: $selectedTab) {
			LeaderboardView(viewModel: leaderboardVM)
				.tabItem {
					Label("Leaderboard", systemImage: "trophy")
				}
				.tag(Tab.leaderboard)

			VerifyView(viewModel: verifyVM)
				.tabItem {
					Label("Verify", systemImage: "checkmark.shield")
				}
				.tag(Tab.verify)

			AboutView()
				.tabItem {
					Label("About", systemImage: "info.circle")
				}
				.tag(Tab.about)
		}
	}
}
