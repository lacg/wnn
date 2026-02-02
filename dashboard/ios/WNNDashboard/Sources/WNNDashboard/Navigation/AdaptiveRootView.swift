// AdaptiveRootView - iPhone/iPad layout switcher

import SwiftUI

public struct AdaptiveRootView: View {
    @Environment(\.horizontalSizeClass) private var horizontalSizeClass
    @State private var selectedDestination: NavigationDestination? = .dashboard

    public init() {}

    public var body: some View {
        if horizontalSizeClass == .regular {
            iPadLayout
        } else {
            iPhoneLayout
        }
    }

    private var iPadLayout: some View {
        NavigationSplitView {
            SidebarView(selection: $selectedDestination)
        } detail: {
            detailView
        }
    }

    private var iPhoneLayout: some View {
        TabView(selection: Binding(
            get: { selectedDestination ?? .dashboard },
            set: { selectedDestination = $0 }
        )) {
            DashboardView()
                .tabItem {
                    Label("Iterations", systemImage: "chart.line.uptrend.xyaxis")
                }
                .tag(NavigationDestination.dashboard)

            FlowsListView()
                .tabItem {
                    Label("Flows", systemImage: "arrow.triangle.branch")
                }
                .tag(NavigationDestination.flows)

            CheckpointsListView()
                .tabItem {
                    Label("Checkpoints", systemImage: "externaldrive")
                }
                .tag(NavigationDestination.checkpoints)

            SettingsView()
                .tabItem {
                    Label("Settings", systemImage: "gear")
                }
                .tag(NavigationDestination.settings)
        }
    }

    @ViewBuilder
    private var detailView: some View {
        switch selectedDestination {
        case .dashboard, .none:
            DashboardView()
        case .flows:
            FlowsListView()
        case .checkpoints:
            CheckpointsListView()
        case .settings:
            SettingsView()
        }
    }
}
