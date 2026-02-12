# WNN Dashboard iOS App

Native iOS app for monitoring and controlling WNN experiments.

## Status: **Active Development**

## Recent Updates

### 2026-02-02
- Renamed "Dashboard" tab to "Iterations"
- Added historical experiment selection when no experiment is running
- Added flow selection to view completed experiments
- Fixed platform compatibility issues

## Features

### Connection Management
- **Dual server URLs**: Local (home network) and Remote (Tailscale/VPN)
- **Connection modes**: Local, Remote, Auto (try local first)
- **Runtime configurable**: Full URL text fields in Settings

### Real-time Updates
- **WebSocket**: Snapshot, iteration completed, phase events
- **Exponential backoff**: With ±30% jitter to prevent thundering herd
- **Auto-reconnect**: On disconnect or network change

### Iterations Tab (Main Dashboard)
- Metrics cards (Best CE, Best Acc, Phase, Iteration)
- Progress bar with phase completion
- Dual-axis chart (CE vs Accuracy) using Swift Charts
- Phase timeline (horizontal scroll, tap to view history)
- Iterations list with detail sheet
- **Historical viewing**: Select completed flows and experiments when idle

### Flows Tab
- Flow list with status badges
- Flow detail with experiments
- New flow creation form
- Stop/restart/delete actions

### Checkpoints Tab
- List with type filtering
- Size and metrics display
- Swipe to delete or download

### Settings Tab
- Local/Remote server URL configuration
- Connection mode selection
- Connection status and testing

## Project Structure

```
WNNDashboard/
├── Package.swift              # iOS 16+ only
└── Sources/WNNDashboard/
    ├── WNNDashboardApp.swift  # App entry with centralized state
    ├── ContentView.swift      # Tab navigation (Iterations, Flows, Checkpoints, Settings)
    ├── Models/                # 10 Codable structs matching Rust backend
    ├── Services/
    │   ├── SettingsStore.swift      # UserDefaults persistence
    │   ├── ConnectionManager.swift  # Local/Remote/Auto switching
    │   ├── APIClient.swift          # REST API calls
    │   └── WebSocketManager.swift   # Real-time updates
    ├── ViewModels/
    │   ├── DashboardViewModel.swift # Live + historical state
    │   ├── FlowsViewModel.swift
    │   └── CheckpointsViewModel.swift
    ├── Views/
    │   ├── Dashboard/           # Iterations, charts, phase timeline
    │   ├── Flows/               # Flow list, detail, create new
    │   ├── Checkpoints/         # Checkpoint management
    │   └── Settings/            # Connection configuration
    └── Utilities/
        ├── Formatters.swift     # CE/Acc/Date formatting
        └── Theme.swift          # Colors and styles
```

## How to Open in Xcode

1. Open Xcode
2. File → Open
3. Navigate to `dashboard/ios/WNNDashboard/` folder
4. Select the **folder** (not any file)
5. Click Open

Xcode will recognize it as a Swift Package and create a scheme automatically.

## Building

### In Xcode (Recommended)
- Build for iOS Simulator or device directly in Xcode

### Command Line
```bash
# Build for iOS (requires iOS SDK)
cd dashboard/ios/WNNDashboard
swift build --sdk $(xcrun --sdk iphoneos --show-sdk-path) --triple arm64-apple-ios16.0
```

Note: `xcodebuild` may fail on some Xcode versions due to simulator plugin issues. Building directly in Xcode works correctly.

## Architecture

- **Swift Package Manager**: Avoids fragile `.xcodeproj` files
- **iOS 16+**: Required for Swift Charts
- **No external dependencies**: Pure SwiftUI + Foundation
- **MVVM architecture**: ViewModels with `@Published` properties
- **Environment objects**: Centralized state management

## Known Issues

- **Xcode 26.2**: The `xcodebuild` command may fail with "DVTDownloads.framework" errors. Building directly in Xcode works correctly.
- **Chart tooltips**: Require iOS 17+ for `plotFrame` API

## Future Enhancements

- iPadOS optimized layout
- Push notifications for flow completion
- Widgets for home screen metrics
- TestFlight distribution

## Related Files

- Backend API: `dashboard/src/api/mod.rs`
- Database: `dashboard/src/db/mod.rs`
- Web frontend: `dashboard/frontend/`
