# WNN Dashboard iOS App

Native iOS app for monitoring and controlling WNN experiments.

## Status: **Parked** (Structure Complete, Needs Testing)

## What Was Done

### 1. Project Structure
Created as a **Swift Package** (not Xcode project) for cleaner maintenance:

```
WNNDashboard/
├── Package.swift              # iOS 16+, macOS 13+
└── Sources/WNNDashboard/
    ├── WNNDashboardApp.swift  # App entry with centralized state
    ├── ContentView.swift      # Tab navigation (Dashboard, Flows, Checkpoints, Settings)
    ├── Models/                # 10 Codable structs matching Rust backend
    │   ├── StatusEnums.swift
    │   ├── Flow.swift
    │   ├── Experiment.swift
    │   ├── Phase.swift
    │   ├── Iteration.swift
    │   ├── GenomeEvaluation.swift
    │   ├── Checkpoint.swift
    │   ├── Snapshot.swift
    │   ├── WebSocketMessage.swift
    │   └── AnyCodable.swift
    ├── Services/
    │   ├── SettingsStore.swift      # UserDefaults persistence
    │   ├── ConnectionManager.swift  # Local/Remote/Auto switching
    │   ├── APIClient.swift          # REST API calls
    │   └── WebSocketManager.swift   # Real-time updates with reconnection
    ├── ViewModels/
    │   ├── DashboardViewModel.swift
    │   ├── FlowsViewModel.swift
    │   └── CheckpointsViewModel.swift
    ├── Views/
    │   ├── Dashboard/           # Live metrics, charts, iterations
    │   ├── Flows/               # Flow list, detail, create new
    │   ├── Checkpoints/         # Checkpoint management
    │   └── Settings/            # Connection configuration
    ├── Utilities/
    │   ├── Formatters.swift     # CE/Acc/Date formatting
    │   └── Theme.swift          # Colors and styles
    └── Resources/
        └── Assets.xcassets/     # App icon, accent color
```

### 2. Key Features Implemented

#### Connection Management
- **Dual server URLs**: Local (home network) and Remote (Tailscale/VPN)
- **Connection modes**: Local, Remote, Auto (try local first)
- **Runtime configurable**: Full URL text fields in Settings
- **Network detection**: Uses NWPathMonitor for WiFi/Cellular status

#### WebSocket
- **Real-time updates**: Snapshot, iteration completed, phase events
- **Exponential backoff**: With ±30% jitter to prevent thundering herd
- **Auto-reconnect**: On disconnect or network change
- **Tagged enum decoding**: Handles `{"type": "...", "data": {...}}` format

#### Dashboard
- Metrics cards (Best CE, Best Acc, Phase, Iteration)
- Progress bar with phase completion
- Dual-axis chart (CE vs Accuracy) using Swift Charts
- Phase timeline (horizontal scroll)
- Iterations list with detail sheet

#### Flows
- Flow list with status badges
- Flow detail with experiments
- New flow creation form
- Stop/restart/delete actions

#### Checkpoints
- List with type filtering
- Size and metrics display
- Swipe to delete or download

### 3. Technical Decisions

- **Swift Package Manager**: Avoids fragile `.xcodeproj` files
- **iOS 16+**: Required for Swift Charts
- **No external dependencies**: Pure SwiftUI + Foundation
- **MVVM architecture**: ViewModels with `@Published` properties
- **Environment objects**: Centralized state management

## How to Open in Xcode

1. Open Xcode
2. File → Open
3. Navigate to `dashboard/ios/WNNDashboard/` folder
4. Select the **folder** (not any file)
5. Click Open

Xcode will recognize it as a Swift Package and create a scheme automatically.

## Next Steps

### Must Do Before Testing
1. **Add app entry point for iOS target**: The package is a library, need to add an iOS app target or create a separate Xcode project that imports the package
2. **Configure Info.plist**: App transport security for local HTTP connections
3. **Test on Simulator**: Verify API connections work

### Testing Phase
1. **Connection testing**: Verify local IP connection from Simulator
2. **WebSocket testing**: Confirm live updates appear
3. **API testing**: Test fetching experiments, phases, iterations
4. **Flow control**: Create a test flow, stop it, delete it
5. **Device testing**: Deploy to physical iPhone

### Future Enhancements
- Push notifications for flow completion
- iPad layout optimization
- Widgets for home screen metrics
- App icon design
- TestFlight distribution

## Related Commits
- `d5c9265` - Add iOS app for WNN Dashboard (SwiftUI)
- `b6bd279` - Improve iOS settings with full URL configuration
- `73dd080` - Restructure iOS app as Swift Package

## Notes
- The Rust backend supports HTTPS with self-signed certs
- iOS needs App Transport Security exception for local development
- WebSocket reconnection handles network transitions (WiFi ↔ Cellular)
