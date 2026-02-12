// LayoutConstants - Device-specific dimensions

import SwiftUI

public struct LayoutConstants {
    // Chart dimensions
    public static func chartHeight(for sizeClass: UserInterfaceSizeClass?) -> CGFloat {
        sizeClass == .regular ? 500 : 280
    }

    // Sidebar width for iPad dashboard layout
    public static let iPadSidebarWidth: CGFloat = 320

    // Grid columns
    public static func gridColumns(for sizeClass: UserInterfaceSizeClass?) -> [GridItem] {
        if sizeClass == .regular {
            return [GridItem(.adaptive(minimum: 140))]
        } else {
            return [GridItem(.flexible())]
        }
    }

    // Settings form max width
    public static func formMaxWidth(for sizeClass: UserInterfaceSizeClass?) -> CGFloat? {
        sizeClass == .regular ? 700 : nil
    }
}
