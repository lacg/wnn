// TrustingURLSession - URLSession that accepts self-signed certificates
// Used for connecting to local development servers with self-signed TLS

import Foundation

public final class SelfSignedCertDelegate: NSObject, URLSessionDelegate, Sendable {
    public static let shared = SelfSignedCertDelegate()

    public func urlSession(
        _ session: URLSession,
        didReceive challenge: URLAuthenticationChallenge,
        completionHandler: @escaping (URLSession.AuthChallengeDisposition, URLCredential?) -> Void
    ) {
        guard challenge.protectionSpace.authenticationMethod == NSURLAuthenticationMethodServerTrust,
              let trust = challenge.protectionSpace.serverTrust else {
            completionHandler(.performDefaultHandling, nil)
            return
        }
        completionHandler(.useCredential, URLCredential(trust: trust))
    }
}

public enum TrustingURLSession {
    /// URLSession that accepts self-signed certificates
    public static let shared: URLSession = {
        let config = URLSessionConfiguration.default
        config.timeoutIntervalForRequest = 10
        return URLSession(configuration: config, delegate: SelfSignedCertDelegate.shared, delegateQueue: nil)
    }()
}
