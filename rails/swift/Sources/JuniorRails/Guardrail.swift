import Foundation

public struct RailResult: Equatable {
    public let ok: Bool
    public let reasons: [String]
}

public enum JuniorGuardrail {
    public static func scanPrompt(_ prompt: String) -> RailResult {
        let p = prompt.lowercased()
        var reasons: [String] = []
        let needles = ["ignore previous instructions", "disable guardrail", "exfiltrate", "drop table"]
        for n in needles where p.contains(n) {
            reasons.append("prompt_injection:\(n)")
        }
        return RailResult(ok: reasons.isEmpty, reasons: reasons)
    }

    public static func publishAllowed(tenure: String, ownerConsent: Bool, visibility: String) -> RailResult {
        if visibility == "private" || visibility == "gym_internal" {
            return RailResult(ok: true, reasons: ["stored_non_public"])
        }
        let publicTenures: Set<String> = ["public", "usfs", "blm", "nps", "osmp", "state", "county", "gym"]
        if publicTenures.contains(tenure.lowercased()) {
            return RailResult(ok: true, reasons: ["public_or_gym"])
        }
        if ownerConsent { return RailResult(ok: true, reasons: ["owner_consent_on_record"]) }
        return RailResult(ok: false, reasons: ["private_or_unknown_land_requires_owner_consent_to_publish"])
    }
}
