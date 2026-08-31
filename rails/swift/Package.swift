// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "JuniorRails",
    platforms: [.macOS(.v13), .iOS(.v16)],
    products: [.library(name: "JuniorRails", targets: ["JuniorRails"])],
    targets: [.target(name: "JuniorRails")]
)
