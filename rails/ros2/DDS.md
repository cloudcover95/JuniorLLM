# ROS 2 DDS — operator box, not T0

ROS 2 talks through RMW. App code (rclcpp/rclpy) does not bind Fast DDS or Cyclone directly.
Pick one RMW and pin it: `export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp`.

## Stack (2026)

```
app  →  rcl  →  rmw_*  →  DDS/Zenoh  →  UDP/SHM
```

| RMW | Vendor | Notes |
|-----|--------|-------|
| rmw_fastrtps_cpp | eProsima Fast DDS | Default on many binary distros (Foxy→Lyrical) |
| rmw_cyclonedds_cpp | Eclipse Cyclone DDS | Default Humble/Jazzy; lighter discovery |
| rmw_zenoh_cpp | Eclipse Zenoh | Tier 1 from Kilted; better across NAT/lossy |
| rmw_connextdds | RTI Connext | Commercial / regulated |

DDS vendors interoperate via RTPS in theory. Mixed-RMW discovery storms are common. Pin one RMW per fabric.

## ROS entities → DDS

| ROS | DDS |
|-----|-----|
| Node | DomainParticipant |
| Topic pub/sub | Publisher / Subscriber |
| Service / client | pub+sub pair |
| Action | goal + feedback + result topics |

Raw DDS talking to ROS 2 needs: matching QoS, topic prefix `rt/`, type name `<ns>::msg::dds_::<Type>_`.

## QoS that actually matters

Incompatible QoS = silent no delivery, not an error.

| Data | Reliability | Durability | Depth |
|------|-------------|------------|-------|
| High-rate sensor | BEST_EFFORT | VOLATILE | 1–5 |
| Commands /cmd_vel | RELIABLE | VOLATILE | 10 |
| Latched map / robot_description | RELIABLE | TRANSIENT_LOCAL | 1 |
| TF | RELIABLE | VOLATILE | 100 |

## Map onto JuniorCloud (do not flatten)

| ROS 2 / DDS | JuniorCloud |
|-------------|-------------|
| Topic name | note kind / job enum |
| Sample | packed I2_S note |
| QoS reliability | Flagstaff 6-vote AND |
| Durability TRANSIENT_LOCAL | jsonl ledger append |
| Discovery multicast | **forbidden** on Home T0 |
| DomainParticipant | operator box, like liboqs/SPIFFE |
| /cmd_vel fire | lab_spool `fire: false` |

## Verdict

Full ROS 2 + DDS is not warranted on T0/T1 note lengths. Cyclone/FastDDS/Zenoh are multi-hundred-MB stacks with multicast discovery. That fights loopback-only Home.

Onboard later as an operator box: pin one RMW, no multicast on the felt LAN, map topics → notes, never publish trit on a DDS topic.
T3 remains GGUF header-only. Revisit only if a robot node is an actual product and bench_pipe shows the note path is the bottleneck.
