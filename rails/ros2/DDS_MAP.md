# ROS 2 DDS → Junior T-layers

ROS 2 app primitives sit on RMW; RMW sits on DDS (Cyclone default on Humble/Jazzy, Fast DDS still common, Zenoh rising).
Home does not ship an RMW. Map only:

| ROS 2 | DDS | Junior |
|-------|-----|--------|
| Node | Participant | Gaia / FieldCore agent |
| Topic pub/sub | DataWriter/Reader | note kind + I2_S key |
| Service | req/rep pair | handshake(note, job) |
| Action | long RPC | lurch gen+ jsonl |
| QoS reliability | DDS QoS | Flagstaff 6-vote AND |
| Discovery | multicast | **forbidden** on Home (no 0.0.0.0) |
| Shared mem | iceoryx (Linux) | not T0 |

T0 Python pack. T1 optional .so (absmean). T2 FieldCore jsonl. T3 GGUF header if file present.
RMW_IMPLEMENTATION=rmw_cyclonedds_cpp is operator-box, same class as liboqs.
