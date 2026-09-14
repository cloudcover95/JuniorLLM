// TensorRT IPluginV2 stub. Compile on T1 with TensorRT SDK.
// enqueue: AbsMean + clip-round + dot. Not built in this repo CI.
#pragma once
namespace juniorosai {
struct BitLinearPlugin {
  static constexpr const char* name = "JuniorOSaiTritOn";
  bool built = false;
};
}
