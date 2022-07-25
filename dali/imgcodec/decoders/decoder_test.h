// Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef DALI_IMGCODEC_DECODERS_DECODER_IMPL_H_
#define DALI_IMGCODEC_DECODERS_DECODER_IMPL_H_

#include "dali/imgcodec/image_decoder.h"
#include "dali/test/dali_test.h"
#include "dali/pipeline/util/thread_pool.h"

namespace dali {
namespace imgcodec {
namespace test {

class CpuDecoderTest : public ::testing::Test {
 public:

  CpuDecoderTest() : tp_(4, CPU_ONLY_DEVICE_ID, false, "Decoder test") {}

  // template<typename OutputType>
  // void Run(const std::string &image, const std::string &reference);

  Tensor<CPUBackend> ReadNumpy(const std::string &path);

 private:
  ThreadPool tp_;
  std::shared_ptr<ImageDecoderInstance> decoder_instance_;
};

}  // namespace test
}  // namespace imgcodec
}  // namespace dali

#endif  // DALI_IMGCODEC_DECODERS_DECODER_IMPL_H_