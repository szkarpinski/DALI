// Copyright (c) 2019-2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "dali/operators/image/color/colorize.h"

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 * This is just a "hello world" operator, which means it is redundant (see ColorTwist), incomplete,*
 * badly designed, poorly documented, not flexible and many more BUT it can make doggos pink which *
 * is cute. Things that were intentionally skipped to make stuff simple are marked with            *
 * "TODO(nobody) tag"                                                                              *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *  */

namespace dali {

// TODO(nobody) elaborate documentation
DALI_SCHEMA(Colorize)
    .DocStr(R"(Colorizes an RGB image with a specified color.)")
    .NumInput(1)
    .NumOutput(1)
    .AddOptionalArg<std::vector<float>>("color", "The color in RGB.", nullptr, true)
    .AddParent("ColorTransformBase");
    // TODO(nobody) optional "dtype" argument for specifying output type different than input's


bool ColorizeCpu::SetupImpl(std::vector<OutputDesc> &output_descs,
                            const workspace_t<CPUBackend> &ws) {
  output_descs.resize(1);
  auto &input = ws.template InputRef<CPUBackend>(0);

  DALI_ENFORCE(input.shape().sample_dim() >= 2, "At least 2D input expected.");

  TYPE_SWITCH(input.type(), type2id, DataType, (uint8_t, int16_t, int32_t, float, float16), (
      using Kernel = TheKernel<DataType>;
      kernel_manager_.Initialize<Kernel>();
      output_descs[0] = {input.shape(), input.type()};
  ), DALI_FAIL(make_string("Unsupported input type: ", input.type())))  // NOLINT

  for (int i = 0; i < input.shape().num_samples(); i++) {
    auto sample_shape = input.shape().tensor_shape_span(i);
    DALI_ENFORCE(sample_shape.back() == 3, "The pixel dimension must be 3");
  }

  return true;
}

void ColorizeCpu::RunImpl(workspace_t<CPUBackend> &ws) {
  const auto &input = ws.template InputRef<CPUBackend>(0);
  auto &output = ws.template OutputRef<CPUBackend>(0);
  output.SetLayout(input.GetLayout());
  auto out_shape = output.shape();
  auto& tp = ws.GetThreadPool();

  int nthreads = tp.NumThreads();

  TYPE_SWITCH(input.type(), type2id, DataType, (uint8_t, int16_t, int32_t, float, float16), (
    using Kernel = TheKernel<DataType>;
    kernel_manager_.template Resize<Kernel>(nthreads, nthreads);

    for (int sample_id = 0; sample_id < input.shape().num_samples(); sample_id++) {
      tp.AddWork([&, sample_id](int thread_id) {
        kernels::KernelContext ctx;
        auto tvin = view<const DataType, 3>(input[sample_id]);
        auto tvout = view<DataType, 3>(output[sample_id]);
        kernel_manager_.Run<Kernel>(thread_id, thread_id, ctx, tvout, tvin, M, vec3(0, 0, 0));
      }, out_shape.tensor_size(sample_id));
    }
  ), DALI_FAIL(make_string("Unsupported input type: ", input.type())))  // NOLINT
  tp.RunAll();
}

DALI_REGISTER_OPERATOR(Colorize, ColorizeCpu, CPU);

}  // namespace dali


