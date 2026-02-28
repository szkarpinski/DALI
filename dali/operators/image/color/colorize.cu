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
#include "dali/kernels/imgproc/pointwise/linear_transformation_gpu.h"

  /* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 * This is just a "hello world" operator, which means it is redundant (see ColorTwist), incomplete,*
 * badly designed, poorly documented, not flexible and many more BUT it can make doggos pink which *
 * is cute. Things that were intentionally skipped to make stuff simple are marked with            *
 * "TODO(nobody) tag"                                                                              *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *  */

namespace dali {

namespace {
  template<typename DataType>
  using TheKernel = kernels::LinearTransformationGpu<DataType, DataType, 3, 3, 2>;
}

DALI_REGISTER_OPERATOR(Colorize, ColorizeGpu, GPU);

bool ColorizeGpu::SetupImpl(std::vector<OutputDesc> &output_descs,
                            const workspace_t<GPUBackend> &ws) {
  output_descs.resize(1);
  auto &input = ws.template InputRef<GPUBackend>(0);

  DALI_ENFORCE(input.shape().sample_dim() >= 2, "At least 2D input expected.");

  TYPE_SWITCH(input.type(), type2id, DataType, (uint8_t, int16_t, int32_t, float, float16), (

      tmatrices_ = std::vector<mat3x3>(input.shape().num_samples(), M); //TODO ASK rly?

      using Kernel = TheKernel<DataType>;
      kernel_manager_.Initialize<Kernel>();
      kernel_manager_.Resize<Kernel>(1, 1);
      kernels::KernelContext ctx;
      ctx.gpu.stream = ws.stream();

      auto tvin = view<const DataType, 3>(input);
      kernel_manager_.Setup<Kernel>(0, ctx, tvin, make_cspan(tmatrices_));
      output_descs[0] = {input.shape(), input.type()};
  ), DALI_FAIL(make_string("Unsupported input type: ", input.type())))  // NOLINT

  for (int i = 0; i < input.shape().num_samples(); i++) {
    auto sample_shape = input.shape().tensor_shape_span(i);
    DALI_ENFORCE(sample_shape.back() == 3, "The pixel dimension must be 3");
  }

  return true;
}

void ColorizeGpu::RunImpl(workspace_t<GPUBackend> &ws) {
  const auto &input = ws.template InputRef<GPUBackend>(0);
  auto &output = ws.template OutputRef<GPUBackend>(0);
  output.SetLayout(input.GetLayout());
  auto out_shape = output.shape();

  TYPE_SWITCH(input.type(), type2id, DataType, (uint8_t, int16_t, int32_t, float, float16), (
    using Kernel = TheKernel<DataType>;
    kernels::KernelContext ctx;
    ctx.gpu.stream = ws.stream();
    auto tvin = view<const DataType, 3>(input);
    auto tvout = view<DataType, 3>(output);

    kernel_manager_.Run<Kernel>(ws.thread_idx(), 0, ctx, tvout, tvin, make_cspan(tmatrices_));
  ), DALI_FAIL(make_string("Unsupported input type: ", input.type())))  // NOLINT
}

}  // namespace dali


