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
// limitations under the License.

#include <vector>

#include "dali/core/static_switch.h"
#include "dali/kernels/kernel_manager.h"
#include "dali/kernels/imgproc/pointwise/linear_transformation_cpu.h"
#include "dali/pipeline/data/views.h"
#include "dali/pipeline/operator/common.h"
#include "dali/pipeline/operator/operator.h"
#include "dali/core/format.h"
#include "dali/core/geom/vec.h"
#include "dali/core/geom/mat.h"

#ifndef DALI_OPERATORS_IMAGE_COLOR_COLORIZE_H_
#define DALI_OPERATORS_IMAGE_COLOR_COLORIZE_H_

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 * This is just a "hello world" operator, which means it is redundant (see ColorTwist), incomplete,*
 * badly designed, poorly documented, not flexible and many more BUT it can make doggos pink which *
 * is cute. Things that were intentionally skipped to make stuff simple are marked with            *
 * "TODO(nobody) tag"                                                                              *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *  */

namespace dali {

template <typename Backend>
class ColorizeBase : public Operator<Backend> {
 public:
  ~ColorizeBase() override = default;

 protected:
  mat3x3 M;
  const vec<3, float> zeros = {0, 0, 0};

  explicit ColorizeBase(const OpSpec &spec) : Operator<Backend>(spec) {
    if (spec.ArgumentDefined("color")) {
      color_ = spec.GetRepeatedArgument<float>("color");
    } else {
      color_ = {1., 1., 1.};
    }

    DALI_ENFORCE(color_.size() == 3, "Color must be a vector of length 3");

    this->M = {{{color_[0]/3, color_[0]/3, color_[0]/3},
                {color_[1]/3, color_[1]/3, color_[1]/3},
                {color_[2]/3, color_[2]/3, color_[2]/3}}};
  }

  bool CanInferOutputs() const override { return true; }

  std::vector<float> color_;
  kernels::KernelManager kernel_manager_;
};

class ColorizeCpu : public ColorizeBase<CPUBackend> {
 public:
  explicit ColorizeCpu(const OpSpec &spec) : ColorizeBase(spec) {}
  ~ColorizeCpu() override = default;

 protected:
  template<typename DataType>
  using TheKernel = kernels::LinearTransformationCpu<DataType, DataType, 3, 3, 3>;

  using ColorizeBase<CPUBackend>::M;

  bool SetupImpl(std::vector<OutputDesc> &output_descs, const workspace_t<CPUBackend> &ws) override;
  void RunImpl(workspace_t<CPUBackend> &ws) override;
};

class ColorizeGpu : public ColorizeBase<GPUBackend> {
 public:
  explicit ColorizeGpu(const OpSpec &spec) : ColorizeBase(spec) {}
  ~ColorizeGpu() override = default;

 protected:
  std::vector<mat3x3> tmatrices_;

  using ColorizeBase<GPUBackend>::M;

  bool SetupImpl(std::vector<OutputDesc> &output_descs, const workspace_t<GPUBackend> &ws) override;
  void RunImpl(workspace_t<GPUBackend> &ws) override;
};

}  // namespace dali
#endif  // DALI_OPERATORS_IMAGE_COLOR_COLORIZE_H_
