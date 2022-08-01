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

#ifndef DALI_IMGCODEC_UTIL_CONVERT_H_
#define DALI_IMGCODEC_UTIL_CONVERT_H_

#include <utility>
#include "dali/core/common.h"
#include "dali/core/convert.h"
#include "dali/core/static_switch.h"
#include "dali/core/geom/vec.h"
#include "dali/pipeline/data/backend.h"
#include "dali/pipeline/data/sample_view.h"
#include "dali/kernels/imgproc/color_manipulation/color_space_conversion_impl.h"

namespace dali {
namespace imgcodec {

/**
 * @brief Applies a conversion function `func` to the input data
 *
 * The data is strided - even the innermost dimension can have a non-unit stride.
 * `func` takes a pointer to output and input pointers; it can have some context (but not state),
 * to facilitate color space conversion with strided channels.
 */
template <int static_ndim = -1, typename Out, typename In, typename ConvertFunc>
void Convert(Out *out, const int64_t *out_strides,
             const In *in, const int64_t *in_strides,
             const int64_t *size, int ndim,
             ConvertFunc &&func) {
  if constexpr (static_ndim < 0) {
    VALUE_SWITCH(ndim, NDim, (0, 1, 2, 3, 4),
      (Convert<NDim>(out, out_strides, in, in_strides, size, NDim,
                     std::forward<ConvertFunc>(func));
      return;), ()
    );  // NOLINT
  }

  int64_t extent = size[0];
  int64_t in_stride = in_strides[0];
  int64_t out_stride = out_strides[0];

  if constexpr (static_ndim == 0) {
    func(out, in);
  } else if constexpr (static_ndim == 1) {  // NOLINT - if constexpr not recognized
    for (int64_t i = 0; i < extent; i++) {
      func(out + i * out_stride, in + i * in_stride);
    }
  } else {
    assert(ndim != 1 && "This should go with the static ndim codepath");
    for (int64_t i = 0; i < extent; i++) {
      const int next_ndim = static_ndim < 0 ? -1 : static_ndim - 1;
      Convert<next_ndim>(out + i * out_stride, out_strides + 1,
                         in + i * in_stride, in_strides + 1,
                         size + 1, ndim - 1,
                         std::forward<ConvertFunc>(func));
    }
  }
}

/**
 * @brief A functor for converting between color spaces.
 *
 * It reads the input data from a tensor, passes it to user-provided conversion function, and
 * then stores the result. Both the argument and return value of the conversion function can
 * either be a scalar or a vector.
 */
template <typename FuncIn, typename FuncOut>
struct ConvertColorSpace {
  using FuncType = FuncOut(*)(FuncIn);

  using InVec = typename std::conditional<is_vec<FuncIn>::value, FuncIn, vec<1, FuncIn>>::type;
  using OutVec = typename std::conditional<is_vec<FuncOut>::value, FuncOut, vec<1, FuncOut>>::type;
  using In = typename InVec::element_t;
  using Out = typename OutVec::element_t;

  void load(In& target, const In *in) {
    target = *in;
  }

  void load(InVec& target, const In *in) {
    for (int i = 0; i < target.size(); i++)
      target[i] = in[i * in_channel_stride];
  }

  void store(Out *out, Out source) {
    *out = source;
  }

  void store(Out *out, const OutVec& source) {
    for (int i = 0; i < source.size(); i++)
      out[i * out_channel_stride] = source[i];
  }

  void operator()(Out *out, const In *in) {
    FuncIn func_in;
    load(func_in, in);
    store(out, func(func_in));
  }

  ConvertColorSpace(FuncType func,
                    ptrdiff_t out_channel_stride = 1, ptrdiff_t in_channel_stride = 1)
    : func(func), out_channel_stride(out_channel_stride), in_channel_stride(in_channel_stride) {}

  FuncType func;
  ptrdiff_t out_channel_stride, in_channel_stride;
};

/**
 * @brief Converts a data type of a single-channel value.
 */
template <typename Out, typename In>
inline void ConvertDType(Out *out, const In *in) {
  *out = ConvertSatNorm<Out>(*in);
}


template <typename Out, typename In>
void Convert(Out *out, const int64_t *out_strides, int out_channel_dim, DALIImageType out_format,
             const In *in, const int64_t *in_strides, int in_channel_dim, DALIImageType in_format,
             const int64_t *size, int ndim) {
  DALI_ENFORCE(out_channel_dim == ndim - 1 && in_channel_dim == ndim - 1,
    "Not implemented: currently only channels-last layout is supported");

  std::function<void(Out *, const In *)> convert_func;
  if (in_format == out_format) {
    convert_func = &ConvertDType<Out, In>;
  } else if (in_format == DALI_RGB && out_format == DALI_GRAY) {
    convert_func = ConvertColorSpace(kernels::color::rgb_to_gray<Out, In>, 1, 1);
  } else if (in_format == DALI_GRAY && out_format == DALI_RGB) {
    convert_func = ConvertColorSpace(kernels::color::gray_to_rgb<Out, In>, 1, 1);
  } else {
    DALI_FAIL(make_string("Not implemented: conversion from ", to_string(in_format), " to ",
              to_string(out_format), " is not supported"));
  }

  int ndim_new = (in_format == out_format ? ndim : ndim - 1);
  Convert(out, out_strides, in, in_strides, size, ndim_new, convert_func);
}

/**
 * @brief Converts an image stored in `in` and stores it in `out`.
 *
 * The function converts data type (normalizing) and color space.
 * When roi_start or roi_end is empty, it is assumed to be the lower bound and upport bound
 * of the spatial extent. Channel dimension must not be included in ROI specification.
 */
void Convert(SampleView<CPUBackend> out, TensorLayout out_layout, DALIImageType out_format,
             ConstSampleView<CPUBackend> in, TensorLayout in_layout, DALIImageType in_format,
             TensorShape<> roi_start, TensorShape<> roi_end);


}  // namespace imgcodec
}  // namespace dali


#endif  // DALI_IMGCODEC_UTIL_CONVERT_H_
