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
  if (static_ndim < 0) {
    VALUE_SWITCH(ndim, NDim, (0, 1, 2, 3, 4),
      (Convert<NDim>(out, out_strides, in, in_strides, size, NDim,
                     std::forward<ConvertFunc>(func));
      return;), ()
    );  // NOLINT
  }

  int64_t extent = size[0];
  int64_t in_stride = in_strides[0];
  int64_t out_stride = out_strides[0];

  if (static_ndim == 0) {
    func(out, in);
  } else if (static_ndim == 1) {
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
 * @brief Converts a data type of a vector.
 */
template <typename Out, typename In, int N>
inline vec<N, Out> ConvertSatNormVec(const vec<N, In> &in) {
  vec<N, Out> out = {};
  for (int i = 0; i < N; i++)
    out[i] = ConvertSatNorm<Out>(in[i]);
  return out;
}

template <typename Out, int out_channels, typename In, int in_channels>
struct ColorConversionBase {
  static In load(const In *in) {
    return *in;
  }

  vec<in_channels, In> vload(const In *in) const {
    vec<in_channels, In> v{};
    for (int i = 0; i < in_channels; i++)
      v[i] = in[i * in_channel_stride];
    return v;
  }

  static void store(Out *out, const Out &value) {
    *out = value;
  }

  void vstore(Out *out, const vec<out_channels, Out> &v) const {
    for (int i = 0; i < out_channels; i++)
      out[i * out_channel_stride] = v[i];
  }

  ptrdiff_t out_channel_stride, in_channel_stride;
};

template <typename Out, typename In, int channels>
struct ConvertPixelDType : ColorConversionBase<Out, channels, In, channels> {
  void operator()(Out *out, const In *in) const {
    auto v = this->vload(in);
    this-> vstore(out, ConvertSatNormVec<Out>(v));
  }
};


template <typename Out, typename In, DALIImageType out_format, DALIImageType in_format>
struct ConvertPixel;

template <typename Out, typename In>
struct ConvertPixel<Out, In, DALI_GRAY, DALI_GRAY> : ConvertPixelDType<Out, In, 1> {};

template <typename Out, typename In>
struct ConvertPixel<Out, In, DALI_RGB, DALI_RGB> : ConvertPixelDType<Out, In, 3> {};

template <typename Out, typename In>
struct ConvertPixel<Out, In, DALI_YCbCr, DALI_YCbCr> : ConvertPixelDType<Out, In, 3> {};

template <typename Out, typename In>
struct ConvertPixel<Out, In, DALI_GRAY, DALI_RGB> : ColorConversionBase<Out, 1, In, 3> {
  void operator()(Out *out, const In *in) const {
    auto rgb = this->vload(in);
    this->store(out, kernels::color::rgb_to_gray<Out, In>(rgb));
  }
};

template <typename Out, typename In>
struct ConvertPixel<Out, In, DALI_YCbCr, DALI_RGB> : ColorConversionBase<Out, 3, In, 3> {
  void operator()(Out *out, const In *in) const {
    auto rgb = this->vload(in);
    this->vstore(out, kernels::color::itu_r_bt_601::rgb_to_ycbcr<Out, In>(rgb));
  }
};

template <typename Out, typename In>
struct ConvertPixel<Out, In, DALI_YCbCr, DALI_GRAY> : ColorConversionBase<Out, 3, In, 1> {
  void operator()(Out *out, const In *in) const {
    auto gray = this->load(in);
    this->vstore(out, kernels::color::itu_r_bt_601::gray_to_ycbcr<Out, In>(gray));
  }
};
template <typename Out, typename In>
struct ConvertPixel<Out, In, DALI_RGB, DALI_GRAY> : ColorConversionBase<Out, 3, In, 1> {
  void operator()(Out *out, const In *in) const {
    auto gray = this->load(in);
    this->vstore(out, kernels::color::gray_to_rgb<Out, In>(gray));
  }
};

template <typename Out, typename In>
struct ConvertPixel<Out, In, DALI_RGB, DALI_YCbCr> : ColorConversionBase<Out, 3, In, 3> {
  void operator()(Out *out, const In *in) const {
    auto ycbcr = this->vload(in);
    this->vstore(out, kernels::color::itu_r_bt_601::ycbcr_to_rgb<Out, In>(ycbcr));
  }
};
template <typename Out, typename In>
struct ConvertPixel<Out, In, DALI_GRAY, DALI_YCbCr> : ColorConversionBase<Out, 1, In, 3> {
  void operator()(Out *out, const In *in) const {
    auto ycbcr = this->vload(in);
    this->store(out, kernels::color::itu_r_bt_601::ycbcr_to_gray<Out, In>(ycbcr));
  }
};


/**
 * @brief Converts an image stored in `in` and stores it in `out`.
 *
 * This is a wrapper for more generic variant of Convert. Based on image format and strides it
 * chooses an appropriate conversion function and runs the generic Convert with it.
 */
template <typename Out, typename In>
void Convert(Out *out, const int64_t *out_strides, int out_channel_dim, DALIImageType out_format,
             const In *in, const int64_t *in_strides, int in_channel_dim, DALIImageType in_format,
             const int64_t *size, int ndim) {
  // TODO(skarpinski) Support other layouts
  DALI_ENFORCE(out_channel_dim == ndim - 1 && in_channel_dim == ndim - 1,
    "Not implemented: currently only channels-last layout is supported");
  ptrdiff_t in_channel_stride = 1, out_channel_stride = 1;

  if (in_format == DALI_BGR) {
    // We will use RGB conversion, but we will load the pixel in reverse order.
    in_format = DALI_RGB;
    in += 2 * in_channel_stride;
    in_channel_stride = -in_channel_stride;
  }

  if (out_format == DALI_BGR) {
    // We will use RGB conversion, but we will store the pixel in reverse order.
    out_format = DALI_RGB;
    out += 2 * out_channel_stride;
    out_channel_stride = -out_channel_stride;
  }

  if (out_format == DALI_ANY_DATA) {
    // Conversion to DALI_ANY_DATA is a no-op
    out_format = in_format;
  }

  VALUE_SWITCH(out_format, OutFormat, (DALI_RGB, DALI_YCbCr, DALI_GRAY), (
    VALUE_SWITCH(in_format, InFormat, (DALI_RGB, DALI_YCbCr, DALI_GRAY), (
      auto func = ConvertPixel<Out, In, OutFormat, InFormat>{out_channel_stride, in_channel_stride};
      Convert(out, out_strides, in, in_strides, size, ndim - 1, func);
    ), throw std::logic_error(  // NOLINT
        make_string("Unsupported input format " , to_string(in_format))););  // NOLINT
  ), throw std::logic_error(  // NOLINT
      make_string("Unsupported output format " , to_string(out_format))););  // NOLINT
}

/**
 * @brief Converts an image stored in `in` and stores it in `out`.
 *
 * The function converts data type (normalizing) and color space.
 * When roi_start or roi_end is empty, it is assumed to be the lower bound and upport bound
 * of the spatial extent. Channel dimension must not be included in ROI specification.
 */
void DLL_PUBLIC Convert(
    SampleView<CPUBackend> out, TensorLayout out_layout, DALIImageType out_format,
    ConstSampleView<CPUBackend> in, TensorLayout in_layout, DALIImageType in_format,
    TensorShape<> roi_start, TensorShape<> roi_end);


}  // namespace imgcodec
}  // namespace dali


#endif  // DALI_IMGCODEC_UTIL_CONVERT_H_
