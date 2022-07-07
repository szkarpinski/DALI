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

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include "dali/imgcodec/decoders/opencv_fallback.h"
#include "dali/imgcodec/util/convert.h"

namespace dali {
namespace imgcodec {

inline DALIDataType CV2DaliType(int cv_type_id) {
  switch (cv_type_id) {
    case CV_8U:
      return DALI_UINT8;
    case CV_8S:
      return DALI_INT8;
    case CV_16U:
      return DALI_UINT16;
    case CV_16S:
      return DALI_INT16;
    case CV_32S:
      return DALI_INT32;
    case CV_32F:
      return DALI_FLOAT;
    case CV_64F:
      return DALI_FLOAT64;
    default:
      throw std::invalid_argument(make_string("Unexpected OpenCV type: ", cv_type_id));
  }
}

DecodeResult OpenCVDecoderInstance::Decode(SampleView<CPUBackend> out,
                                           ImageSource *in,
                                           DecodeParams opts) {
  int flags = 0;

  switch (opts.format) {
    case DALI_ANY_DATA:
      // Note: IMREAD_UNCHANGED always ignores orientation
      flags |= cv::IMREAD_UNCHANGED;
      break;
    case DALI_GRAY:
      flags |= cv::IMREAD_GRAYSCALE | cv::IMREAD_IGNORE_ORIENTATION;
      break;
    default:
      flags |= cv::IMREAD_COLOR | cv::IMREAD_IGNORE_ORIENTATION;
  }

  if (opts.dtype != DALI_UINT8)
    flags |= cv::IMREAD_ANYDEPTH;

  DecodeResult res;
  try {
    cv::Mat cvimg;
    if (in->Kind() == InputKind::Filename) {
      cvimg = cv::imread(in->Filename(), flags);
    } else {
      assert(in->Kind() == InputKind::HostMemory);
      auto *raw = static_cast<const uint8_t *>(in->RawData());
      cvimg = cv::imdecode(cv::_InputArray(raw, in->Size()), flags);
    }

    if (flags & cv::IMREAD_COLOR)  // TODO - move this to Convert
      cv::cvtColor(cvimg, cvimg, cv::COLOR_BGR2RGB);

    res.success = cvimg.ptr(0) != nullptr;
    if (res.success) {
      DALI_ENFORCE(cvimg.dims + 1 == out.shape().sample_dim(), make_string(
        "The decoded image has an unexpected number of spatial dimensions: ", cvimg.dims,
        "\nExpected: ", out.shape().sample_dim() - 1));

      DALIDataType type = CV2DaliType(cvimg.depth());

      TensorShape<> shape;
      shape.resize(cvimg.dims + 1);
      for (int d = 0; d < cvimg.dims; d++)
        shape[d] = cvimg.size[d];
      shape[cvimg.dims] = cvimg.channels();

      SampleView<CPUBackend> in(cvimg.ptr(0), shape, type);

      TensorShape<> roi_start, roi_end;

      TensorLayout layout = cvimg.dims == 3 ? "DHWC" : "HWC";

      Convert(out, layout, opts.format, in, layout, opts.format /* TODO - use actual format */,
              roi_start, roi_end);
    }
  } catch (...) {
    res.exception = std::current_exception();
    res.success = false;
  }

  return res;
}


}  // namespace imgcodec
}  // namespace dali
