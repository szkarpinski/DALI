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

#include <tiffio.h>
#include "dali/imgcodec/decoders/tiff_libtiff.h"
#include "dali/imgcodec/util/convert.h"
#include "dali/core/tensor_shape_print.h"

#define LIBTIFF_CALL_SUCCESS 1
#define LIBTIFF_CALL(call)                                \
  do {                                                    \
    int retcode = (call);                                 \
    DALI_ENFORCE(LIBTIFF_CALL_SUCCESS == retcode,         \
      "libtiff call failed with code "                    \
      + std::to_string(retcode) + ": " #call);            \
  } while (0)

namespace dali {
namespace imgcodec {

class DecoderHelper {
 private:
  std::shared_ptr<InputStream> stream;
  const void *buffer;
  size_t buffer_size;

 public:
  explicit DecoderHelper(ImageSource *in) : stream(in->Open()) {
    if (in->Kind() == InputKind::HostMemory) {
      buffer = in->RawData();
      buffer_size = in->Size();
    } else {
      buffer = nullptr;
    }
  }

  static tmsize_t read(thandle_t handle, void *buffer, tmsize_t n) {
    DecoderHelper *helper = reinterpret_cast<DecoderHelper *>(handle);
    return helper->stream->Read(buffer, n);
  }

  static tmsize_t write(thandle_t, void *, tmsize_t) {
    // Not used for decoding.
    return 0;
  }

  static toff_t seek(thandle_t handle, toff_t offset, int whence) {
    DecoderHelper *helper = reinterpret_cast<DecoderHelper *>(handle);
    helper->stream->SeekRead(offset, whence);
    return helper->stream->TellRead();
  }

  static int map(thandle_t handle, void **base, toff_t *size) {
    DecoderHelper *helper = reinterpret_cast<DecoderHelper *>(handle);
    *base = const_cast<void*>(helper->buffer);
    *size = helper->buffer_size;
    return 0;
  }

  static toff_t size(thandle_t handle) {
    DecoderHelper *helper = reinterpret_cast<DecoderHelper *>(handle);
    return helper->stream->Size();
  }

  static int close(thandle_t handle) {
    DecoderHelper *helper = reinterpret_cast<DecoderHelper *>(handle);
    delete helper;
    return 0;
  }
};

TIFF *openTiff(ImageSource *in) {
  if (in->Kind() == InputKind::Filename) {
    return TIFFOpen(in->Filename(), "r");
  } else {
    TIFFMapFileProc mapproc;
    if (in->Kind() == InputKind::HostMemory)
      mapproc = &DecoderHelper::map;
    else
      mapproc = nullptr;
    return TIFFClientOpen("", "r", reinterpret_cast<thandle_t>(new DecoderHelper(in)),
                          &DecoderHelper::read,
                          &DecoderHelper::write,
                          &DecoderHelper::seek,
                          &DecoderHelper::close,
                          &DecoderHelper::size,
                          mapproc,
                          /* unmap */ 0);
  }
}

DecodeResult LibTiffDecoderInstance::Decode(SampleView<CPUBackend> out,
                                           ImageSource *in,
                                           DecodeParams opts) {
  std::unique_ptr<TIFF, void (*)(TIFF *)> tiff = {openTiff(in), &TIFFClose};
  DALI_ENFORCE(tiff != nullptr, make_string("Unable to open TIFF image: ", in->SourceInfo()));

  uint32_t image_width, image_height, rows_per_strip;
  uint16_t in_channels, bit_depth, orientation, compression;
  bool is_tiled = TIFFIsTiled(tiff.get());
  LIBTIFF_CALL(TIFFGetField(tiff.get(), TIFFTAG_IMAGEWIDTH, &image_width));
  LIBTIFF_CALL(TIFFGetField(tiff.get(), TIFFTAG_IMAGELENGTH, &image_height));
  LIBTIFF_CALL(TIFFGetFieldDefaulted(tiff.get(), TIFFTAG_SAMPLESPERPIXEL, &in_channels));
  LIBTIFF_CALL(TIFFGetFieldDefaulted(tiff.get(), TIFFTAG_BITSPERSAMPLE, &bit_depth));
  LIBTIFF_CALL(TIFFGetFieldDefaulted(tiff.get(), TIFFTAG_ORIENTATION, &orientation));
  LIBTIFF_CALL(TIFFGetFieldDefaulted(tiff.get(), TIFFTAG_COMPRESSION, &compression));
  LIBTIFF_CALL(TIFFGetFieldDefaulted(tiff.get(), TIFFTAG_ROWSPERSTRIP, &rows_per_strip));

  // TODO(skarpinski) support other color formats
  DALI_ENFORCE(opts.format == DALI_RGB, "Only RGB output is supported");
  DALI_ENFORCE(in_channels == 3, "Only 3-channel input is supported");
  unsigned out_channels = 3;

  // TODO(skarpinski) support different types
  using InType = uint8_t;
  using OutType = uint8_t;
  DALI_ENFORCE(bit_depth == 8);

  // TODO(skarpinski) support tiled tiffs
  DALI_ENFORCE(!is_tiled);

  auto row_nbytes = TIFFScanlineSize(tiff.get());
  std::unique_ptr<InType, void(*)(void*)> row_buf{
    static_cast<InType *>(_TIFFmalloc(row_nbytes)), _TIFFfree};
  DALI_ENFORCE(row_buf.get() != nullptr, "Could not allocate memory");
  InType * const row_in  = row_buf.get();
  OutType * const img_out = out.mutable_data<OutType>();

  if (!opts.use_roi) {
    opts.roi.begin = {0, 0, 0};
    opts.roi.end = out.shape();
  }

  // Need to read sequentially since not all the images support random access
  // From: http://www.libtiff.org/man/TIFFReadScanline.3t.html
  // Compression algorithm does not support random access. Data was requested in a non-sequential
  // order from a file that uses a compression algorithm and that has RowsPerStrip greater than one.
  // That is, data in the image is stored in a compressed form, and with multiple rows packed into a
  // strip. In this case, the library does not support random access to the data. The data should
  // either be accessed sequentially, or the file should be converted so that each strip is made up
  // of one row of data.
  const bool allow_random_row_access = (compression == COMPRESSION_NONE || rows_per_strip == 1);
  if (!allow_random_row_access) {
    for (int64_t y = 0; y < opts.roi.begin[0]; y++) {
      LIBTIFF_CALL(TIFFReadScanline(tiff.get(), row_in, y, 0));
    }
  }

  uint64_t out_row_stride = opts.roi.shape()[1] * out_channels;

  for (int64_t roi_y = 0; roi_y < opts.roi.shape()[0]; roi_y++) {
    LIBTIFF_CALL(TIFFReadScanline(tiff.get(), row_in, opts.roi.begin[0] + roi_y, 0));

    // TODO(skarpinski) Color space conversion
    _TIFFmemcpy(img_out + (roi_y * out_row_stride), row_in + opts.roi.begin[1] * in_channels,
           opts.roi.shape()[1] * in_channels);
  }

  return {true, nullptr};
}


}  // namespace imgcodec
}  // namespace dali