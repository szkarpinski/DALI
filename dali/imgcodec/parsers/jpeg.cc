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

#include "dali/imgcodec/parsers/jpeg.h"
#include "dali/core/byte_io.h"

namespace dali {
namespace imgcodec {

using jpeg_marker_t = std::array<uint8_t, 2>;

constexpr jpeg_marker_t sos_marker = {0xff, 0xda};
constexpr jpeg_marker_t soi_marker = {0xff, 0xd8};
constexpr jpeg_marker_t eoi_marker = {0xff, 0xd9};

bool isValidMarker(const jpeg_marker_t &marker) {
  return marker[0] == 0xff && marker[1] != 0x00;
}

bool isSofMarker(const jpeg_marker_t &marker) {
  if (!isValidMarker(marker) || marker[1] < 0xc0 || marker[1] > 0xcf) return false;
  return (marker[1] != 0xc4 && marker[1] != 0xc8 && marker[1] != 0xcc);
}

ImageInfo JpegParser::GetInfo(ImageSource *encoded) const {
  ImageInfo info{};
  auto stream = encoded->Open();

  jpeg_marker_t first_marker = stream->ReadOne<jpeg_marker_t>();
  DALI_ENFORCE(first_marker == soi_marker);

  bool read_shape = false;
  while (!read_shape) {
    auto marker = stream->ReadOne<jpeg_marker_t>();

    DALI_ENFORCE(isValidMarker(marker));
    if (marker == sos_marker) break;

    uint16_t size = ReadValueBE<uint16_t>(*stream);
    ptrdiff_t next_marker_offset = stream->TellRead() - 2 + size;
    if (isSofMarker(marker)) {
      stream->Skip(1);  // Skip the precision field
      auto height = ReadValueBE<uint16_t>(*stream);
      auto width = ReadValueBE<uint16_t>(*stream);
      auto nchannels = stream->ReadOne<uint8_t>();
      info.shape = {height, width, nchannels};
      read_shape = true;
    }
    stream->SeekRead(next_marker_offset, SEEK_SET);
  }
  if (!read_shape)
    DALI_FAIL("Couldn't read dims of JPEG image.");
  return info;
}

bool JpegParser::CanParse(ImageSource *encoded) const {
  jpeg_marker_t first_marker;
  return (ReadHeader(first_marker.data(), encoded, first_marker.size()) == first_marker.size() &&
          first_marker == soi_marker);
}

}  // namespace imgcodec
}  // namespace dali
