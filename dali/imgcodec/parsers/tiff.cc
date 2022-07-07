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

#include "dali/imgcodec/parsers/tiff.h"
#include "dali/core/byte_io.h"

namespace dali {
namespace imgcodec {

constexpr int ENTRY_SIZE = 12;
constexpr int WIDTH_TAG = 256;
constexpr int HEIGHT_TAG = 257;
constexpr int SAMPLESPERPIXEL_TAG = 277;
constexpr int TYPE_WORD = 3;
constexpr int TYPE_DWORD = 4;

constexpr std::array<uint8_t, 4> le_header = {'I', 'I', 42, 0}, be_header = {'M', 'M', 0, 42};

template<typename T>
T TiffRead(InputStream& stream, bool is_little_endian) {
  if (is_little_endian) {
    return ReadValueLE<T>(stream);
  } else {
    return ReadValueBE<T>(stream);
  }
}

ImageInfo TiffParser::GetInfo(ImageSource *encoded) const {
  auto stream = encoded->Open();
  DALI_ENFORCE(stream->Size() >= 8);

  std::array<uint8_t, 4> header = stream->ReadOne<decltype(header)>();
  bool is_little_endian = (header == le_header);
  const auto ifd_offset = TiffRead<uint32_t>(*stream, is_little_endian);
  stream->SeekRead(ifd_offset, SEEK_SET);
  const auto entry_count = TiffRead<uint16_t>(*stream, is_little_endian);

  bool width_read = false, height_read = false, nchannels_read = false;
  int64_t width, height, nchannels;
  for (int entry_idx = 0;
       entry_idx < entry_count && !(width_read && height_read && nchannels_read);
       entry_idx++) {
    const auto entry_offset = ifd_offset + sizeof(uint16_t) + entry_idx * ENTRY_SIZE;
    stream->SeekRead(entry_offset, SEEK_SET);
    const auto tag_id = TiffRead<uint16_t>(*stream, is_little_endian);
    if (tag_id == WIDTH_TAG || tag_id == HEIGHT_TAG || tag_id == SAMPLESPERPIXEL_TAG) {
      const auto value_type = TiffRead<uint16_t>(*stream, is_little_endian);
      const auto value_count = TiffRead<uint32_t>(*stream, is_little_endian);
      DALI_ENFORCE(value_count == 1);

      int64_t value;
      if (value_type == TYPE_WORD) {
        value = TiffRead<uint16_t>(*stream, is_little_endian);
      } else if (value_type == TYPE_DWORD) {
        value = TiffRead<uint32_t>(*stream, is_little_endian);
      } else {
        DALI_FAIL("Couldn't read TIFF image dims.");
      }

      if (tag_id == WIDTH_TAG) {
        width = value;
        width_read = true;
      } else if (tag_id == HEIGHT_TAG) {
        height = value;
        height_read = true;
      } else if (tag_id == SAMPLESPERPIXEL_TAG) {
        nchannels = value;
        nchannels_read = true;
      }
    }
  }

  DALI_ENFORCE(width_read && height_read && nchannels_read,
    "TIFF image dims haven't been read properly");

  ImageInfo info;
  info.shape = {height, width, nchannels};
  return info;
}

bool TiffParser::CanParse(ImageSource *encoded) const {
  std::array<uint8_t, 4> header;
  if (encoded->Kind() == InputKind::HostMemory) {
    if (encoded->Size() < sizeof(header))
      return false;
    std::copy_n(encoded->RawData<char>(), sizeof(header), header.begin());
  } else {
    auto stream = encoded->Open();
    if (stream->Size() < sizeof(header))
      return false;
    header = stream->ReadOne<decltype(header)>();
  }
  return (header == le_header || header == be_header);
}

}  // namespace imgcodec
}  // namespace dali
