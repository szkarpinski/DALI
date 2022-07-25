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


// Stolen from numpy_loader

#include "dali/imgcodec/decoders/test/numpy_helper.h"
#include "dali/pipeline/data/tensor.h"
#include "dali/operators/reader/loader/numpy_loader.h"
#include "dali/kernels/transpose/transpose.h"
#include "dali/core/static_switch.h"
#include "dali/pipeline/data/views.h"

#define NUMPY_ALLOWED_TYPES \
  (bool, uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t, float, float16, \
  double)

namespace dali {
namespace imgcodec {
namespace test {

const TypeInfo &TypeFromNumpyStr(const std::string &format) {
  if (format == "u1") return TypeTable::GetTypeInfo<uint8_t>();
  if (format == "u2") return TypeTable::GetTypeInfo<uint16_t>();
  if (format == "u4") return TypeTable::GetTypeInfo<uint32_t>();
  if (format == "u8") return TypeTable::GetTypeInfo<uint64_t>();
  if (format == "i1") return TypeTable::GetTypeInfo<int8_t>();
  if (format == "i2") return TypeTable::GetTypeInfo<int16_t>();
  if (format == "i4") return TypeTable::GetTypeInfo<int32_t>();
  if (format == "i8") return TypeTable::GetTypeInfo<int64_t>();
  if (format == "f2") return TypeTable::GetTypeInfo<float16>();
  if (format == "f4") return TypeTable::GetTypeInfo<float>();
  if (format == "f8") return TypeTable::GetTypeInfo<double>();
  DALI_FAIL("Unknown Numpy type string");
}

inline void SkipSpaces(const char*& ptr) {
  while (::isspace(*ptr))
    ptr++;
}

template <size_t N>
void Skip(const char*& ptr, const char (&what)[N]) {
  DALI_ENFORCE(!strncmp(ptr, what, N - 1),
               make_string("Expected \"", what, "\" but got \"", ptr, "\""));
  ptr += N - 1;
}

template <size_t N>
bool TrySkip(const char*& ptr, const char (&what)[N]) {
  if (!strncmp(ptr, what, N-1)) {
    ptr += N - 1;
    return true;
  } else {
    return false;
  }
}

template <size_t N>
void SkipFieldName(const char*& ptr, const char (&name)[N]) {
  SkipSpaces(ptr);
  Skip(ptr, "'");
  Skip(ptr, name);
  Skip(ptr, "'");
  SkipSpaces(ptr);
  Skip(ptr, ":");
  SkipSpaces(ptr);
}

template <typename T = int64_t>
T ParseInteger(const char*& ptr) {
  char *out_ptr = const_cast<char*>(ptr);  // strtol takes a non-const pointer
  T value = static_cast<T>(strtol(ptr, &out_ptr, 10));
  DALI_ENFORCE(out_ptr != ptr, "Parse error: expected a number.");
  ptr = out_ptr;
  return value;
}

std::string ParseStringValue(const char*& input, char delim_start = '\'', char delim_end = '\'') {
  DALI_ENFORCE(*input++ == delim_start, make_string("Expected \'", delim_start, "\'"));
  std::string out;
  for (; *input != '\0'; input++) {
    if (*input == '\\') {
      switch (*++input) {
        case '\\':
          out += '\\';
          break;
        case '\'':
          out += '\'';
          break;
        case '\t':
          out += '\t';
          break;
        case '\n':
          out += '\n';
          break;
        case '\"':
          out += '\"';
          break;
        default:
          out += '\\';
          out += *input;
          break;
      }
    } else if (*input == delim_end) {
      break;
    } else {
      out += *input;
    }
  }
  DALI_ENFORCE(*input++ == delim_end, make_string("Expected \'", delim_end, "\'"));
  return out;
}

void ParseHeaderMetadata(NumpyHeaderMeta& target, const std::string &header) {
  target.shape = {};
  const char* hdr = header.c_str();
  SkipSpaces(hdr);
  Skip(hdr, "{");
  SkipFieldName(hdr, "descr");
  auto typestr = ParseStringValue(hdr);
    // < means LE, | means N/A, = means native. In all those cases, we can read
  bool little_endian = (typestr[0] == '<' || typestr[0] == '|' || typestr[0] == '=');
  DALI_ENFORCE(little_endian, "Big Endian files are not supported.");
  target.type_info = &TypeFromNumpyStr(typestr.substr(1));

  SkipSpaces(hdr);
  Skip(hdr, ",");
  SkipFieldName(hdr, "fortran_order");
  if (TrySkip(hdr, "True")) {
    target.fortran_order = true;
  } else if (TrySkip(hdr, "False")) {
    target.fortran_order = false;
  } else {
    DALI_FAIL("Failed to parse fortran_order field.");
  }
  SkipSpaces(hdr);
  Skip(hdr, ",");
  SkipFieldName(hdr, "shape");
  Skip(hdr, "(");
  SkipSpaces(hdr);
  while (*hdr != ')') {
    // ParseInteger already skips the leading spaces (strtol does).
    target.shape.shape.push_back(ParseInteger<int64_t>(hdr));
    SkipSpaces(hdr);
    DALI_ENFORCE(TrySkip(hdr, ",") || target.shape.size() > 1,
                 "The first number in a tuple must be followed by a comma.");
  }
  if (target.fortran_order) {
    // cheapest thing to do is to define the tensor in an reversed way
    std::reverse(target.shape.begin(), target.shape.end());
  }
}

void ParseHeader(FileStream *file, NumpyHeaderMeta& parsed_header) {
  // check if the file is actually a numpy file
  std::vector<uint8_t> token(11);
  int64_t nread = file->Read(token.data(), 10);
  DALI_ENFORCE(nread == 10, "Can not read header.");
  token[nread] = '\0';

  // check if heqder is too short
  std::string header = std::string(reinterpret_cast<char*>(token.data()));
  DALI_ENFORCE(header.find_first_of("NUMPY") != std::string::npos,
               "File is not a numpy file.");

  // extract header length
  uint16_t header_len = 0;
  memcpy(&header_len, &token[8], 2);
  DALI_ENFORCE((header_len + 10) % 16 == 0,
               "Error extracting header length.");

  // read header: the offset is a magic number
  int64 offset = 6 + 1 + 1 + 2;
  // the header_len can be 4GiB according to the NPYv2 file format
  // specification: https://numpy.org/neps/nep-0001-npy-format.html
  // while this allocation could be sizable, it is performed on the host.
  token.resize(header_len+1);
  file->SeekRead(offset);
  nread = file->Read(token.data(), header_len);
  DALI_ENFORCE(nread == header_len, "Can not read header.");
  token[header_len] = '\0';
  header = std::string(reinterpret_cast<const char*>(token.data()));
  DALI_ENFORCE(header.find('{') != std::string::npos, "Header is corrupted.");
  offset += header_len;
  file->SeekRead(offset);  // michalz: Why isn't it done when actually reading the payload?

  ParseHeaderMetadata(parsed_header, header);
  parsed_header.data_offset = offset;
}

static void TransposeHelper(SampleView<CPUBackend> output, ConstSampleView<CPUBackend> input) {
  int n_dims = input.shape().sample_dim();
  SmallVector<int, 6> perm;
  perm.resize(n_dims);
  for (int i = 0; i < n_dims; ++i)
    perm[i] = n_dims - i - 1;
  TYPE_SWITCH(input.type(), type2id, T, NUMPY_ALLOWED_TYPES, (
    kernels::TransposeGrouped(view<T>(output), view<const T>(input), make_cspan(perm));
  ), DALI_FAIL(make_string("Unsupported input type: ", input.type())));  // NOLINT
}

Tensor<CPUBackend> ReadNumpy(const std::string &path) {
  auto file = FileStream::Open(path, false, false);
  NumpyHeaderMeta meta;
  ParseHeader(file.get(), meta);
  file->SeekRead(meta.data_offset, SEEK_SET);

  Tensor<CPUBackend> data;
  data.Resize(meta.shape, meta.type());
  Index ret = file->Read(static_cast<uint8_t*>(data.raw_mutable_data()), meta.nbytes());
  DALI_ENFORCE(ret == (Index)meta.nbytes(), make_string("Failed to read file: ", path));

  if (meta.fortran_order) {
    Tensor<CPUBackend> transposed;
    transposed.Resize(data.shape(), data.type());
    SampleView<CPUBackend> input(data.raw_mutable_data(), data.shape(), data.type());
    SampleView<CPUBackend> output(transposed.raw_mutable_data(), transposed.shape(),
                                  transposed.type());
    TransposeHelper(output, input);
    return transposed;
  }
  return data;
}

}  // namespace test
}  // namespace imgcodec
}  // namespace dali