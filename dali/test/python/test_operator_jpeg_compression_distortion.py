# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from nvidia.dali import pipeline, pipeline_def
import nvidia.dali.fn as fn
import nvidia.dali.types as types
import nvidia.dali as dali
from random import shuffle
import numpy as np
from numpy.testing import assert_array_equal, assert_allclose
import os
import cv2

test_data_root = os.environ['DALI_EXTRA_PATH']
images_dir = os.path.join(test_data_root, 'db', 'single', 'tiff')
dump_images = False
dump_broken = False
sequence_length = 10

class InputImagesIter(object):
  def __init__(self, sequence_length):
    self.sequence_length = sequence_length
    with open(os.path.join(images_dir, 'image_list.txt')) as file:
      self.files = [line.rstrip() for line in file if line != '']
    shuffle(self.files)

  def _load_next(self):
    in_img = None
    # Skip input image if format isn't supported by OpenCV
    while in_img is None:
      filename, label = self.files[self.i].split(' ')
      in_img = cv2.imread(os.path.join(images_dir, filename))
      self.i = (self.i + 1) % len(self.files)
    # Convert to rgb, to match dali channel order
    rgb = cv2.cvtColor(in_img, cv2.COLOR_BGR2RGB)
    return rgb

  def __iter__(self):
    self.i = 0
    return self

  def __next__(self):
    first = self._load_next()
    seq = [first]
    for _ in range(self.sequence_length):
      img = self._load_next()
      if img.shape != first.shape:
        img = cv2.resize(img, (first.shape[1], first.shape[0]))
      seq.append(img)
    return np.stack(seq)

def _compare_to_cv_distortion(in_img, out_img, q, no):
  bgr = cv2.cvtColor(in_img, cv2.COLOR_RGB2BGR)
  encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), q]
  _, encoded_img = cv2.imencode('.jpg', bgr, params=encode_params)

  decoded_img_bgr = cv2.imdecode(encoded_img, cv2.IMREAD_COLOR)
  decoded_img = cv2.cvtColor(decoded_img_bgr, cv2.COLOR_BGR2RGB)

  diff = cv2.absdiff(out_img, decoded_img)
  diff_in_range = np.average(diff) < 5, f"Absolute difference with the reference is too big: {np.average(diff)}"

  if dump_images or (dump_broken and not diff_in_range):
    i, j = no
    cv2.imwrite(f"./reference_q{q}_sample{i}_{j}.bmp", cv2.cvtColor(decoded_img, cv2.COLOR_BGR2RGB))
    cv2.imwrite(f"./output_q{q}_sample{i}_{j}.bmp", cv2.cvtColor(out_img, cv2.COLOR_BGR2RGB))

  assert diff_in_range, f"Absolute difference with the reference is too big: {np.average(diff)}"

def _testimpl_jpeg_compression_distortion(batch_size, device, quality, layout):
  @pipeline_def(batch_size=batch_size, num_threads=3, device_id=0)
  def jpeg_distortion_pipe(device='cpu', quality=None):
    if layout == 'FHWC':
      iii = InputImagesIter(sequence_length)
      in_tensors = fn.external_source(source=iii, layout='FHWC', batch=False)
    else:
      encoded, _ = fn.readers.file(file_root=images_dir)
      in_tensors = fn.decoders.image(encoded, device='cpu')

    inputs = in_tensors.gpu() if device == 'gpu' else in_tensors
    if quality is None:
      quality = fn.random.uniform(range=[1, 99], dtype=types.INT32)
    out_tensors = fn.jpeg_compression_distortion(inputs, quality=quality)
    return (out_tensors, in_tensors, quality)

  pipe = jpeg_distortion_pipe(device=device, quality=quality,
                              batch_size=batch_size, num_threads=2, device_id=0)
  pipe.build()
  for _ in range(3):
    out = pipe.run()
    out_data = out[0].as_cpu() if device == 'gpu' else out[0]
    in_data = out[1]
    quality = out[2]
    for i in range(batch_size):
      out_tensor = np.array(out_data[i])
      in_tensor = np.array(in_data[i])
      q = int(np.array(quality[i]))
      if layout == 'FHWC':
        for j in range(in_tensor.shape[0]):
          _compare_to_cv_distortion(in_tensor[j], out_tensor[j], q, (i, j))
      else:
        _compare_to_cv_distortion(in_tensor, out_tensor, q, (i, 0))

def test_jpeg_compression_distortion():
  for batch_size in [1, 15]:
    for device in ['cpu', 'gpu']:
      for quality in [2, None, 50]:
        for layout in ['HWC', 'FHWC']:
          yield _testimpl_jpeg_compression_distortion, batch_size, device, quality, layout
