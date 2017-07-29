# Copyright 2015 Google Inc. All Rights Reserved.
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
# ==============================================================================

"""Routine for decoding the CIFAR-10 binary file format."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
"""
这个文件主要包括以下的函数

函数名 功能 
read_cifar10(filename_queue) 从filename_queue中读取二进制数据，构造成样本数据 
_generate_image_and_label_batch() 构造batch_size样本集 
distorted_inputs(data_dir, batch_size) 将样本数据进行预处理，构造成训练数据 
inputs(eval_data, data_dir, batch_size) 将样本数据进行预处理，构造成测试数据

read_cifar10() 和 _generate_image_and_label_batch()是供distorted_inputs()和input() 
调用的子函数

该文件主要做了以下三件事情：

1、 从二进制文件中读取数据，每次读取一个样本，并将二进制数据解析为tensor类型数据 
2、 对1产生的样本数据进行预处理 
3、 构造样本队列，将2产生的样本插入队列，每次运行distorted_inputs()和input()，就从队列中取出batch_size大小的数据集

下面详细说一下这个文件代码中几个比较难懂的地方。

输入数据的构造

我觉得这部分对于没有接触过的人来说，还是有一定的难度的。我就是属于之前不了解的，所以花了好几天的时间才研究明白。

之前的教程中，tensorflow的模型输入都是使用placeholder来实现的，但是这种情况不适用于大数据量的情况。因此本篇教程介绍了tensorflow的另一种数据读取机制，从文件中读取数据，这是一种更通用的方法。

大体的实现思路是，让tensorflow单独创建一个queue runner线程，它负责从文件中读取样本数据，并将其装载到一个队列中。我们只需要开启这个线程，在需要数据时从队列中获取想要的size的数据集就可以了，队列数据的装载由该线程自动实现的。

图像数据的预处理

原始图片经过了部分预处理之后，才送入模型进行训练或评估。

原始的图片尺寸为32*32的像素尺寸，主要的预处理是两步

1、 首先将其裁剪为24*24像素大小的图片，其中训练集是随机裁剪，测试集是沿中心裁 
2、 将图片进行归一化，变为0均值，1方差

其中为了增加样本量，我们还对训练集增加如下的预处理

1、 随机的对图片进行由左到右的翻转 
2、 随机的改变图片的亮度 
3、 随机的改变图片的对比度

关于tf.train.shuffle_batch 中的参数 shuffle、min_after_dequeue

shuffle的作用在于指定是否需要随机打乱样本的顺序，一般作用于训练阶段，提高鲁棒性。 
1、当shuffle = false时，每次dequeue是从队列中按顺序取数据，遵从先入先出的原则 
2、当shuffle = true时，每次从队列中dequeue取数据时，不再按顺序，而是随机的，所以打乱了样本的原有顺序。

shuffle还要配合参数min_after_dequeue使用才能发挥作用。 
这个参数min_after_dequeue的意思是队列中，做dequeue（取数据）的操作后，queue runner线程要保证队列中至少剩下min_after_dequeue个数据。 
如果min_after_dequeue设置的过少，则即使shuffle为true，也达不到好的混合效果。

这个地方可能不太好理解，我尝试解释一下吧，但可能解释的不太好。 
假设你有一个队列，现在里面有m个数据，你想要每次随机从队列中取n个数据，则代表先混合了m个数据，再从中取走n个。 
当第一次取走n个后，队列就变为m-n个数据； 
当你下次再想要取n个时，假设队列在此期间插进来了k个数据，则现在的队列中有 
(m-n+k)个数据，则此时会从混合的(m-n+k)个数据中随机取走n个，。如果队列填充的速度比较慢，k就比较小，那你取出来的n个数据只是与周围很小的一部分(m-n+k)个数据进行了混合。

因为我们的目的肯定是想尽最大可能的混合数据，因此设置min_after_dequeue，可以保证每次dequeue后都有足够量的数据填充尽队列，保证下次dequeue时可以很充分的混合数据。

但是min_after_dequeue也不能设置的太大，这样会导致队列填充的时间变长，尤其是在最初的装载阶段，会花费比较长的时间。
"""


import os

import tensorflow.python.platform
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

from tensorflow.python.platform import gfile

# Process images of this size. Note that this differs from the original CIFAR
# image size of 32 x 32. If one alters this number, then the entire model
# architecture will change and any model would need to be retrained.

#裁剪图片大小
IMAGE_SIZE = 24

# Global constants describing the CIFAR-10 data set.
NUM_CLASSES = 10
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 10000


def read_cifar10(filename_queue):
  """Reads and parses examples from CIFAR10 data files.
  #从CIFAR10数据文件中读并且解析出一个样本
    
  Recommendation: if you want N-way read parallelism, call this function
  N times.  This will give you N independent Readers reading different
  files & positions within those files, which will give better mixing of
  examples.

  Args:
    filename_queue: A queue of strings with the filenames to read from.
                    一个string队列，里面是文件名字

  Returns:
    An object representing a single example, with the following fields:
      height: number of rows in the result (32)
      width: number of columns in the result (32)
      depth: number of color channels in the result (3)
      key: a scalar string Tensor describing the filename & record number
        for this example.
      label: an int32 Tensor with the label in the range 0..9.
      uint8image: a [height, width, depth] uint8 Tensor with the image data
  """

  class CIFAR10Record(object):
    pass
  result = CIFAR10Record()

  # Dimensions of the images in the CIFAR-10 dataset.
  # See http://www.cs.toronto.edu/~kriz/cifar.html for a description of the
  # input format.
  label_bytes = 1  # 2 for CIFAR-100
  result.height = 32
  result.width = 32
  result.depth = 3
  image_bytes = result.height * result.width * result.depth   #32*32*3
  # Every record consists of a label followed by the image, with a
  # fixed number of bytes for each.
  #每条记录包含一个label和一个图片数据，大小 32*32*3+1  bytes
  record_bytes = label_bytes + image_bytes

  # Read a record, getting filenames from the filename_queue.  No
  # header or footer in the CIFAR-10 format, so we leave header_bytes
  # and footer_bytes at their default of 0.
  #从一个文件队列中取一个filename，然后读出固定长度
  reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
  result.key, value = reader.read(filename_queue)

  # Convert from a string to a vector of uint8 that is record_bytes long.
  #将读出的固定长度的数据转化为unit8类型的数组
  record_bytes = tf.decode_raw(value, tf.uint8)

  # The first bytes represent the label, which we convert from uint8->int32.
  #第一个byte代表label，转化为int32
  result.label = tf.cast(
          tf.slice(record_bytes, [0], [label_bytes]), tf.int32)

  # The remaining bytes after the label represent the image, which we reshape
  # from [depth * height * width] to [depth, height, width].
  #剩余的代表一张图片数据，并且reshape 为 [depth, height, width].
  depth_major = tf.reshape(tf.slice(record_bytes, [label_bytes], [image_bytes]),
                           [result.depth, result.height, result.width])
  # Convert from [depth, height, width] to [height, width, depth].
  #转置， [depth, height, width] 到 [height, width, depth]
  result.uint8image = tf.transpose(depth_major, [1, 2, 0])

  return result


def _generate_image_and_label_batch(image, label, min_queue_examples,
                                    batch_size):
  """Construct a queued batch of images and labels.
  #构造一个batch的数据（image,label）

  Args:
    image: 3-D Tensor of [height, width, 3] of type.float32.
    label: 1-D Tensor of type.int32
    min_queue_examples: int32, minimum number of samples to retain
      in the queue that provides of batches of examples.
      int32,队列中保存的最小的样本数量，以方便从中取出一个batch
    batch_size: Number of images per batch.

  Returns:
    images: Images. 4D tensor of [batch_size, height, width, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.
  """
  # Create a queue that shuffles the examples, and then
  # read 'batch_size' images + labels from the example queue.
  #创建一个队列用来shuffle样本，并从中取出一个batch大小的样本（image,label)
  num_preprocess_threads = 16
  # min_after_dequeue defines how big a buffer we will randomly sample
  #   from -- bigger means better shuffling but slower start up and more
  #   memory used.
  #min_after_dequeue定义了一个buffer，我们从中随机抽样，太大使得混合更均匀，但是会造成更大的时间和内存开销
  # capacity must be larger than min_after_dequeue and the amount larger
  #   determines the maximum we will prefetch.  Recommendation:
  #   min_after_dequeue + (num_threads + a small safety margin) * batch_size
  #capacity必须大于min_after_dequeue，推荐min_after_dequeue + (num_threads + a small safety margin) * batch_size
  images, label_batch = tf.train.shuffle_batch(
      [image, label],
      batch_size=batch_size,
      num_threads=num_preprocess_threads,
      capacity=min_queue_examples + 3 * batch_size,
      min_after_dequeue=min_queue_examples)

  # Display the training images in the visualizer.
  #展示训练图片
  tf.summary.image('images', images)

  return images, tf.reshape(label_batch, [batch_size])


def distorted_inputs(data_dir, batch_size):
  """Construct distorted input for CIFAR training using the Reader ops.

  Args:
    data_dir: Path to the CIFAR-10 data directory.
    batch_size: Number of images per batch.

  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.
  """
  filenames = [os.path.join(data_dir, 'data_batch_%d.bin' % i)
               for i in xrange(1, 6)]
  for f in filenames:
    if not gfile.Exists(f):
      raise ValueError('Failed to find file: ' + f)

  # Create a queue that produces the filenames to read.
  #创建一个队列，其中存放shuffle过的文件名
  filename_queue = tf.train.string_input_producer(filenames)

  # Read examples from files in the filename queue.
  read_input = read_cifar10(filename_queue)
  reshaped_image = tf.cast(read_input.uint8image, tf.float32)

  height = IMAGE_SIZE
  width = IMAGE_SIZE

  # Image processing for training the network. Note the many random
  # distortions applied to the image.

  # Randomly crop a [height, width] section of the image.
#随机裁剪
  distorted_image = tf.random_crop(reshaped_image, [height, width,3])

  # Randomly flip the image horizontally.
  #随机翻转
  distorted_image = tf.image.random_flip_left_right(distorted_image)

  # Because these operations are not commutative, consider randomizing
  # randomize the order their operation.
  #随机变换亮度
  distorted_image = tf.image.random_brightness(distorted_image,
                                               max_delta=63)
  #随机变换对比度
  distorted_image = tf.image.random_contrast(distorted_image,
                                             lower=0.2, upper=1.8)

  # Subtract off the mean and divide by the variance of the pixels.
  #减去均值，除以方差（近似的白化处理）
  float_image = tf.image.per_image_standardization(distorted_image)

  # Ensure that the random shuffling has good mixing properties.
  min_fraction_of_examples_in_queue = 0.4
  min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN *
                           min_fraction_of_examples_in_queue)
  print ('Filling queue with %d CIFAR images before starting to train. '
         'This will take a few minutes.' % min_queue_examples)

  # Generate a batch of images and labels by building up a queue of examples.
  return _generate_image_and_label_batch(float_image, read_input.label,
                                         min_queue_examples, batch_size)


def inputs(eval_data, data_dir, batch_size):
  """Construct input for CIFAR evaluation using the Reader ops.
  #构造验证数据集

  Args:
    eval_data: bool, indicating if one should use the train or eval data set.
            bool类型，表明
    data_dir: Path to the CIFAR-10 data directory.
    batch_size: Number of images per batch.

  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.
  """
  if not eval_data:
    filenames = [os.path.join(data_dir, 'data_batch_%d.bin' % i)
                 for i in xrange(1, 6)]
    num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
  else:
    filenames = [os.path.join(data_dir, 'test_batch.bin')]
    num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_EVAL

  for f in filenames:
    if not gfile.Exists(f):
      raise ValueError('Failed to find file: ' + f)

  # Create a queue that produces the filenames to read.
  filename_queue = tf.train.string_input_producer(filenames)

  # Read examples from files in the filename queue.
  read_input = read_cifar10(filename_queue)
  reshaped_image = tf.cast(read_input.uint8image, tf.float32)

  height = IMAGE_SIZE
  width = IMAGE_SIZE

  # Image processing for evaluation.
  # Crop the central [height, width] of the image.
  #用中心的数据
  resized_image = tf.image.resize_image_with_crop_or_pad(reshaped_image,
                                                         width, height)

  # Subtract off the mean and divide by the variance of the pixels.
  #标准化
  float_image = tf.image.per_image_standardization(resized_image)

  # Ensure that the random shuffling has good mixing properties.
  min_fraction_of_examples_in_queue = 0.4
  min_queue_examples = int(num_examples_per_epoch *
                           min_fraction_of_examples_in_queue)

  # Generate a batch of images and labels by building up a queue of examples.
  return _generate_image_and_label_batch(float_image, read_input.label,
                                         min_queue_examples, batch_size)
