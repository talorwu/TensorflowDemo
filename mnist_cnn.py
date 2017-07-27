import tensorflow as tf
import numpy as np
import tensorflow.examples.tutorials.mnist.input_data
#初始化W
def weight_variable(shape):
    #正态分布，但是只保留[mean-2*stddev,mean+2*stddev]
    initial = tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)
#初始化b
def bias_variable(shape):
    initial = tf.constant(0.1,shape=shape)
    return tf.Variable(initial)

#卷积操作
def conv2d(x,W):
    #padding='SAME'表示卷积之后大小不变,strides是步长
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')
#池化操作
def max_pool_2x2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

if __name__ == "__main__":
    mnist = tensorflow.examples.tutorials.mnist.input_data.read_data_sets('MNIST_data', one_hot=True)
    sess = tf.InteractiveSession()

    x = tf.placeholder("float", shape=[None, 784])
    y_ = tf.placeholder("float", shape=[None, 10])

    #卷积层第一层，32个核，patch大小[5,5]
    W_conv1 = weight_variable([5,5,1,32])
    b_conv1 = bias_variable([32])

    #x重新规定性状,-1代表系统可以指定这一维大小，这里相当于batch的大小
    x_image = tf.reshape(x,[-1,28,28,1])

    #卷积和pooling操作
    h_conv1 = tf.nn.relu(conv2d(x_image,W_conv1)+b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    #卷积层第二层,64个核，patch大小[5,5],上一层32个通道
    W_conv2 = weight_variable([5,5,32,64])
    b_conv2 = bias_variable([64])

    h_conv2 = tf.nn.relu(conv2d(h_pool1,W_conv2)+b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    #此时图像大小为7*7
    #添加一个全连接层，输出单元1024个
    W_fc1 = weight_variable([7*7*64,1024])
    b_fc1 = bias_variable([1024])

    h_pool2_flat = tf.reshape(h_pool2,[-1,7*7*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    #加入dropout
    keep_prob = tf.placeholder("float")
    h_fc1_drop = tf.nn.dropout(h_fc1,keep_prob=keep_prob)

    #softmax层
    W_fc2 = weight_variable([1024,10])
    b_fc2 = bias_variable([10])

    y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop,W_fc2)+b_fc2)

    #训练和评估
    cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv,1),tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,"float"))
    sess.run(tf.initialize_all_variables())

    for i in range(1000):
        batch = mnist.train.next_batch(50)
        if i%100 == 0:
            train_accuracy = accuracy.eval(feed_dict={x:batch[0],y_:batch[1],keep_prob:1.0})
            print("step %d, training accuracy %g",i,train_accuracy)
        train_step.run(feed_dict={x:batch[0],y_:batch[1],keep_prob:0.5})

    print("test accuracy %g",accuracy.eval(feed_dict={x:mnist.test.images,y_:mnist.test.labels,keep_prob:1.0}))






