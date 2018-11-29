# coding = utf8
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# mnist = input_data.read_data_sets("/path/to/MNIST_data")
# print("Training data size: ", mnist.train.num_examples)
# print("Validating data size: ", mnist.validation.num_examples)
# print("Testing data size: ", mnist.test.num_examples)

# mnist数据集相关的常数
INPUT_NODE = 784    # 输入层的节点数（MNIST数据集中图片的像素）
OUTPUT_NODE = 10    # 输出层的节点数，等于类别的数目

# 配置神经网络的参数9
LAYER1_NODE = 500   # 隐藏层节点数

BATCH_SIZE = 100    # 一个训练batch中训练数据的个数
LEARN_RATE_BASE = 0.8   # 基础学习率
LEARNN_RATE_DECAY = 0.99     # 学习率的衰减率
REGULATION_RATE = 0.0001    # 描述模型复杂度的正则化项在损失函数中的系数
TRAIN_STEPS = 20000     # 训练轮数
MOVING_AVERAGE_DECAY = 0.99     # 滑动平均衰减率


# 辅助函数
# 给定神经网络的输入和所有参数，计算神经网络的前向传播结果
# 定义了一个使用ReLU激活函数的三层全连接神经网络
# 通过加入隐藏层实现了多层网络结构，通过激活函数实现了去线性化
# 支持传入用于计算参数平均值的类，方便在测试时使用滑动平均模型
def inference(input_tensor, avg_class,
              weights1, biases1, weights2, biases2):
    # 当没有提供滑动平均类时，直接使用参数当前取值
    if avg_class == None:
        # 使用ReLU函数计算隐藏层的前向传播结果
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights1) + biases1)

        return tf.matmul(layer1, weights2) + biases2
    else:
        # 首先使用avg_class.average函数来计算得出变量的滑动平均值
        # 然后再计算相应的神经网络前向传播结果
        layer1 = tf.nn.relu(tf.matmul(input_tensor, avg_class.average(weights1)) +
                            avg_class.average(biases1))
        return tf.matmul(layer1, avg_class.average(weights2)) + avg_class.average(biases2)


# 训练模型的过程
def train(mnist):
    x = tf.placeholder(tf.float32, [None, INPUT_NODE], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, OUTPUT_NODE], name='y-input')

    # 生成隐藏层的参数
    weights1 = tf.Variable(tf.truncated_normal([INPUT_NODE, LAYER1_NODE], stddev=0.1))
    biases1 = tf.Variable(tf.constant(0.1, shape=[LAYER1_NODE]))
    # 生成输出层的参数
    weights2 = tf.Variable(tf.truncated_normal([LAYER1_NODE, OUTPUT_NODE], stddev=0.1))
    biases2 = tf.Variable(tf.constant(0.1, shape=[OUTPUT_NODE]))

    # 计算在当前参数下神经网络前向传播的结果
    # 这里给出的用于计算滑动平均的类为None，因而函数不会使用参数的滑动平均值
    y = inference(x, None, weights1, biases1, weights2, biases2)

    # 定义存储训练轮数的变量
    # 该变量不需要计算滑动平均值，因而这里指定该变量为不可训练的变量
    # 在训练神经网络时，一般将代表训练轮数的变量指定为n不可训练的参数
    global_step = tf.Variable(0, trainable=False)

    # 给定滑动平均衰减率和训练轮数的变量，初始化滑动平均类
    variable_averages = tf.train.ExponentialMovingAverage(
        MOVING_AVERAGE_DECAY, global_step)

    # 在所有代表神经网络参数的变量上使用滑动平均
    variable_averages_op = variable_averages.apply(tf.trainable_variables())

    # 计算使用了滑动平均之后的前向传播结果
    # 滑动平均不会改变变量本身的取值，而是会维护一个影子变量来记录滑动平均值
    average_y = inference(x, variable_averages, weights1, biases1, weights2, biases2)

    # 计算交叉熵作为刻画预测值和真实值之间差距的损失函数
    # 使用tf.argmax函数来得到正确答案对应的编号
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.arg_max(y_, 1))
    # 计算当前batch中所有样例的交叉熵平均值
    cross_entropy_mean = tf.reduce_mean(cross_entropy)

    # 计算L2正则化损失函数
    regularizer = tf.contrib.layers.l2_regularizer(REGULATION_RATE)
    # 计算模型的正则化损失
    regularization = regularizer(weights1) + regularizer(weights2)
    # 总损失等于交叉熵损失和正则化损失的和
    loss = cross_entropy_mean + regularization
    # 设置指数衰减的学习率
    learning_rate = tf.train.exponential_decay(LEARN_RATE_BASE, global_step,
                                               mnist.train.num_examples / BATCH_SIZE, LEARNN_RATE_DECAY)
    # 使用tf.train.GradientDescentOptimizer优化算法来优化损失函数
    # 这里的损失函数包含了交叉熵损失和L2正则化损失
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

    # 通过反向传播来更新神经网络中的参数
    # 更新每一个参数的滑动平均值
    # 该功能也可以通过以下语句实现：train_op = tf.group(train_step, variables_averages_op)
    with tf.control_dependencies([train_step, variable_averages_op]):
        train_op = tf.no_op(name='train')

    # 检验使用了滑动平均模型的神经网络前向传播结果是否正确
    # tf.argmax(average_y, 1)计算每一个样例的预测答案
    # tf.equal函数判断两个张量的每一维是否相等，返回值为bool类型
    correct_prediction = tf.equal(tf.arg_max(average_y, 1), tf.argmax(y_, 1))

    # 将布尔型数值转换为实数型，然后计算平均值
    # 该平均值就是模型在这一组数据上的正确率
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # 初始化会话并开始训练过程
    with tf.Session() as sess:
        tf.initialize_all_variables().run()
        # 准备验证数据
        validate_feed = {x: mnist.validation.images,
                         y_:mnist.validation.labels}
        # 准备测试数据
        # 在真是应用中，这部分数据是不可见的，这个数据只作为模型优劣的最后评价标准
        test_feed = {x: mnist.test.images,
                     y_: mnist.test.labels}
        # 迭代地训练神经网络
        for i in range(TRAIN_STEPS):
            # 每1000轮输出一次在验证数据集上的测试结果
            if i % 1000 == 0:
                # 计算滑动平均模型在验证数据集上的结果
                # 因MNIST数据集比较小，一次可处理所有验证数据
                # 当神经网络模型比较复杂或者验证数据集比较大时，太大的batch会导致计算时间过长甚至发生内存溢出错误
                validate_acc = sess.run(accuracy, feed_dict=validate_feed)
                print("After %d training step(s), validation accuracy "
                      "using average model is %g" % (i, validate_acc))

                # 产生这一轮使用的一个batch的训练数据，并运行训练过程
                xs, ys = mnist.train.next_batch(BATCH_SIZE)
                sess.run(train_op, feed_dict={x: xs, y_: ys})

                # 训练结束后，在测试数据上检测神经网络模型的最终正确率
                test_acc = sess.run(accuracy, feed_dict=test_feed)
                print("After %d training step(s), test accuracy using "
                      "average model is %g" % (TRAIN_STEPS, test_acc))


# 主程序入口
def main(argv=None):
    # 声明处理MNIST数据集的类，这个类在初始化时会自动下载数据
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    train(mnist)


# TensorFlow提供的一个主程序入口，tf.app.run会调用上面定义的函数
if __name__ == '__main__':
    tf.app.run()
