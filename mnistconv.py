# -*- coding: utf-8 -*-

import tensorflow as tf #导入tensorflow库
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
import pylab 
import cv2 as cv
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
'''
tf.truncated_normal(shape, mean, stddev) :shape表示生成张量的维度，mean是均值，stddev是标准差。
这个函数产生正太分布，均值和标准差自己设定。这是一个截断的产生正太分布的函数，就是说产生正太分布的值如果与均值的差值大于两倍的标准差，那就重新生成.
'''

summary_dir="logs"#设置路径

def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)

def bias_variable(shape):
	initial = tf.constant(0.1, shape=shape)#创建一个常数张量
	return tf.Variable(initial)
  
'''
第一个参数input：卷积输入
第二个参数filter：卷积核
第三个参数strides：卷积时在图像每一维的步长，这是一个一维的向量，长度4
第四个参数padding：string类型的量，只能是"SAME","VALID"其中之一，这个值决定了不同的卷积方式
第五个参数：use_cudnn_on_gpu:bool类型，是否使用cudnn加速，默认为true
结果返回一个Tensor，这个输出，就是我们常说的feature map
'''
def conv2d(x, W):
	return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')#实现卷积函数

'''
第一个参数value：需要池化的输入，一般池化层接在卷积层后面，所以输入通常是feature map
第二个参数ksize：池化窗口的大小，取一个四维向量,一般是[1, height, width, 1]，因为我们不想在batqch和channels上做池化，所以这两个维度设为了1
第三个参数strides：和卷积类似，窗口在每一个维度上滑动的步长，一般也是[1, stride,stride, 1]
第四个参数padding：和卷积类似，可以取'VALID' 或者'SAME' 	
'''
def max_pool_2x2(x):
	return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')  

'''
value: 一个4维tensor，形状为[batch, height, width, channels] 元素类型可以是 float32, float64, qint8, quint8, or qint32. 
ksize: 整形列表,长度 >= 4. 表示窗口在输入的每个维度上面的尺寸.一般在二维的图像的情况下,都是[1,高,宽,1]
strides: 整形列表,长度 >= 4. 表示窗口滑动在输入tensor上面每个维度滑动的的步长.和卷积操作是一样的.
padding: 两种模式 ‘VALID’ 或者 ‘SAME’.
data_format: 两种模式 ‘NHWC’ 和 ‘NCHW’
name: 可选，操作名	
'''
def avg_pool_7x7(x):
	return tf.nn.avg_pool(x, ksize=[1, 7, 7, 1],strides=[1, 7, 7, 1], padding='SAME')

# tf Graph Input
with tf.name_scope('Input'):
	x = tf.placeholder(tf.float32, [None, 784]) # mnist data维度 28*28=784
	y = tf.placeholder(tf.float32, [None, 10]) # 0-9 数字=> 10 classes

with tf.name_scope('Inference'):
	W_conv1 = weight_variable([5, 5, 1, 32])
	b_conv1 = bias_variable([32])

	x_image = tf.reshape(x, [-1,28,28,1])
	tf.summary.image('images', x_image)

	h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
	h_pool1 = max_pool_2x2(h_conv1)

	W_conv2 = weight_variable([5, 5, 32, 64])
	b_conv2 = bias_variable([64])

	h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
	h_pool2 = max_pool_2x2(h_conv2)

	W_conv3 = weight_variable([5, 5, 64, 10])
	b_conv3 = bias_variable([10])

	h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
	nt_hpool3=avg_pool_7x7(h_conv3)#64

	nt_hpool3_flat = tf.reshape(nt_hpool3, [-1, 10])
	y_conv=tf.nn.softmax(nt_hpool3_flat)

with tf.name_scope('Optimization'):
	cross_entropy = -tf.reduce_sum(y*tf.log(y_conv))
	train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

#计算准确度
with tf.name_scope('Accuracy'):
	correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
	tf.summary.scalar('Accuracy', accuracy)
	tf.summary.histogram('Accuracy', accuracy)

merge = tf.summary.merge_all()

saver = tf.train.Saver()
model_path = "log/CNNmodel.ckpt"
train_flag=0 #1训练神经网络0使用神经网络模型

if train_flag==1:
# 启动session
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		summary_writer = tf.summary.FileWriter(logdir=summary_dir, graph=sess.graph)
		for i in range(20000):#20000
			batch = mnist.train.next_batch(50)#50
			if i%20 == 0:
				train_accuracy = accuracy.eval(feed_dict={x:batch[0], y: batch[1]})
				print( "step %d, training accuracy %g"%(i, train_accuracy))
			_,summary=sess.run([train_step,merge],feed_dict={x: batch[0], y: batch[1]})
			summary_writer.add_summary(summary, global_step=i)
		
		print ("test accuracy %g"%accuracy.eval(feed_dict={x: mnist.test.images, y: mnist.test.labels}))
		save_path = saver.save(sess, model_path)
		print("Model saved in file : %s" % save_path)
		summary_writer.close()
	
else :
	#读取模型
	testnum=1
	print("Starting test...")

	with tf.Session() as sess:
	
		# Initialize variables
		sess.run(tf.global_variables_initializer())
		# Restore model weights from previously saved model
		saver.restore(sess, model_path)
		
		#1.载入图片
		img=cv.imread('2.png',cv.IMREAD_COLOR)
		#2.将图片转为灰度图
		img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
		kernel = np.ones((2,2),np.uint8)  
		img_gray = cv.erode(img_gray,kernel,iterations = 1)
		img_gray[np.where(img_gray>170)] = 255
		img_gray[np.where(img_gray<200)] = 0

		#cv.imshow('2018',img_gray)  #显示
		#cv.waitKey(0) 

		img_temp=img_gray
		#3.取一个42x42的框 1(161x36) 5(457x77) 7(137x311)
		xx=0
		yy=0
		#t=0
		#9.滑动窗口
		while(1):			#整个循环过程是个死循环
			if(xx+56+12>640):  #右边边界
				xx=0
				yy=yy+12		#步进为6
			xx=xx+12
			if yy + 56 > 480:
				break
			#t=t+1
			#print t
			crop_img= img_temp[yy:yy+56, xx:xx+56]
			#4.resize成（28x28）
			size = crop_img.shape
			#print size
			img_resize =np.array(cv.resize(crop_img,(int(size[1]/2),int(size[0]/2))))
			#print img_resize.shape
			#5.映射
			img_ys = cv.divide(img_resize.astype(float), 255)
			#6.反转
			img_res = 1-img_ys
			#print img_res.shape
			#7.过滤
			img_res[np.where(img_res<0.5)] = 0
			#8.测试
			#print img_res.shape
			#print "xx= %d,yy=%d"%(xx,yy)
			img_res = img_res.reshape(1,784)
			n = sess.run(y_conv, feed_dict={x: img_res}) #前向传播
			#print n.max	
			if n.max() > 0.96:
				cv.rectangle(img_gray,(xx,yy),(xx+56,yy+56),(0,255,0))
				print n.max()

		cv.imshow('2018',img_gray)  #显示
		cv.waitKey(0)  
