import tensorflow as tf
import cnn_lstm.utils as util
import numpy as np
import os


def cnn_encoder_layer(data, filter_layer, strides):
    """
    :param data: the input data, when it is the first layer is 5 * 30 * 30 * 3, the second layer is 30 * 30 * 32,
                 the third layer is 15 * 15 * 64, the fourth layer is 8 * 8 * 128
    :param filter_layer:
    :param strides:
    :return: the result after conv, the first layer is 30 * 30 * 32, the second layer is 15 * 15 * 64, the third layer
             is 8 * 8 * 128, the final layer is 4 * 4 * 256
    """

    result = tf.nn.conv2d(
        input=data,
        filter=filter_layer,
        strides=strides,
        padding="SAME")
    return tf.nn.selu(result)


def tensor_variable(shape, name):
    """
    Tensor variable declaration initialization
    :param shape:
    :param name:
    :return:
    """
    variable = tf.Variable(tf.zeros(shape), name=name)
    variable = tf.compat.v1.get_variable(name, shape=shape, initializer=tf.contrib.layers.xavier_initializer())
    return variable


def cnn_encoder(data):
    """

    :param data: the input data size is 5 * 30 * 30 * 3
    :return:
    """
    # the first layer,the output size is 30 * 30 * 32
    filter1 = tensor_variable([3, 3, 3, 32], "filter1")
    strides1 = (1, 1, 1, 1)
    cnn1_out = cnn_encoder_layer(data, filter1, strides1)

    # the second layer, the output size is 15 * 15 * 64
    filter2 = tensor_variable([3, 3, 32, 64], "filter2")
    strides2 = (1, 2, 2, 1)
    cnn2_out = cnn_encoder_layer(cnn1_out, filter2, strides2)

    # the third layer, the output size is 8 * 8 * 128
    filter3 = tensor_variable([2, 2, 64, 128], "filter3")
    strides3 = (1, 2, 2, 1)
    cnn3_out = cnn_encoder_layer(cnn2_out, filter3, strides3)

    # the fourth layer, the output size is 4 * 4 * 256
    filter4 = tensor_variable([2, 2, 128, 256], "filter4")
    strides4 = (1, 2, 2, 1)
    cnn4_out = cnn_encoder_layer(cnn3_out, filter4, strides4)

    return cnn1_out, cnn2_out, cnn3_out, cnn4_out


def cnn_lstm_attention_layer(input_data, layer_number):
    """

    :param input_data:
    :param layer_number:
    :return:
    """
    convlstm_layer = tf.contrib.rnn.ConvLSTMCell(
        conv_ndims=2,
        input_shape=[input_data.shape[2], input_data.shape[3], input_data.shape[4]],
        output_channels=input_data.shape[-1],
        kernel_shape=[2, 2],
        use_bias=True,
        skip_connection=False,
        forget_bias=1.0,
        initializers=None,
        name="conv_lstm_cell" + str(layer_number))

    outputs, state = tf.nn.dynamic_rnn(convlstm_layer, input_data, dtype=input_data.dtype)

    # attention based on inner-product between feature representation of last step and other steps
    attention_w = []
    for k in range(util.step_max):
        attention_w.append(tf.reduce_sum(tf.multiply(outputs[0][k], outputs[0][-1])) / util.step_max)
    attention_w = tf.reshape(tf.nn.softmax(tf.stack(attention_w)), [1, util.step_max])

    outputs = tf.reshape(outputs[0], [util.step_max, -1])
    outputs = tf.matmul(attention_w, outputs)
    outputs = tf.reshape(outputs, [1, input_data.shape[2], input_data.shape[3], input_data.shape[4]])

    return outputs, attention_w


def cnn_decoder_layer(conv_lstm_out_c, filter, output_shape, strides):
    """

    :param conv_lstm_out_c:
    :param filter:
    :param output_shape:
    :param strides:
    :return:
    """

    deconv = tf.nn.conv2d_transpose(
        value=conv_lstm_out_c,
        filter=filter,
        output_shape=output_shape,
        strides=strides,
        padding="SAME")
    deconv = tf.nn.selu(deconv)
    return deconv


def cnn_decoder(lstm1_out, lstm2_out, lstm3_out, lstm4_out):
    d_filter4 = tensor_variable([2, 2, 128, 256], "d_filter4")
    dec4 = cnn_decoder_layer(lstm4_out, d_filter4, [1, 8, 8, 128], (1, 2, 2, 1))
    dec4_concat = tf.concat([dec4, lstm3_out], axis=3)

    d_filter3 = tensor_variable([2, 2, 64, 256], "d_filter3")
    dec3 = cnn_decoder_layer(dec4_concat, d_filter3, [1, 15, 15, 64], (1, 2, 2, 1))
    dec3_concat = tf.concat([dec3, lstm2_out], axis=3)

    d_filter2 = tensor_variable([3, 3, 32, 128], "d_filter2")
    dec2 = cnn_decoder_layer(dec3_concat, d_filter2, [1, 30, 30, 32], (1, 2, 2, 1))
    dec2_concat = tf.concat([dec2, lstm1_out], axis=3)

    d_filter1 = tensor_variable([3, 3, 3, 64], "d_filter1")
    dec1 = cnn_decoder_layer(dec2_concat, d_filter1, [1, 30, 30, 3], (1, 1, 1, 1))

    return dec1


def main():
    # Read dataset from file
    matrix_data_path = util.train_data_path + "train.npy"
    matrix_gt_1 = np.load(matrix_data_path)

    sess = tf.Session()
    data_input = tf.compat.v1.placeholder(tf.float32, [util.step_max, 30, 30, 3])

    # cnn encoder
    conv1_out, conv2_out, conv3_out, conv4_out = cnn_encoder(data_input)

    conv1_out = tf.reshape(conv1_out, [-1, 5, 30, 30, 32])
    conv2_out = tf.reshape(conv2_out, [-1, 5, 15, 15, 64])
    conv3_out = tf.reshape(conv3_out, [-1, 5, 8, 8, 128])
    conv4_out = tf.reshape(conv4_out, [-1, 5, 4, 4, 256])

    # lstm with attention
    conv1_lstm_attention_out, atten_weight_1 = cnn_lstm_attention_layer(conv1_out, 1)
    conv2_lstm_attention_out, atten_weight_2 = cnn_lstm_attention_layer(conv2_out, 2)
    conv3_lstm_attention_out, atten_weight_3 = cnn_lstm_attention_layer(conv3_out, 3)
    conv4_lstm_attention_out, atten_weight_4 = cnn_lstm_attention_layer(conv4_out, 4)

    # cnn decoder
    deconv_out = cnn_decoder(conv1_lstm_attention_out, conv2_lstm_attention_out, conv3_lstm_attention_out,
                             conv4_lstm_attention_out)
    # loss function: reconstruction error of last step matrix
    loss = tf.reduce_mean(tf.square(data_input[-1] - deconv_out))
    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=util.learning_rate).minimize(loss)

    # variable initialization
    init = tf.global_variables_initializer()
    sess.run(init)

    # training
    for idx in range(util.train_start_id, util.train_end_id):
        matrix_gt = matrix_gt_1[idx - util.train_start_id]
        feed_dict = {data_input: np.asarray(matrix_gt)}
        a, loss_value = sess.run([optimizer, loss], feed_dict)
        print("mse of last train data: " + str(loss_value))

    # test
    # Read the data from test file.
    matrix_data_path = util.test_data_path + "test.npy"
    matrix_gt_1 = np.load(matrix_data_path)
    result_all = []
    for idx in range(util.test_start_id, util.test_end_id):
        matrix_gt = matrix_gt_1[idx - util.test_start_id]
        feed_dict = {data_input: np.asarray(matrix_gt)}
        result, loss_value = sess.run([deconv_out, loss], feed_dict)
        result_all.append(result)
        print("mse of last test data: " + str(loss_value))

    # Write the reconstructed matrix to the file
    reconstructed_path = util.reconstructed_data_path
    if not os.path.exists(reconstructed_path):
        os.makedirs(reconstructed_path)
    reconstructed_path = reconstructed_path + "test_reconstructed.npy"

    result_all = np.asarray(result_all).reshape((-1, 30, 30, 3))
    print(result_all.shape)
    np.save(reconstructed_path, result_all)


if __name__ == '__main__':
    main()
