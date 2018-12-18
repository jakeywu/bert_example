import tensorflow as tf
import modeling


class FineTuningClassifier(object):
    def __init__(self, labels, input_ids, vocab_size, num_classes, learning_rate, is_training, max_seq_length):
        self._vocab_size = vocab_size
        self._is_training = is_training
        self._input_ids = tf.cast(input_ids, tf.int32)
        self._max_seq_length = max_seq_length
        self._labels = tf.cast(labels, tf.int32)
        self._num_classes = num_classes
        self._learning_rate = learning_rate

    def bert_model(self):
        real_len = tf.reduce_sum(tf.cast(tf.not_equal(tf.to_int32(0), self._input_ids), tf.int32), axis=1)
        input_mask = tf.cast(tf.sequence_mask(real_len, self._max_seq_length), tf.int32)
        base_model = modeling.BertModel(
            config=modeling.BertConfig(vocab_size=self._vocab_size),
            is_training=self._is_training,
            input_ids=self._input_ids,
            input_mask=input_mask,
            token_type_ids=tf.zeros_like(self._input_ids, tf.int32),
            use_one_hot_embeddings=False
        )
        output_layer = base_model.get_pooled_output()
        self._inference(output_layer)
        self._build_train_op()

    def _inference(self, output_layer):
        dim = output_layer.get_shape().as_list()[1]
        with tf.variable_scope("train_op"):
            w = tf.get_variable(name="w", shape=[dim, self._num_classes], initializer=tf.truncated_normal_initializer(stddev=0.1))
            b = tf.get_variable(name="b", shape=[self._num_classes], initializer=tf.constant_initializer(0.))
            self.logits = tf.matmul(output_layer, w) + b
            self.predictions = tf.argmax(self.logits, axis=1, name="predictions")

    def _build_train_op(self):
        total_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self._labels, logits=self.logits)
        self.loss = tf.reduce_mean(total_loss)
        self.train_op = tf.train.AdamOptimizer(self._learning_rate).minimize(self.loss)
        with tf.variable_scope("accuracy"):
            correct_predictions = tf.equal(tf.cast(self.predictions, tf.int32), self._labels)
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32), name="accuracy")
