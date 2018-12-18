import os
import tensorflow as tf
from classifier import FineTuningClassifier
from pre_process import ReadDataSet

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 使用第一块GPU
tf.flags.DEFINE_string("output_dir", "./data/model", "The output directory where the model checkpoints will be written.")
tf.flags.DEFINE_string("init_checkpoint", None, "Initial checkpoint (usually from a pre-trained BERT model).")
tf.flags.DEFINE_string("bert_config_file", None, "The config json file corresponding to the pre-trained BERT model. ")
tf.flags.DEFINE_integer("max_seq_length", 128, "The maximum total input sequence length after WordPiece tokenization. ")
tf.flags.DEFINE_integer("batch_size", 32, "Total batch size for training/eval/predict")
tf.flags.DEFINE_integer("epoch_size", 3, "Total epoch size for training")
tf.flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for Adam.")

FLAG = tf.flags.FLAGS


def main(_):
    rds = ReadDataSet()
    iterator = rds.make_data_iterator()
    label_ids, input_ids = iterator.get_next()
    ftc = FineTuningClassifier(
        labels=label_ids,
        input_ids=input_ids,
        vocab_size=rds.vocabs_size,
        num_classes=rds.num_classes,
        learning_rate=FLAG.learning_rate,
        is_training=True,
        max_seq_length=FLAG.max_seq_length)
    ftc.bert_model()
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        for _ in range(FLAG.epoch_size):
            sess.run(iterator.initializer, feed_dict={
                rds.filenames: ["./data/tfrecords/train_1.tfrecords"],
                rds.batch_size: FLAG.batch_size,
            })
            while True:
                try:
                    _, loss, acc = sess.run([ftc.train_op, ftc.loss, ftc.accuracy])
                    print("损失为{} \t 准确率为{}".format(loss, acc))
                except tf.errors.OutOfRangeError:
                    print("Done traini")
                    break


if __name__ == "__main__":
    tf.flags.mark_flags_as_required(["init_checkpoint", "bert_config_file"])
    tf.app.run(main)
