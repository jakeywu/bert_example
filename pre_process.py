import codecs
import unicodedata
import tensorflow as tf


class DataBase:
    def __init__(self):
        self.vocabs_ids = self._chinese_vocab()
        self.ids_vocabs = {v: k for k, v in self.vocabs_ids.items()}
        self.classes_ids = self._classify_name_ids()
        self.ids_classes = {v: k for k, v in self.classes_ids.items()}
        self.vocabs_size = len(self.vocabs_ids.keys())
        self.num_classes = len(self.classes_ids.keys())

    @staticmethod
    def _chinese_vocab():
        vocab_ids = dict()
        with codecs.open("data/vocab.txt", "r", "utf8") as f:
            for i, line in enumerate(f.readlines()):
                vocab_ids[line.strip()] = i
        return vocab_ids

    @staticmethod
    def _classify_name_ids():
        return {
            "政策监管": 0,
            "合作竞争": 1,
            "股价行情": 2,
            "欺诈骗局": 3,
            "裁员降薪": 4,
            "企业盈亏": 5,
            "股权变动": 6,
            "公告公示": 7,
            "成果奖项": 8,
            "高管动态": 9,
            "上市资讯": 10,
            "债务信息": 11,
            "司法涉诉": 12,
            "运营状况": 13,
            "事故信息": 14,
            "投资融资": 15,
            "产品相关": 16,
            "战略发展": 17
        }

    @staticmethod
    def _read_dataset(file_path):
        """读取原始数据"""
        with codecs.open(file_path, "r", "utf8") as f:
            return f.readlines()


class PrepareDataSet(DataBase):
    def __init__(self, max_seq_length=128):
        super().__init__()
        self._max_seq_length = max_seq_length
        self.file_split_num = 10000

    def _pad_or_truncated(self, sequence_lst):
        """补充对齐"""
        if len(sequence_lst) >= self._max_seq_length:
            return sequence_lst[0:self._max_seq_length]
        sequence_lst += [self.vocabs_ids["[PAD]"]] * (self._max_seq_length - len(sequence_lst))
        return sequence_lst

    @staticmethod
    def __is_whitespace(char):
        """判断是否是空白符"""
        if char == " " or char == "\t" or char == "\n" or char == "\r":
            return True
        cat = unicodedata.category(char)
        if cat == "Zs":
            return True
        return False

    def prepare_train_data(self):
        train_lst = []
        count = 0
        train_tfrecords_name = "./data/tfrecords/train_{}.tfrecords"
        for line in self._read_dataset("data/origindata/trainset.txt"):
            classify_name = line.split(" ")[0]
            class_id = self.classes_ids[classify_name]
            char_ids = [self.vocabs_ids["[CLS]"]]
            sentence = unicodedata.normalize("NFC", line.lstrip(classify_name).strip())
            for char in sentence:
                if self.__is_whitespace(char):
                    continue
                try:
                    char_id = self.vocabs_ids[char]
                except KeyError:
                    # todo 需要添加自定义词典, 中文词典vocab没有a/b/A/B等很多新词
                    char_id = self.vocabs_ids["[UNK]"]
                char_ids.append(char_id)
            char_ids.append(self.vocabs_ids["[SEP]"])
            char_ids = self._pad_or_truncated(char_ids)
            train_lst.append([class_id, char_ids])
            if len(train_lst) % self.file_split_num == 0:
                count += 1
                self._convert_tf_records(train_lst, train_tfrecords_name.format(str(count)))
                train_lst = []
        if len(train_lst) > 0:
            self._convert_tf_records(train_lst, train_tfrecords_name.format(str(count+1)))
        del train_lst

    def prepare_test_data(self):
        test_lst = []
        count = 0
        test_tfrecords_name = "./data/tfrecords/test_{}.tfrecords"
        for line in self._read_dataset("data/origindata/testset.txt"):
            classify_name = line.split(" ")[0]
            class_id = self.classes_ids[classify_name]
            char_ids = [self.vocabs_ids["[CLS]"]]
            sentence = unicodedata.normalize("NFC", line.lstrip(classify_name).strip())
            for char in sentence:
                if self.__is_whitespace(char):
                    continue
                try:
                    char_id = self.vocabs_ids[char]
                except KeyError:
                    # todo 需要添加自定义词典, 中文词典vocab没有a/b/A/B等很多新词
                    char_id = self.vocabs_ids["[UNK]"]
                char_ids.append(char_id)
            char_ids.append(self.vocabs_ids["[SEP]"])
            char_ids = self._pad_or_truncated(char_ids)
            test_lst.append([class_id, char_ids])
            if len(test_lst) % self.file_split_num == 0:
                count += 1
                self._convert_tf_records(test_lst, test_tfrecords_name.format(str(count)))
                train_lst = []
        if len(test_lst) > 0:
            self._convert_tf_records(test_lst, test_tfrecords_name.format(str(count+1)))
        del test_lst

    @staticmethod
    def _convert_tf_records(data_lst, tfrecords_filename="./data/tfrecords/train.tfrecords"):
        writer = tf.python_io.TFRecordWriter(tfrecords_filename)
        for l_ids, c_ids in data_lst:
            example = tf.train.Example(
                features=tf.train.Features(
                    feature={
                        "label_id": tf.train.Feature(int64_list=tf.train.Int64List(value=[l_ids])),
                        "input_ids": tf.train.Feature(int64_list=tf.train.Int64List(value=c_ids))
                    }
                )
            )
            writer.write(example.SerializeToString())
        writer.close()


class ReadDataSet(DataBase):
    def __init__(self, max_seq_length=128):
        DataBase.__init__(self)
        self._max_seq_length = max_seq_length
        self._buffer_size = 1000
        self.filenames = tf.placeholder(dtype=tf.string, shape=[None], name="f_tfrecord")
        self.batch_size = tf.placeholder(dtype=tf.int64, shape=None, name="batch_size")

    def _parse_functions(self, example_proto):
        features = {
            "label_id": tf.FixedLenFeature([], tf.int64),
            "input_ids": tf.FixedLenFeature([self._max_seq_length], tf.int64)
        }
        parsed_features = tf.parse_single_example(example_proto, features)
        return parsed_features["label_id"], parsed_features["input_ids"]

    def make_data_iterator(self):
        dataset = tf.data.TFRecordDataset(self.filenames)
        dataset = dataset.map(self._parse_functions)
        dataset = dataset.shuffle(buffer_size=self._buffer_size)
        dataset = dataset.batch(batch_size=self.batch_size)
        iterator = dataset.make_initializable_iterator()
        return iterator


class DataSetUtil(object):
    def __init__(self):
        self.vocab_ids = self._chinese_vocab()
        self.id_vocab = {v: k for k, v in self.vocab_ids.items()}

    @staticmethod
    def read_dataset(file_path):
        """读取原始数据"""
        with codecs.open(file_path, "r", "utf8") as f:
            while True:
                line = f.readline()
                if line:
                    yield line.strip()
                break

    @staticmethod
    def _chinese_vocab():
        vocab_ids = dict()
        with codecs.open("/usr/projects/github/bert_classifier/data/vocab.txt", "r", "utf8") as f:
            for i, line in enumerate(f.readlines()):
                vocab_ids[line.strip()] = i
        return vocab_ids


if __name__ == "__main__":
    # dsu = PrepareDataSet()
    # dsu.prepare_train_data()
    # dsu.prepare_test_data()
    db = DataBase()
    rds = ReadDataSet()
    iter1 = rds.make_data_iterator()
    a, b = iter1.get_next()
    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        for _ in range(3):
            sess.run(iter1.initializer, feed_dict={
                rds.filenames: ["./data/tfrecords/train_1.tfrecords"],
                rds.batch_size: 64
            })
            while True:
                try:
                    print(sess.run(a))
                    print(sess.run(b))
                except tf.errors.OutOfRangeError:
                    break
        # aa = tf.reduce_sum(tf.cast(tf.not_equal(tf.to_int32(0), inputs_id), tf.int32), axis=1)
        # labels, inputs, aa = sess.run([label_id, inputs_id, aa])
        # import pdb
        # pdb.set_trace()
        # print(db.ids_classes[labels[0]] + "\t" + "".join(db.ids_vocabs[k] for k in inputs[0]))
