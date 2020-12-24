# !pip install easytransfer==0.1.1 --user
# 定义配置文件
config_json = {
    "worker_hosts": "localhost",
    "task_index": 1,
    "job_name": "chief",
    "num_gpus": 1,
    "num_workers": 1,
    "preprocess_config": {
        "input_schema": "input_ids:int:128,input_mask:int:128,segment_ids:int:128,label_id:int:1",
        "sequence_length": 128,
        "output_schema": "bert_predict"  # 新增
    },

    "model_config": {
        "pretrain_model_name_or_path": "brightmart-roberta-large-zh",
    },

    "train_config": {
        "train_input_fp": "train.list_tfrecord",
        "train_batch_size": 8,
        "num_epochs": 3,
        "model_dir": "model_dir",
        "optimizer_config": {
            "learning_rate": 1e-5
        },
        # 新增distribution_config
        "distribution_config": {
            "distribution_strategy": None,
        }

    },

    "predict_config": {
        "predict_checkpoint_path": None,
        "predict_input_fp": "dev.list_tfrecord",
        "predict_batch_size": 8,
        "predict_output_fp": "./output_dir"  # 新增
    }
}

import sys
#sys.path.append("/data/nas/workspace/envs/python3.6/site-packages/")
import os
import tensorflow as tf
from easytransfer import base_model, Config, FLAGS
from easytransfer import layers
from easytransfer import model_zoo
from easytransfer import preprocessors
from easytransfer.datasets import TFRecordReader
from easytransfer.losses import softmax_cross_entropy
from sklearn.metrics import classification_report
import numpy as np

class MultiTaskTFRecordReader(TFRecordReader):
    def __init__(self, input_glob, batch_size, is_training=False,
                 **kwargs):

        super(MultiTaskTFRecordReader, self).__init__(input_glob, batch_size, is_training, **kwargs)
        self.task_fps = []
        with tf.gfile.Open(input_glob, 'r') as f:
            for line in f:
                line = line.strip()
                self.task_fps.append(line)

    def get_input_fn(self):
        def input_fn():
            num_datasets = len(self.task_fps)
            datasets = []
            for input_glob in self.task_fps:
                dataset = tf.data.TFRecordDataset(input_glob)
                dataset = self._get_data_pipeline(dataset, self._decode_tfrecord)
                datasets.append(dataset)

            choice_dataset = tf.data.Dataset.range(num_datasets).repeat()
            return tf.data.experimental.choose_from_datasets(datasets, choice_dataset)

        return input_fn

class Application(base_model):
    def __init__(self, **kwargs):
        super(Application, self).__init__(**kwargs)
        self.user_defined_config = kwargs["user_defined_config"]
        # self.weight_tnews = tf.Variable([1], name="weight_tnews")
        # self.weight_ocnli = tf.Variable([1], name="weight_ocnli")
        # self.weight_ocemotion = tf.Variable([1], name="weight_ocemotion")

    def build_logits(self, features, mode=None):

        preprocessor = preprocessors.get_preprocessor(self.pretrain_model_name_or_path,
                                                      user_defined_config=self.user_defined_config)

        model = model_zoo.get_pretrained_model(self.pretrain_model_name_or_path)

        global_step = tf.train.get_or_create_global_step()

        tnews_dense = layers.Dense(15,
                     kernel_initializer=layers.get_initializer(0.02),
                     name='tnews_dense')

        ocemotion_dense = layers.Dense(7,
                             kernel_initializer=layers.get_initializer(0.02),
                             name='ocemotion_dense')

        ocnli_dense = layers.Dense(3,
                             kernel_initializer=layers.get_initializer(0.02),
                             name='ocnli_dense')
        ##add bilstm func
        fw_encoder_cell = tf.nn.rnn_cell.GRUCell(1024, name='fw_cell')
        bw_encoder_cell = tf.nn.rnn_cell.GRUCell(1024, name='bw_cell')
        discriminator_dense = tf.layers.Dense(1024, name='discriminator_dense')

        input_ids, input_mask, segment_ids, label_ids = preprocessor(features)
        ##add bilstm
        text_lens = tf.cast(tf.reduce_sum(tf.sign(input_mask), 1), tf.int32)
        outputs = model([input_ids, input_mask, segment_ids], mode=mode)
        output = outputs[0]
        _, states = tf.nn.bidirectional_dynamic_rnn(cell_fw=fw_encoder_cell,
                                                    cell_bw=bw_encoder_cell,
                                                    sequence_length=text_lens,
                                                    inputs=output,
                                                    dtype=tf.float32)
        concat_states = tf.concat(states, 1)
        concat_states = tf.nn.dropout(concat_states, keep_prob=0.7)

        dropout_output = discriminator_dense(concat_states)
        dropout_output = tf.nn.tanh(dropout_output)



        if mode == tf.estimator.ModeKeys.TRAIN:
            dropout_output = tf.nn.dropout(dropout_output, keep_prob=0.9)

        logits = tf.case([(tf.equal(tf.mod(global_step, 3), 0), lambda: tnews_dense(dropout_output)),
                          (tf.equal(tf.mod(global_step, 3), 1), lambda: ocemotion_dense(dropout_output)),
                          (tf.equal(tf.mod(global_step, 3), 2), lambda: ocnli_dense(dropout_output)),
                          ], exclusive=True)

        if mode == tf.estimator.ModeKeys.PREDICT:
            ret = {
                "tnews_logits": tnews_dense(dropout_output),
                "ocemotion_logits": ocemotion_dense(dropout_output),
                "ocnli_logits": ocnli_dense(dropout_output),
                "label_ids": label_ids
            }
            return ret

        return logits, label_ids

    def build_loss(self, logits, labels):
        global_step = tf.train.get_or_create_global_step()
        ##loss-baseline
        # return tf.case([(tf.equal(tf.mod(global_step, 3), 0), lambda : softmax_cross_entropy(labels, 15, logits)),
        #               (tf.equal(tf.mod(global_step, 3), 1), lambda : softmax_cross_entropy(labels, 7, logits)),
        #               (tf.equal(tf.mod(global_step, 3), 2), lambda : softmax_cross_entropy(labels, 3, logits))
        #               ], exclusive=True)
        ##add learning loss param
        self.weight_tnews = tf.Variable([1.0], name="weight_tnews")
        self.weight_ocnli = tf.Variable([1.0], name="weight_ocnli")
        self.weight_ocemotion = tf.Variable([1.0], name="weight_ocemotion")
        return tf.case([(tf.equal(tf.mod(global_step, 3), 0), lambda: self.weight_tnews*softmax_cross_entropy(labels, 15, logits)),
                        (tf.equal(tf.mod(global_step, 3), 1), lambda: self.weight_ocemotion*softmax_cross_entropy(labels, 7, logits)),
                        (tf.equal(tf.mod(global_step, 3), 2), lambda: self.weight_ocnli*softmax_cross_entropy(labels, 3, logits))
                        ], exclusive=True)

    def build_predictions(self, output):
        tnews_logits = output['tnews_logits']
        ocemotion_logits = output['ocemotion_logits']
        ocnli_logits = output['ocnli_logits']

        tnews_predictions = tf.argmax(tnews_logits, axis=-1, output_type=tf.int32)
        ocemotion_predictions = tf.argmax(ocemotion_logits, axis=-1, output_type=tf.int32)
        ocnli_predictions = tf.argmax(ocnli_logits, axis=-1, output_type=tf.int32)

        ret_dict = {
            "tnews_predictions": tnews_predictions,
            "ocemotion_predictions": ocemotion_predictions,
            "ocnli_predictions": ocnli_predictions,
            "label_ids": output['label_ids']
        }
        return ret_dict


config = Config(mode="predict", config_json=config_json)
app = Application(user_defined_config=config)

predict_reader = MultiTaskTFRecordReader(input_glob=app.predict_input_fp,
                                         is_training=False,
                                         input_schema=app.input_schema,
                                         batch_size=app.predict_batch_size)

ckpts = set()
with tf.gfile.GFile(os.path.join(config_json["train_config"]["model_dir"], "checkpoint"), mode='r') as reader:
    for line in reader:
        line = line.strip()
        line = line.replace("oss://", "")
        ckpts.add(int(line.split(":")[1].strip().replace("\"", "").split("/")[-1].replace("model.ckpt-", "")))

best_macro_f1 = 0
best_ckpt = None
for ckpt in sorted(ckpts):
    checkpoint_path = os.path.join(config_json["train_config"]["model_dir"], "model.ckpt-" + str(ckpt))
    tf.logging.info("checkpoint_path is {}".format(checkpoint_path))
    all_tnews_preds = []
    all_tnews_gts = []
    all_ocemotion_preds = []
    all_ocemotion_gts = []
    all_ocnli_preds = []
    all_ocnli_gts = []
    for i, output in enumerate(app.run_predict(reader=predict_reader, checkpoint_path=checkpoint_path)):
        label_ids = np.squeeze(output['label_ids'])
        if i % 3 == 0:
            tnews_predictions = output['tnews_predictions']
            all_tnews_preds.extend(tnews_predictions.tolist())
            all_tnews_gts.extend(label_ids.tolist())
        elif i % 3 == 1:
            ocemotion_predictions = output['ocemotion_predictions']
            all_ocemotion_preds.extend(ocemotion_predictions.tolist())
            all_ocemotion_gts.extend(label_ids.tolist())
        elif i % 3 == 2:
            ocnli_predictions = output['ocnli_predictions']
            all_ocnli_preds.extend(ocnli_predictions.tolist())
            all_ocnli_gts.extend(label_ids.tolist())

        # if i == 20:
        #     break
    tenws_idx2tag={'0':'115','1':'114','2':'108','3':'109','4':'116','5':'110','6':'113','7':'112','8':'102','9':'103','10':'100','11':'101','12':'106','13':'107','14':'104'}
    ocnli_idx2tag = {'0':'0','1':'1','2':'2'}
    ocemotion_idx2tag = {'0':'like','1':'happiness','2':'sadness','3':'anger','4':'disgust','5':'fear','6':'surprise'}

    import json
    i=0
    with open("./output_dir/tnews_predict.json", "w") as f:
        for i in range(1500):
            line = {'id': str(i),'label': tenws_idx2tag[str(all_tnews_preds[i])]}
            f.write(json.dumps(line))
            if i!=1499:
                f.write("\n")
    with open("./output_dir/ocnli_predict.json", "w") as f:
        for i in range(1500):
            line = {'id': str(i),'label': ocnli_idx2tag[str(all_ocnli_preds[i])]}
            f.write(json.dumps(line))
            if i != 1499:
                f.write("\n")
    with open("./output_dir/ocemotion_predict.json", "w") as f:
        for i in range(1500):
            line = {'id': str(i),'label': ocemotion_idx2tag[str(all_ocemotion_preds[i])]}
            f.write(json.dumps(line))
            if i != 1499:
                f.write("\n")

    tnews_report = classification_report(all_tnews_gts, all_tnews_preds, digits=4)
    tnews_macro_avg_f1 = float(tnews_report.split()[-8])

    ocemotion_report = classification_report(all_ocemotion_gts, all_ocemotion_preds, digits=4)
    ocemotion_macro_avg_f1 = float(ocemotion_report.split()[-8])

    ocnli_report = classification_report(all_ocnli_gts, all_ocnli_preds, digits=4)
    ocnli_macro_avg_f1 = float(ocnli_report.split()[-8])

    macro_f1 = (tnews_macro_avg_f1 + ocemotion_macro_avg_f1 + ocnli_macro_avg_f1) / 3.0
    if macro_f1 >= best_macro_f1:
        best_macro_f1 = macro_f1
        best_ckpt = ckpt

tf.logging.info("best ckpt {}, best best_macro_f1 {}".format(best_ckpt, best_macro_f1))
