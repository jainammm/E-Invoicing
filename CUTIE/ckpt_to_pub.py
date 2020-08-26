import os
import tensorflow as tf
import json


from data_loader_json import DataLoader
# from configs.config import model_params
from model_cutie_aspp import CUTIERes as CUTIEv1

trained_checkpoint_prefix = '/home/jainammm/flipkart-grid2.0/code/CUTIEPI/pretrained_model/CUTIE.ckpt'
export_dir = os.path.join('model_for_serving', 'CUTIE', '1')

num_words = 20000

num_classes = 26

# model
network = CUTIEv1(num_words, num_classes, model_params)

model_input = network.data_grid
model_output = network.get_output('softmax')

ckpt_saver = tf.train.Saver()
config = tf.ConfigProto(allow_soft_placement=True)

graph = tf.Graph()
with tf.Session(graph=graph, config=config) as sess:
    # Restore from checkpoint
    print('jainam')
    loader = tf.train.import_meta_graph(trained_checkpoint_prefix + '.meta')
    loader.restore(sess, trained_checkpoint_prefix)

    # Export checkpoint to SavedModel
    prediction_signature = tf.saved_model.signature_def_utils.predict_signature_def(
        {"input": model_input}, {"output": model_output})

    builder = tf.saved_model.builder.SavedModelBuilder(export_dir)
    builder.add_meta_graph_and_variables(sess,
                                         [tf.saved_model.tag_constants.SERVING],
                                         signature_def_map={
                                             "serving_default": prediction_signature},
                                         strip_default_attrs=True)
    builder.save()
