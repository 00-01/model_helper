### ============================== CODE VERSION ==============================
import tensorflow as tf

### 0 ###
converter = tf.lite.TFLiteConverter.from_keras_model_file('model.h5')    # from_model_file

tflite_model = converter.convert()
with open('model.tflite', 'wb') as f:
  f.write(tflite_model)

### 1 ###
# converter = tf.lite.TFLiteConverter.from_keras_model(model)    # from_keras_model(model = tf.keras.models.Sequential())

### 2 ###
# converter = tf.lite.TFLiteConverter.from_saved_model('/home/a/model/')    # from_saved_model_path

### 3 ###
# class Squared(tf.Module):    # Create a model using low-level tf.* APIs
#   @tf.function(input_signature=[tf.TensorSpec(shape=[None], dtype=tf.float32)])
#   def __call__(self, x):
#     return tf.square(x)
# model = Squared()
# # (to run your model) result = Squared(5.0) # This prints "25.0"
# # (to generate a SavedModel) tf.saved_model.save(model, "saved_model_tf_dir")
# concrete_func = model.__call__.get_concrete_function()
# converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func], model)  # if TensorFlow 2.7 < current_tf_version, pass only first argument:([concrete_func])


### ============================== COMMAND LINE VERSION ==============================
model_base_path=/home/z/MODEL/
model=MVPC10
model_path=${model_base_path}${model}
tflite_convert --keras_model_file=${model_path}/model.h5 --output_file=${model_path}/model.tflite    # H5 FILE
# tflite_convert --saved_model_dir=${model_path} --output_file=${model_path}/model.tflite    # SAVED MODEL
tflite_convert \
  --input_shape=1,80,80,1 \
  --input_arrays=normalized_input_image_tensor \
  --output_arrays=TFLite_Detection_PostProcess,TFLite_Detection_PostProcess:1,TFLite_Detection_PostProcess:2,TFLite_Detection_PostProcess:3 \
  --allow_custom_ops \
  --graph_def_file=/content/models/research/fine_tuned_model/tflite/tflite_graph.pb \
  --output_file="${model_path}/model.tflite"
