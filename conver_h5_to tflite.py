### ============================== CODE VERSION ==============================
import tensorflow as tf

converter = tf.lite.TFLiteConverter.from_keras_model_file('model.h5')    # from_model_file

tflite_model = converter.convert()
with open('model.tflite', 'wb') as f:
  f.write(tflite_model)



# converter = tf.lite.TFLiteConverter.from_keras_model(model)    # from_keras_model(model = tf.keras.models.Sequential())
# converter = tf.lite.TFLiteConverter.from_saved_model('/home/a/model/')    # from_saved_model_path

# class Squared(tf.Module):    # Create a model using low-level tf.* APIs
#   @tf.function(input_signature=[tf.TensorSpec(shape=[None], dtype=tf.float32)])
#   def __call__(self, x):
#     return tf.square(x)
# model = Squared()
# # (to run your model) result = Squared(5.0) # This prints "25.0"
# # (to generate a SavedModel) tf.saved_model.save(model, "saved_model_tf_dir")
# concrete_func = model.__call__.get_concrete_function()
# converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func], model)    # if TensorFlow 2.7 <, pass only first argument: from_concrete_functions([concrete_func])

  

  
### ============================== COMMAND LINE VERSION ==============================
model_path=/home/z/model
tflite_convert --keras_model_file=${model_path}/model.h5 --output_file=${model_path}/model.tflite    # H5 FILE
# tflite_convert --saved_model_dir=${model_path}/model --output_file=${model_path}/model.tflite    # SAVED MODEL
