import tensorflow.keras as k

model = k.models.load_model("/home/z/PycharmProjects/pplcnt_model_gwt/Train/ssd_model/pb_model/saved_model")

model.save('test.h5')
