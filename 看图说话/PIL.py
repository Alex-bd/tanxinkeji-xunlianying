# from PIL import Image as pil_image
# img = pil_image.open(filepath)
# img.resize(target_size,pil_image.NEAREST)
# return np.asarray(img,dtype=K.floatx())
#
# from keras.models import model_from_json
# json_file = open("vgg16_exported.json","r")
# loaded_model_json = json_file.read()
# json_file.close()
#
# model = model_from_json(loaded_model_json)
# model.load_weights("vgg16_exported.h5")