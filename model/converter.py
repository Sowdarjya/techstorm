import tensorflow as tf

model = tf.keras.models.load_model('best_waste_model.h5')

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]  # Quantize weights
tflite_model = converter.convert()

with open('best_waste_model.tflite', 'wb') as f:
    f.write(tflite_model)

print("Model converted to TensorFlow Lite and saved as 'best_waste_model.tflite'")
