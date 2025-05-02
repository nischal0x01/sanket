import tensorflow as tf

# Load the model
model = tf.keras.models.load_model("nsl_model.h5")

# Print model summary
model.summary()