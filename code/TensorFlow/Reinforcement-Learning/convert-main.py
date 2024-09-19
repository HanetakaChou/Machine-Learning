import os
import numpy
import tensorflow
import gymnasium

# Model
q_network = tensorflow.keras.models.load_model(os.path.join(os.path.dirname(os.path.abspath(__file__)), "q-network.keras"))

# Data
env = gymnasium.make("LunarLander-v2", render_mode="human")

maximum_step_count = 1000

# Inference  
state, _ = env.reset()

for _ in range(maximum_step_count):
    action = numpy.argmax(q_network(numpy.expand_dims(state, axis=0), training=False)[0])
    
    next_state, _, done, _, _ = env.step(action)
    
    env.render()
    
    state = next_state
    
    if done:
        break

env.close()

# Convert
tflite_model = tensorflow.lite.TFLiteConverter.from_keras_model(q_network).convert()

# Serialization
file_tflite_model_binary = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "q-network.tflite"), 'wb')
file_tflite_model_binary.write(tflite_model)
file_tflite_model_binary.close()

file_tflite_model_text = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "q-network.inl"), 'w')
file_tflite_model_text.write(", ".join([f"0X{ubyte:02X}" for ubyte in tflite_model]))
file_tflite_model_text.close()
