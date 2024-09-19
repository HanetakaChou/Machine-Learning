import os
import numpy
import tensorflow.lite
import gymnasium

# Data
env = gymnasium.make("LunarLander-v2", render_mode="human")

maximum_step_count = 1000

# Model
tflite_interpreter = tensorflow.lite.Interpreter(model_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "q-network.tflite"))

tflite_interpreter.allocate_tensors()

tflite_input_details = tflite_interpreter.get_input_details()
tflite_output_details = tflite_interpreter.get_output_details()

# Inference
state, _ = env.reset()

for _ in range(maximum_step_count):
    tflite_interpreter.set_tensor(tflite_input_details[0]["index"], numpy.expand_dims(state, axis=0).astype(numpy.float32))
    
    tflite_interpreter.invoke()   
    
    action = numpy.argmax(tflite_interpreter.get_tensor(tflite_output_details[0]["index"]))

    next_state, _, done, _, _ = env.step(action)
    
    env.render()
    
    state = next_state
    
    if done:
        break

env.close()
