import os
import random
from collections import deque
import numpy
import tensorflow
import gymnasium
import tqdm
import matplotlib.pyplot

# Reference Environment
env = gymnasium.make("LunarLander-v2")
state_count = env.observation_space.shape[0]
action_count = env.action_space.n

# Deep Q-Network
q_network = tensorflow.keras.models.Sequential([
    tensorflow.keras.layers.Dense(units=128, activation="relu", input_shape=(state_count,)),  
    tensorflow.keras.layers.Dense(units=128, activation="relu"),  
    tensorflow.keras.layers.Dense(units=action_count, activation="linear")
    ])

# Target Network
target_q_network = tensorflow.keras.models.clone_model(q_network)

# Soft Update
tau = 1e-3

# Epsilon-Greedy Policy 
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.995

# Experience Replay
mini_batch_size = 128
# pre_fill_size = mini_batch_size * 10
replay_buffer = deque(maxlen=200000)

# Discount Factor
gamma = 0.995

# Learning Rate
alpha = 1e-3

episode_count = 7000
maximum_step_count = 1000

update_main_network_step_count = 4
# soft_update_target_network_step_count = 50

# Score
score_history = numpy.zeros(episode_count)
recent_scores_count = 100
score_threshold = 250

optimizer = tensorflow.keras.optimizers.Adam(learning_rate=alpha)

cost_function = tensorflow.keras.losses.MeanSquaredError()

# Reinforcement Learning Loop
with tqdm.tqdm(range(episode_count), desc="Episode Index", leave=True) as episode_progress_bar:
    for episode_index in episode_progress_bar:
        state, _ = env.reset()

        current_score = 0.0

        with tqdm.tqdm(range(maximum_step_count), desc="Step Index", leave=False) as step_progress_bar:
            for step_index in step_progress_bar:
                action = random.choice(range(action_count)) if numpy.random.rand() <= epsilon else numpy.argmax(q_network(numpy.expand_dims(state, axis=0), training=False)[0])
        
                next_state, reward, done, _, _ = env.step(action)

                replay_buffer.append((state.copy(), action, reward, next_state.copy(), 1.0 if done != False else 0.0))

                update_main_network = step_index % update_main_network_step_count == 0 and len(replay_buffer) > mini_batch_size

                if update_main_network:
                    experiences = random.sample(replay_buffer, k=mini_batch_size)

                    states = tensorflow.convert_to_tensor(numpy.array([experience[0] for experience in experiences]), dtype=tensorflow.float32)
                    actions = tensorflow.convert_to_tensor(numpy.array([experience[1] for experience in experiences]), dtype=tensorflow.int32)
                    rewards = tensorflow.convert_to_tensor(numpy.array([experience[2] for experience in experiences]), dtype=tensorflow.float32)
                    next_states = tensorflow.convert_to_tensor(numpy.array([experience[3] for experience in experiences]), dtype=tensorflow.float32)
                    dones = tensorflow.convert_to_tensor(numpy.array([experience[4] for experience in experiences]), dtype=tensorflow.float32)

                    with tensorflow.GradientTape() as tape:
                        # We can NOT use "predict" within the GradientTape
                        # q_values = q_network.predict(states)
                        # next_q_values = target_q_network.predict(next_states)

                        # Select Q-values for the corresponding actions
                        q_values = q_network(states)
                        # predictions = tensorflow.gather_nd(q_values, tensorflow.stack([tensorflow.range(q_values.shape[0]), actions], axis=1))
                        predictions = tensorflow.reduce_sum(q_values * tensorflow.one_hot(actions, action_count), axis=-1)

                        # Bellman Equation
                        next_q_values = target_q_network(next_states) 
                        targets = rewards + (1.0 - dones) * gamma * tensorflow.reduce_max(next_q_values, axis=-1)
        
                        cost = cost_function(targets, predictions) 

                    gradients = tape.gradient(cost, q_network.trainable_variables)   

                    optimizer.apply_gradients(zip(gradients, q_network.trainable_variables))

                # soft_update_target_network = step_index % soft_update_target_network_step_count == 0 and len(replay_buffer) > pre_fill_size
                soft_update_target_network = update_main_network

                if soft_update_target_network:
                    new_target_weights = [tau * main_weight + (1 - tau) * target_weight for main_weight, target_weight in zip(q_network.get_weights(), target_q_network.get_weights())]

                    target_q_network.set_weights(new_target_weights)

                state = next_state

                current_score += reward
        
                step_progress_bar.set_postfix({"Current Score": current_score})

                if done:
                    break

        epsilon = max(epsilon_min, epsilon*epsilon_decay)

        score_history[episode_index] = current_score

        recent_average_score = numpy.mean(score_history[max(0, episode_index - recent_scores_count):episode_index + 1]) if episode_index > recent_scores_count else 0.0

        episode_progress_bar.set_postfix({"Current Score": current_score, "Recent Avg Score": recent_average_score})

        # Early Stopping                                   
        if recent_average_score > score_threshold:
            break

env.close()

matplotlib.pyplot.clf()
matplotlib.pyplot.plot(score_history)
matplotlib.pyplot.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), "score-history.png"))

# Serialization
q_network.save(os.path.join(os.path.dirname(os.path.abspath(__file__)), "q-network.keras"), overwrite=True)
