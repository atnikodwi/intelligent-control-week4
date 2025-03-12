import gym
import numpy as np
import random
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam


class DQNAgent:
    def __init__(self, state_size, action_size):  # Perbaikan dari _init_
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.99  # Discount factor untuk MountainCar
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential([
            Input(shape=(self.state_size,)),
            Dense(128, activation='relu'),
            Dense(128, activation='relu'),
            Dense(self.action_size, activation='linear')
        ])
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state, verbose=0)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return

        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(next_state, verbose=0)[0])
            target_f = self.model.predict(state, verbose=0)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


def train_agent():
    env = gym.make('MountainCar-v0')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)
    
    episodes = 30
    batch_size = 32

    for e in range(episodes):
        state, _ = env.reset()
        state = np.reshape(state, [1, state_size])

        for time in range(200):  # Batas langkah untuk MountainCar
            action = agent.act(state)
            next_state, reward, done, truncated, _ = env.step(action)

            # Modifikasi reward: berikan insentif jika mobil mendekati puncak bukit
            if next_state[0] > -0.5:
                reward += 10

            done = done or truncated
            next_state = np.reshape(next_state, [1, state_size])
            agent.remember(state, action, reward, next_state, done)
            state = next_state

            if done:
                print(f"Episode: {e+1}/{episodes}, Steps: {time}, Epsilon: {agent.epsilon:.2f}")
                break

        if len(agent.memory) > batch_size:
            agent.replay(batch_size)

    agent.model.save("dqn_mountaincar.h5")


def test_agent():
    env = gym.make('MountainCar-v0', render_mode='human')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)
    agent.model.load_weights("dqn_mountaincar.h5")
    agent.epsilon = 0.001  # Minimalkan eksplorasi saat testing

    for e in range(5):
        state, _ = env.reset()
        state = np.reshape(state, [1, state_size])

        for time in range(200):  # Batas langkah
            env.render()
            action = agent.act(state)
            next_state, reward, done, truncated, _ = env.step(action)
            done = done or truncated
            state = np.reshape(next_state, [1, state_size])

            if done:
                print(f"Test Episode: {e+1}, Steps: {time}")
                break

    env.close()


if __name__ == "__main__":  # Perbaikan dari _name_
    mode = input("Masukkan mode (train/test): ")
    if mode == "train":
        train_agent()
    elif mode == "test":
        test_agent()
    else:
        print("Mode tidak dikenali!")
