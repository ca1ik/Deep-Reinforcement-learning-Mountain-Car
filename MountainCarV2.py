    import gym # solve gym pip
    import tensorflow as tf
    import keras
    from tensorflow.keras import models
    from tensorflow.keras import layers
    from tensorflow.keras.optimizers import Adam
    from collections import deque
    import random
    import numpy as np
    import plt

    gpus = tf.config.list_physical_devices('GPU') # will edit for CPU , add app/option for CPU/GPU selection
    if gpus:
        try:
            # GPU configuration
            tf.config.set_virtual_device_configuration(
                gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)])  # 4GB memory limit
            print("GPU is being used.")
        except RuntimeError as e:
            print(e)
    else:
        print("GPU not found. Using CPU instead.")


    class MountainCarTrain:
        def __init__(self, env):
            self.env = env
            self.gamma = 0.99
            self.epsilon = 1
            self.epsilon_decay = 0.05
            self.epsilon_min = 0.01
            self.learningRate = 0.001
            self.replayBuffer = deque(maxlen=20000)
            self.trainNetwork = self.createNetwork()
            self.episodeNum = 400
            self.iterationNum = 201  # max is 200
            self.numPickFromBuffer = 32
            self.targetNetwork = self.createNetwork()
            self.targetNetwork.set_weights(self.trainNetwork.get_weights())

        def createNetwork(self):
            model = models.Sequential()
            state_shape = self.env.observation_space.shape
            model.add(layers.Dense(24, activation='relu', input_shape=state_shape))
            model.add(layers.Dense(48, activation='relu'))
            model.add(layers.Dense(self.env.action_space.n, activation='linear'))
            model.compile(loss='mse', optimizer=Adam(learning_rate=self.learningRate))
            return model

        def getBestAction(self, state):
            self.epsilon = max(self.epsilon_min, self.epsilon)
            if np.random.rand(1) < self.epsilon:
                action = np.random.randint(0, 3)
            else:
                action = np.argmax(self.trainNetwork.predict(state)[0])
            return action

        def trainFromBuffer(self):
            if len(self.replayBuffer) < self.numPickFromBuffer:
                return
            samples = random.sample(self.replayBuffer, self.numPickFromBuffer)
            states, actions, rewards, newstates, dones = zip(*samples)
            states = np.array(states).reshape(self.numPickFromBuffer, 2)
            newstates = np.array(newstates).reshape(self.numPickFromBuffer, 2)

            targets = self.trainNetwork.predict(states)
            new_state_targets = self.targetNetwork.predict(newstates)

            for i in range(self.numPickFromBuffer):
                target = targets[i]
                if dones[i]:
                    target[actions[i]] = rewards[i]
                else:
                    Q_future = max(new_state_targets[i])
                    target[actions[i]] = rewards[i] + Q_future * self.gamma

            self.trainNetwork.fit(states, targets, epochs=1, verbose=0)

        def orginalTry(self, currentState, eps):
            rewardSum = 0
            max_position = -99

            for i in range(self.iterationNum):
                bestAction = self.getBestAction(currentState)

                # Show the animation every 50 episodes
                if eps % 50 == 0:
                    self.env.render()

                # Adjust the unpacking to handle the new 'info' returned by env.step()
                new_state, reward, done, truncated, info = self.env.step(bestAction)  # Updated line
                new_state = new_state.reshape(1, 2)

                # Keep track of max position
                if new_state[0][0] > max_position:
                    max_position = new_state[0][0]

                # Adjust reward for task completion
                if new_state[0][0] >= 0.5:
                    reward += 10

                self.replayBuffer.append([currentState, bestAction, reward, new_state, done])

                # Train from buffer
                self.trainFromBuffer()

                rewardSum += reward
                currentState = new_state

                if done:
                    break

                print(f"Episode {eps} finished with reward: {rewardSum}, max position: {max_position}")
                print(f"Total reward: {rewardSum}, max position achieved: {max_position}")

                # Sync the target network
                self.targetNetwork.set_weights(self.trainNetwork.get_weights())

                print(f"Now epsilon is {max(self.epsilon_min, self.epsilon)}, the reward is {rewardSum}, maxPosition is {max_position}")
                self.epsilon -= self.epsilon_decay

        def start(self):
            for eps in range(self.episodeNum):
                # Reset environment and get the initial state
                currentState = self.env.reset()[0].reshape(1, 2)  # observation'ı doğru şekilde almak için [0] eklenmeli
                self.orginalTry(currentState, eps)


    env = gym.make('MountainCar-v0')
    dqn = MountainCarTrain(env=env)
    dqn.start()
