import numpy as np

class PPOMemory:
    def __init__(self, batch_size):
        self.states = []
        self.masks = []
        self.probs = []
        self.vals = []
        self.actions = []
        self.rewards = []
        self.dones = []

        self.batch_size = batch_size

    def generate_batches(self):
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i+self.batch_size] for i in batch_start]

        return np.array(self.states),\
                np.array(self.masks),\
                np.array(self.actions),\
                np.array(self.probs),\
                np.array(self.vals),\
                np.array(self.rewards),\
                np.array(self.dones),\
                batches

    def store_memory(self, state, mask, action, probs, vals, reward, done):
        # if type(reward) != list:
        #     state = [state]
        #     mask = [mask]
        #     action = [action]
        #     probs = [probs]
        #     vals = [vals]
        #     reward = [reward]
        #     done = [done]
        # elif type(reward) == np.ndarray:
        #     state = state.tolist()
        #     mask = mask.tolist()
        #     action = action.tolist()
        #     probs = probs.tolist()
        #     vals = vals.tolist()
        #     reward = reward.tolist()
        #     done = done.tolist()
        self.states += [state.tolist()]
        self.masks += [mask.tolist()]
        self.actions += [action.tolist()]
        self.probs += [probs.item()]
        self.vals += [vals.item()]
        self.rewards += [reward]
        self.dones += [done]
        # print(self.states)
        # print(self.actions)
        # self.states.append(state)
        # self.actions.append(action)
        # self.probs.append(probs)
        # self.vals.append(vals)
        # self.rewards.append(reward)
        # self.dones.append(done)

    def clear_memory(self):
        self.states = []
        self.masks = []
        self.probs = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.vals = []