import numpy as np
from collections import namedtuple, deque

# Defining one Step
Step = namedtuple('Step', ['state', 'action', 'reward', 'done'])


# Making the AI progress on several (n_step) steps

class NStepProgress:

    def __init__(self, trainer, ai, n_step):
        self.ai = ai
        self.rewards = []
        self.trainer = trainer
        self.n_step = n_step

    def __iter__(self):

        self.trainer.new_episode()

        state = self.trainer.get_screen()
        history = deque()
        reward = 0.0
        while True:
            action = self.ai(np.array(state.unsqueeze(0)))[0][0]
            next_state, r, is_done, _ = self.env.step(action)
            reward += r
            history.append(Step(state=state, action=action, reward=r, done=is_done))
            while len(history) > self.n_step + 1:
                history.popleft()
            if len(history) == self.n_step + 1:
                yield tuple(history)
            state = next_state
            if is_done:
                if len(history) > self.n_step + 1:
                    history.popleft()
                while len(history) >= 1:
                    yield tuple(history)
                    history.popleft()
                self.rewards.append(reward)
                reward = 0.0
                state = self.env.reset()
                history.clear()

    def rewards_steps(self):
        rewards_steps = self.rewards
        self.rewards = []
        return rewards_steps
