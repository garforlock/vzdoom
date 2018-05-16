
import logging
import torch
import torch.nn as nn


from tqdm import trange

from models.dqn.learn import Learn
from models.dqn.replaymemory import ReplayMemory
from test import Net

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create replay memory which will store the transitions


def train(params, trainer):

    epochs = 20
    learning_rate = 0.00025
    replay_memory_size = 10000

    model = Net(len(trainer.actions))
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), learning_rate)
    memory = ReplayMemory(capacity=replay_memory_size, resolution=(45, 30))
    batch_size = 64
    discount_factor = 0.99
    learn = Learn(trainer, model, criterion, optimizer, memory, batch_size, discount_factor)


    for epoch in range(epochs):
        logger.info('[ ==> Epoch #: %s STARTED  <== ]', epoch)
        train_episodes_finished = 0
        train_scores = []
        trainer.game.new_episode()

        for learning_step in trange(learn.learning_steps_per_epoch, leave=False):
            learn.perform_learning_step(epoch)
            if trainer.game.is_episode_finished():
                score = trainer.game.get_total_reward()
                train_scores.append(score)
                trainer.game.new_episode()
                train_episodes_finished += 1

            print("%d training episodes played." % train_episodes_finished)






