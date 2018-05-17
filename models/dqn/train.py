import logging
from time import sleep

import numpy as np
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
    model_savefile = "./model-doom.pth"
    model = Net(len(trainer.actions))
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), learning_rate)
    memory = ReplayMemory(capacity=replay_memory_size, resolution=(45, 30))
    batch_size = 64
    discount_factor = 0.99
    learn = Learn(trainer, model, criterion, optimizer, memory, batch_size, discount_factor)
    # Other parameters
    frame_repeat = 12
    resolution = (30, 45)
    episodes_to_watch = 10

    for epoch in range(epochs):
        logger.info('[ ==> Epoch #: %s STARTED  <== ]', epoch)
        train_episodes_finished = 0
        train_scores = []
        trainer.game.new_episode()
        logger.info('  ==> Training...')

        for _ in trange(learn.learning_steps_per_epoch, leave=True):
            learn.perform_learning_step(epoch)
            if trainer.game.is_episode_finished():
                score = trainer.game.get_total_reward()
                train_scores.append(score)
                trainer.game.new_episode()
                train_episodes_finished += 1

        logger.info("  ==>  %d training episodes played.", train_episodes_finished)
        train_scores = np.array(train_scores)

        logger.info("  ==> Trainig Results: mean: %.1f +/- %.1f, min: %.1f, max: %.1f",
                    train_scores.mean(),
                    train_scores.std(),
                    train_scores.min(),
                    train_scores.max())

        logger.info('  ==> Testing...')

        test_scores = []
        for _ in trange(learn.test_episodes_per_epoch, leave=False):
            trainer.game.new_episode()
            while not trainer.game.is_episode_finished():
                state = trainer.get_screen(30, 45)
                state = state.reshape([1, 1, 30, 45])
                best_action_index = learn.get_best_action(state)
                trainer.game.make_action(trainer.actions[best_action_index], frame_repeat)
            r = trainer.game.get_total_reward()
            test_scores.append(r)

        test_scores = np.array(test_scores)

        logger.info("  ==> Test Results: mean: %.1f +/- %.1f, min: %.1f, max: %.1f",
                    test_scores.mean(),
                    test_scores.std(),
                    test_scores.min(),
                    test_scores.max())

        logger.info("  ==> Saving the network weigths to: %s", model_savefile)
        torch.save(model, model_savefile)

        logger.info('[ ==> Epoch #: %s FINISEHD  <== ]', epoch)
        logger.info('')
