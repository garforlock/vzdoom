import logging
import numpy as np
import torch
import torch.nn as nn
from tqdm import trange
from models.dqn.learn import Learn
from models.dqn.replay_memory import ReplayMemory
from models.dqn.net import NET
from time import time, sleep

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train(params, trainer):
    model_savefile = "./model-doom.pth"
    learning_rate = 0.00025
    discount_factor = 0.99
    epochs = 20
    learning_steps_per_epoch = 2000
    test_episodes_per_epoch = 100
    replay_memory_size = 10000
    frame_repeat = 12
    resolution = (30, 45)
    episodes_to_watch = 10
    batch_size = 64
    memory = ReplayMemory(capacity=replay_memory_size, resolution=resolution)
    criterion = nn.MSELoss()

    if params.load_model:
        model = torch.load(model_savefile)
    else:
        model = NET(len(trainer.actions))

    optimizer = torch.optim.SGD(model.parameters(), learning_rate)
    time_start = time()

    if not params.skip_learning:
        for epoch in range(epochs):
            logger.info('[ ==> Epoch #: %s STARTED  <== ]', epoch)
            train_episodes_finished = 0
            train_scores = []
            trainer.game.new_episode()
            learn = Learn(
                trainer,
                model,
                criterion,
                optimizer,
                memory,
                learning_rate,
                discount_factor,
                epochs,
                learning_steps_per_epoch,
                test_episodes_per_epoch,
                replay_memory_size,
                frame_repeat,
                resolution,
                episodes_to_watch,
                batch_size
            )

            logger.info('  --> Training AI...')

            for _ in trange(learning_steps_per_epoch, leave=True):
                learn.perform_learning_step(epoch)
                if trainer.game.is_episode_finished():
                    score = trainer.game.get_total_reward()
                    train_scores.append(score)
                    trainer.game.new_episode()
                    train_episodes_finished += 1

            logger.info("  --> Training AI finished, %d training episodes played.", train_episodes_finished)
            learn = None
            train_scores = np.array(train_scores)

            logger.info("  --> Trainig Results: mean: %.1f +/- %.1f, min: %.1f, max: %.1f",
                        train_scores.mean(),
                        train_scores.std(),
                        train_scores.min(),
                        train_scores.max())

            logger.info('  --> Testing AI...')
            learn = Learn(
                trainer,
                model,
                criterion,
                optimizer,
                memory,
                learning_rate,
                discount_factor,
                epochs,
                learning_steps_per_epoch,
                test_episodes_per_epoch,
                replay_memory_size,
                frame_repeat,
                resolution,
                episodes_to_watch,
                batch_size
            )
            test_scores = []
            for _ in trange(learn.test_episodes_per_epoch, leave=True):
                trainer.game.new_episode()
                while not trainer.game.is_episode_finished():
                    state = trainer.get_screen(resolution[0], resolution[1])
                    state = state.reshape([1, 1, resolution[0], resolution[1]])
                    best_action_index = learn.get_best_action(state)
                    trainer.game.make_action(trainer.actions[best_action_index], frame_repeat)
                r = trainer.game.get_total_reward()
                test_scores.append(r)

            test_scores = np.array(test_scores)

            logger.info("  --> Test Results: mean: %.1f +/- %.1f, min: %.1f, max: %.1f",
                        test_scores.mean(),
                        test_scores.std(),
                        test_scores.min(),
                        test_scores.max())

            logger.info("  --> Saving the network weigths to: %s", model_savefile)
            torch.save(model, model_savefile)
            logger.info("  --> Total elapsed time: %.2f minutes", ((time() - time_start) / 60.0))
            logger.info('[ ==> Epoch #: %s FINISEHD  <== ]', epoch)
            logger.info('')
            learn = None
    # End if

    trainer.stop_game()
    logger.info('[ ==> WATCHING AI WORK  <== ]')
    trainer.toogle_visible(True)

    trainer.start_game()

    for _ in range(episodes_to_watch):
        learn = Learn(
            trainer,
            model,
            criterion,
            optimizer,
            memory,
            learning_rate,
            discount_factor,
            epochs,
            learning_steps_per_epoch,
            test_episodes_per_epoch,
            replay_memory_size,
            frame_repeat,
            resolution,
            episodes_to_watch,
            batch_size
        )
        trainer.game.new_episode()
        while not trainer.game.is_episode_finished():
            state = trainer.get_screen(resolution[0], resolution[1])
            state = state.reshape([1, 1, resolution[0], resolution[1]])
            best_action_index = learn.get_best_action(state)

            # Instead of make_action(a, frame_repeat) in order to make the animation smooth
            trainer.game.set_action(trainer.actions[best_action_index])
            for _ in range(frame_repeat):
                trainer.game.advance_action()

        # Sleep between episodes
        sleep(1.0)
        score = trainer.game.get_total_reward()
        learn = None
        logger.info("  --> Total Score %s", score)

    # End for
    logger.info('[ ==> END WATCHING AI WORK  <== ]')
