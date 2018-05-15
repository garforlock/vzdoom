import random
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train(params, trainer):

    for i in range(params.episodes):
        logger.info('[ ==> Episode #: %s STARTED <== ]', i)
        trainer.game.new_episode()
        while not trainer.game.is_episode_finished():
            state = trainer.game.get_state()
            trainer.game.advance_action()
            last_action = trainer.game.get_last_action()
            reward = trainer.game.get_last_reward()
            img = state.screen_buffer

            logger.info('* State # %s', state.number)
            logger.info('* Game Vars:  %s', state.game_variables)
            logger.info('* Last Action: %s ', last_action)
            logger.info('* Reward: %s ', reward)
            logger.info('* ===================== ')

        logger.info('[ Total Reward Episode # %s: %s ]', i, trainer.game.get_total_reward())
        logger.info('[ ==> Episode #: %s  FINISHED <== ]', i)

