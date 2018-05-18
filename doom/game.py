from doom.doom_trainer import DoomTrainer
from models.dummy.train import train as train_dummy
from models.human.train import train as train_human
from models.dqn.train import train as train_dqn


def play(parameters):
    if parameters.model == 'human':
        play_human(parameters)

    elif parameters.model == 'dqn':
        play_dqn(parameters)
    else:
        play_dummy(parameters)


def play_human(parameters):
    trainer = DoomTrainer(parameters)
    trainer.start_game()
    train_human(parameters, trainer)


def play_dummy(parameters):
    trainer = DoomTrainer(parameters)
    trainer.start_game()
    train_dummy(parameters, trainer)


def play_dqn(parameters):
    trainer = DoomTrainer(parameters)
    trainer.start_game()
    train_dqn(parameters, trainer)
