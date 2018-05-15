from doom.doom_trainer import DoomTrainer
from models.dummy.train import train as train_dummy
from models.human.train import train as train_human


def play(parameters):
    if parameters.model == 'human':
        play_human(parameters)
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
