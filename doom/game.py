from doom.doom_trainer import DoomTrainer
from models.dqn.ai import AI
from models.dqn.net import NET
from models.dqn.nstepprogress import NStepProgress
from models.dqn.replaymemory import ReplayMemory
from models.dqn.softmaxbody import SoftmaxBody
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
    model = NET(trainer.num_actions())
    softmax_body = SoftmaxBody(T=1)
    ai = AI(brain=model, body=softmax_body)
    n_steps = NStepProgress(trainer, ai, n_step=10)
    memory = ReplayMemory(n_steps=n_steps, capacity=10000)
    train_dqn(model, memory, n_steps)