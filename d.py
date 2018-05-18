from doom.game import play


class P:
    def __init__(self, model, config, scenario, episodes=10, n_steps=0, memory=0, skip_learning=False, load_model=False):
        self.model = model
        self.config = config
        self.scenario = scenario
        self.episodes = episodes
        self.n_steps = n_steps
        self.memory = memory
        self.skip_learning = skip_learning
        self.load_model = load_model


p = P("dqn", config="scenarios/rocket_basic.cfg", scenario="basic", skip_learning=False, load_model=True)
play(p)
