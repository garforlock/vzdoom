from doom.game import play


class P:
    def __init__(self, model, config, scenario, episodes, n_steps=0, memory=0):
        self.model = model
        self.config = config
        self.scenario = scenario
        self.episodes = episodes
        self.n_steps = n_steps
        self.memory = memory


p = P("dqn", "scenarios/basic.cfg", "basic", 10)
play(p)
