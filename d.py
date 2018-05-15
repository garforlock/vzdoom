from doom.game import play


class P:
    def __init__(self, model, config, scenario, episodes):
        self.model = model
        self.config = config
        self.scenario = scenario
        self.episodes = episodes


p = P("human", "scenarios/basic.cfg", "basic", 10)
play(p)
