class EnvConfiguration:
    def __init__(self, encoding, agent_pos, agent_dir, goal_pos, carrying, passable):
        self.encoding, self.agent_pos, self.agent_dir, self.goal_pos, self.carrying, self.passable = encoding, agent_pos, agent_dir, goal_pos, carrying, passable

        #print(f"Is passable? {self.passable}")
    def get_configuration(self):
        return self.encoding, self.agent_pos, self.agent_dir, self.goal_pos, self.carrying

    def get_passable(self):
        return self.passable