class EnvConfiguration:
    def __init__(self, encoding, agent_pos, agent_dir, goal_pos, carrying):
        self.encoding, self.agent_pos, self.agent_dir, self.goal_pos, self.carrying = encoding, agent_pos, agent_dir, goal_pos, carrying
    def get_configuration(self):
        return self.encoding, self.agent_pos, self.agent_dir, self.goal_pos, self.carrying