class EnvConfiguration:
    def __init__(self, encoding, agent_pos, agent_dir, goal_pos, carrying, passable, size, name="generic"):
        self.encoding, self.agent_pos, self.agent_dir, self.goal_pos, self.carrying, self.passable, self.size = encoding, agent_pos, agent_dir, goal_pos, carrying, passable, size
        self.name = name

        #print(f"Is passable? {self.passable}")
    def get_configuration(self):
        return self.encoding, self.agent_pos, self.agent_dir, self.goal_pos, self.carrying

    def get_passable(self):
        return self.passable

    def get_size(self):
        return self.size

    def get_name(self):
        return self.name