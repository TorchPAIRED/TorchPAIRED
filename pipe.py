import threading

class AdvPipe:
    def __init__(self):
        self.adv_pro_lock, self.adv_anta_lock, self.ant_lock, self.pro_lock = threading.Lock(), threading.Lock(), threading.Lock(), threading.Lock()

        self.adv_pro_lock.acquire()
        self.adv_anta_lock.acquire()
        self.ant_lock.acquire()
        self.pro_lock.acquire()

        self.ant_rew = 0
        self.pro_rew = 0

    def get_rewards(self, configuration):
        print("adv asked for rews")

        self.configuration = configuration

        self.ant_lock.release()
        self.adv_anta_lock.acquire()

        self.pro_lock.release()
        self.adv_pro_lock.acquire()

        print(f"adv got rews {self.pro_rew - self.ant_rew}")

        return self.ant_rew - self.pro_rew

    def get_configuration(self, from_protag, reward):
        print("protag or antag asked for conf")

        if from_protag:
            callers_lock = self.pro_lock
            adversarys_lock = self.adv_pro_lock
            self.pro_rew = reward
        else:
            callers_lock = self.ant_lock
            adversarys_lock = self.adv_anta_lock
            self.ant_rew = reward

        adversarys_lock.release()   # let adv generate conf
        callers_lock.acquire()      # go to sleep

        print("protag or antag got conf")

        return self.configuration

    def unsafe_get_configuration(self):
        return self.configuration
