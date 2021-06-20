import logging
import os
from collections import deque

import numpy as np

from pfrl.experiments.evaluator import Evaluator, save_agent


def train_paired(
    adv_agent,
    pro_agent,
    ant_agent,
    adv_env,
    pro_env,
    ant_env,
    steps,  # refers to adv's
    outdir,
    checkpoint_freq=None,
    log_interval=None,
    max_episode_len=None,
    step_offset=0,
    evaluator=None,
    successful_score=None,
    step_hooks=(),
    return_window_size=100,
    logger=None,
):
    """Train an agent in a batch environment.

    Args:
        adv_agent: Agent to train.
        env: Environment to train the agent against.
        steps (int): Number of total time steps for training.
        outdir (str): Path to the directory to output things.
        checkpoint_freq (int): frequency at which agents are stored.
        log_interval (int): Interval of logging.
        max_episode_len (int): Maximum episode length.
        step_offset (int): Time step from which training starts.
        return_window_size (int): Number of training episodes used to estimate
            the average returns of the current agent.
        successful_score (float): Finish training if the mean score is greater
            or equal to thisvalue if not None
        step_hooks (Sequence): Sequence of callable objects that accepts
            (env, agent, step) as arguments. They are called every step.
            See pfrl.experiments.hooks.
        logger (logging.Logger): Logger used in this function.
    Returns:
        List of evaluation episode stats dict.
    """

    logger = logger or logging.getLogger(__name__)
    recent_returns = deque(maxlen=return_window_size)

    num_envs = adv_env.num_envs
    episode_r = np.zeros(num_envs, dtype=np.float64)
    episode_idx = np.zeros(num_envs, dtype="i")
    episode_len = np.zeros(num_envs, dtype="i")

    regret_cb = train_agent_batch(
        pro_agent,
        pro_env,
        ant_agent,
        ant_env,
        outdir,
        checkpoint_freq=checkpoint_freq,
        max_episode_len=max_episode_len,
        logger=logger,
    )

    # o_0, r_0
    adv_obss = adv_env.reset()

    t = step_offset
    if hasattr(adv_agent, "t"):
        adv_agent.t = step_offset

    eval_stats_history = []  # List of evaluation episode stats dict
    try:
        while True:
            # a_t
            actions = adv_agent.batch_act(adv_obss)
            # o_{t+1}, r_{t+1}
            adv_obss, unused_rs, dones, infos = adv_env.step(actions)
            episode_len += 1

            if True in dones:
                unused_rs = regret_cb()

            # Agent observes the consequences
            adv_agent.batch_observe(adv_obss, unused_rs, dones, np.zeros(num_envs, dtype=bool))

            # Make mask. 0 if done/reset, 1 if pass
            end = dones
            not_end = np.logical_not(end)

            # For episodes that ends, do the following:
            #   1. increment the episode count
            #   2. record the return
            #   3. clear the record of rewards
            #   4. clear the record of the number of steps
            #   5. reset the env to start a new episode
            # 3-5 are skipped when training is already finished.
            episode_idx += end
            recent_returns.extend(episode_r[end])

            for _ in range(num_envs):
                t += 1
                if checkpoint_freq and t % checkpoint_freq == 0:
                    save_agent(adv_agent, t, outdir, logger, suffix="_checkpoint")

                for hook in step_hooks:
                    hook(adv_env, adv_agent, t)

            if (
                log_interval is not None
                and t >= log_interval
                and t % log_interval < num_envs
            ):
                logger.info(
                    "outdir:{} step:{} episode:{} last_R: {} average_R:{}".format(  # NOQA
                        outdir,
                        t,
                        np.sum(episode_idx),
                        recent_returns[-1] if recent_returns else np.nan,
                        np.mean(recent_returns) if recent_returns else np.nan,
                    )
                )
                logger.info("statistics: {}".format(adv_agent.get_statistics()))
            if evaluator:
                eval_score = evaluator.evaluate_if_necessary(
                    t=t, episodes=np.sum(episode_idx)
                )
                if eval_score is not None:
                    eval_stats = dict(adv_agent.get_statistics())
                    eval_stats["eval_score"] = eval_score
                    eval_stats_history.append(eval_stats)
                    if (
                        successful_score is not None
                        and evaluator.max_score >= successful_score
                    ):
                        break

            if t >= steps:
                break

            # Start new episodes if needed
            episode_r[end] = 0
            episode_len[end] = 0
            adv_obss = adv_env.reset(not_end)

    except (Exception, KeyboardInterrupt):
        # Save the current model before being killed
        save_agent(adv_agent, t, outdir, logger, suffix="_except")
        ant_env.close()
        pro_env.close()
        adv_env.close()
        if evaluator:
            evaluator.env.close()
        raise
    else:
        # Save the final model
        save_agent(adv_agent, t, outdir, logger, suffix="_finish")

    return eval_stats_history

class AgentTrainer:
    def __init__(self, agent, env, max_episode_len, checkpoint_freq, outdir, logger):
        self.agent = agent
        self.env = env
        self.max_episode_len = max_episode_len
        self.checkpoint_freq = checkpoint_freq
        self.outdir = outdir
        self.logger = logger

        num_envs = env.num_envs
        self.num_envs = env.num_envs

        #episode_r = np.zeros(num_envs, dtype=np.float64)
        self.episode_len = np.zeros(num_envs, dtype="i")

        self.last_obss = env.reset()

        self.t = 0
        if hasattr(agent, "t"):
            agent.t = 0

        self.waiters_rews = np.zeros(num_envs)
        self.waiters_obss = np.zeros((num_envs, env.observation_space.low.shape))

        eval_stats_history = []  # List of evaluation episode stats dict

    def init_episodes(self):
        self.episode_idx = np.zeros(self.num_envs, dtype="i")

    def get_step(self):
        actions = self.agent.batch_act(self.last_obss)
        obss, rs, dones, infos = self.env.step(actions)

        self.last_obss = obss
        self.last_dones = dones
        self.last_rs = rs
        self.last_infos = infos

        #self.episode_r += rs
        self.episode_len += 1

        old_waiters_rews = self.waiters_rews.copy()
        old_waiters_obss = self.waiters_obss.copy()
        self.waiters_rews[dones] = rs[dones]
        self.waiters_rews[old_waiters_rews > 0] = old_waiters_rews
        self.waiters_obss[dones] = obss[dones]
        self.waiters_obss[old_waiters_rews > 0] = old_waiters_obss

        return self.waiters_rews

    def do_update(self, unlocked_waiters, regret):
        if self.max_episode_len is None:
            resets = np.zeros(self.num_envs, dtype=bool)
        else:
            resets = self.episode_len == self.max_episode_len
        resets = np.logical_or(
            resets, [info.get("needs_reset", False) for info in self.last_infos]
        )

        self.waiters_rews[unlocked_waiters] = 0
        waiters_obss = self.waiters_obss.copy()
        self.waiters_obss[unlocked_waiters] = 0

        obss = self.last_obss
        obss[unlocked_waiters] = waiters_obss[unlocked_waiters]
        rews = regret
        dones = self.last_dones


        upd_obss = np.delete(obss, self.waiters_rews > 0)
        upd_rews = np.delete(rews, self.waiters_rews > 0)
        upd_dones = np.delete(dones, self.waiters_rews > 0)
        upd_resets = np.delete(resets, self.waiters_rews > 0)

        # Agent observes the consequences
        self.agent.batch_observe(upd_obss, upd_rews, upd_dones, upd_resets)

        dones[self.waiters_rews > 0] = False
        resets[self.waiters_rews > 0] = False

        end = np.logical_or(resets, dones)
        not_end = np.logical_not(end)

        self.episode_idx += end

        for _ in range(self.num_envs):
            self.t += 1
            if self.checkpoint_freq and self.t % self.checkpoint_freq == 0:
                save_agent(self.agent, self.t, self.outdir, self.logger, suffix="_pro_checkpoint")
        """
        if (
                self.log_interval is not None
                and self.t >= self.log_interval
                and self.t % self.log_interval < self.num_envs
        ):
            self.logger.info(
                "outdir:{} step:{} episode:{} last_R: {} average_R:{}".format(  # NOQA
                    self.outdir,
                    self.t,
                    np.sum(self.episode_idx),
                    self.recent_returns[-1] if self.recent_returns else np.nan,
                    np.mean(self.recent_returns) if self.recent_returns else np.nan,
                )
            )
            self.logger.info("statistics: {}".format(self.agent.get_statistics()))
        """
        # Start new episodes if needed
        #episode_r[end] = 0
        #episode_len[end] = 0
        self.last_obss = self.env.reset(not_end)
        return self.episode_idx


def train_agent_batch(
    pro_agent,
    pro_env,
    ant_agent,
    ant_env,
    outdir,
    checkpoint_freq=None,
    max_episode_len=None,
    logger=None,
):
    """Train an agent in a batch environment.

    Args:
        agent: Agent to train.
        env: Environment to train the agent against.
        steps (int): Number of total time steps for training.
        outdir (str): Path to the directory to output things.
        checkpoint_freq (int): frequency at which agents are stored.
        log_interval (int): Interval of logging.
        max_episode_len (int): Maximum episode length.
        step_offset (int): Time step from which training starts.
        return_window_size (int): Number of training episodes used to estimate
            the average returns of the current agent.
        successful_score (float): Finish training if the mean score is greater
            or equal to thisvalue if not None
        step_hooks (Sequence): Sequence of callable objects that accepts
            (env, agent, step) as arguments. They are called every step.
            See pfrl.experiments.hooks.
        logger (logging.Logger): Logger used in this function.
    Returns:
        List of evaluation episode stats dict.
    """
    #recent_returns = deque(maxlen=return_window_size)
    ant = AgentTrainer(ant_agent, ant_env, max_episode_len, checkpoint_freq, outdir, logger)
    pro = AgentTrainer(pro_agent, pro_env, max_episode_len, checkpoint_freq, outdir, logger)

    def regret_episode():
        num_envs = ant.num_envs
        episode_regret = np.zeros(num_envs, dtype=np.float64)
        try:
            while True:
                ant_waiters_rew = ant.get_step()
                pro_waiters_rew = pro.get_step()

                # regret
                unlocked_waiters = np.logical_and(ant_waiters_rew > 0, pro_waiters_rew > 0)
                regret = ant_waiters_rew - pro_waiters_rew
                episode_regret += regret

                a = ant.do_update(unlocked_waiters, regret)
                p = pro.do_update(unlocked_waiters, regret)

                if (a > 2).count_nonzero() == 0:
                    break

        except (Exception, KeyboardInterrupt):
            raise
        return episode_regret
    return regret_episode