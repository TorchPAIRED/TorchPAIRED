import matplotlib.pyplot as plt
import numpy as np
from absl import logging

from adversarial_minigrid import AdversarialEnv


def get_user_input_agent(env):
    """Validates user keyboard input to obtain valid actions for agent.
    Args:
      env: Instance of MultiGrid environment
    Returns:
      An array of integer actions.
    """
    max_action = max(env.Actions).value
    min_action = min(env.Actions).value

    # Print action commands for user convenience.
    print('Actions are:')
    for act in env.Actions:
        print('\t', str(act.value) + ':', act.name)

    prompt = 'Enter action for agent, or q to quit: '

    # Check user input
    while True:
        user_cmd = input(prompt)
        if user_cmd == 'q':
            return False

        actions = user_cmd.split(',')
        if len(actions) != env.n_agents:
            logging.info('Uh oh, you entered commands for %i agents but there is '
                         '%i. Try again?', len(actions), str(env.n_agents))
            continue

        valid = True
        for i, a in enumerate(actions):
            if not a.isdigit() or int(a) > max_action or int(a) < min_action:
                logging.info('Uh oh, action %i is invalid.', i)
                valid = False

        if valid:
            break
        else:
            logging.info('All actions must be an integer between %i and %i',
                         min_action, max_action)

    return [int(a) for a in actions if a]


def get_user_input_environment(env):
    """Validate action input for adversarial environment role.
    Args:
      env: Multigrid environment object.
      reparam: True if this is a reparameterized version of the environment.
    Returns:
      Integer action.
    """
    max_action = env.action_dim - 1
    min_action = 0

    # Check if using the reparameterized environment, in which case objects are
    # placed by the adversary using a different action space.
    obj_type = 'wall'
    # In the reparameterized environment.
    if env.step_count == 0:
        obj_type = 'goal'
    elif env.step_count == 1:
        obj_type = 'agent'
    prompt = 'Place ' + obj_type + ': enter an integer between ' + \
             str(max_action) + ' and ' + str(min_action) + ': '

    # Check user input
    while True:
        user_cmd = input(prompt)
        if user_cmd == 'q':
            return False

        is_r = False
        if user_cmd.startswith("r"):
            is_r = True
            user_cmd = user_cmd[1:]

        if (not user_cmd.isdigit() or int(user_cmd) > max_action or
                int(user_cmd) < min_action):
            print('Invalid action. All actions must be an integer between',
                  min_action, 'and', max_action, "(they can start with 'r' to repeat the action until the phase is over.")
        else:
            break

    return ("r" if is_r else "") + user_cmd


def main():
    env = AdversarialEnv(15)
    obs = env.reset()

    print('You are playing the role of the adversary to place blocks in '
          'the environment.')
    print('You will move through the spots in the grid in order from '
          'left to right, top to bottom. At each step, place the goal '
          '(0), the agent (1), a wall (2), or skip (3)')
    print(env)

    # Adversarial environment loop
    is_r = False
    while True:

        if not is_r:
            action = get_user_input_environment(env)
            if not action:
                break
            if action.startswith("r"):
                action = action[1:]
                is_r = True

        obs, _, done, _ = env.step(int(action))
        plt.imshow(env.render('rgb_array'))
        print(env)

        if done:
            break

    print('Finished. A total of', env.n_clutter_placed, 'blocks were placed.')
    print('Goal was placed at a distance of', env.distance_to_goal)
    print("The shortest possible path would be", env.shortest_path_length)
    print("Is the path passable?", env.passable)

    settings = env.get_env_settings()
    from settable_minigrid import SettableMinigridWrapper
    from gym_minigrid.minigrid import MiniGridEnv
    env = SettableMinigridWrapper(env.size, max_steps=250)
    plt.imshow(env.render('rgb_array'))
    plt.show()
    env.set(*settings)
    plt.imshow(env.render('rgb_array'))
    plt.show()

    # Agent-environment interaction loop
    for name in ['agent', 'adversary agent']:
        logging.info('Now control the %i', name)
        obs = env.reset_agent()
        reward_hist = []
        for i in range(env.max_steps + 1):
            print(env)

            logging.info('Observation:')
            for k in obs.keys():
                if isinstance((obs[k]), list):
                    logging.info(k, len(obs[k]))
                else:
                    logging.info(k, obs[k].shape)

            actions = get_user_input_agent(env)
            if not actions:
                return

            obs, rewards, done, _ = env.step(actions)

            for k in obs.keys():
                logging.info(k, np.array(obs[k]).shape)

            reward_hist.append(rewards)
            plt.imshow(env.render('rgb_array'))
            print('Step:', i)
            print('Rewards:', rewards)
            print('Collective reward history:', reward_hist)
            print('Cumulative collective reward:', np.sum(reward_hist))

            if done:
                logging.info('Game over')
                break


if __name__ == '__main__':
    main()
