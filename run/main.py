import os
import yaml
import cv2
import argparse
import sys
import numpy as np

from yaml.loader import SafeLoader
from colorama import Fore
from importlib import import_module

from carla_env.envs.suites.endless_env import EndlessEnv
from carla_env.envs.suites.nocrash_env import NoCrashEnv
from carla_env.envs.suites.corl2017_env import CoRL2017Env
from carla_env.envs.suites.leaderboard_env import LeaderboardEnv
from run.src.log_file import LogFile
from run.src.evaluate_agent import evaluate_agent
from utilities.common import set_random_seed


def main():

    parser = argparse.ArgumentParser(description='Train/Test RL agents')
    parser.add_argument('-en', '--experiment_name', type=str, required=True)
    parser.add_argument('-ow', '--overwrite', action='store_true')
    parser.add_argument('-vm', '--visualize_monitor', action='store_true')
    parser.add_argument('-ttc', '--train_test_config',
                        type=str, default='train.yaml')
    parser.add_argument('-rt', '--resume_training', action='store_true')

    arglist = [x for x in sys.argv[1:] if not x.startswith('__')]
    args = vars(parser.parse_args(args=arglist))

    experiment_name = args['experiment_name']
    experiment_path = f'{os.getenv("HOME")}/results/rlad/{experiment_name}'
    train_test_config_name = args['train_test_config']

    train = True if train_test_config_name == 'train.yaml' else False
    if train:
        print(f"{Fore.WHITE} Initialize training! {Fore.RESET}")
    else:
        print(f"{Fore.WHITE} Initialize testing! {Fore.RESET}")

    RLAD_ROOT = os.getenv('RLAD_ROOT')
    observation_config_path = f"{RLAD_ROOT}/config/experiments/{experiment_name}/observation.yaml"
    policy_config_path = f"{RLAD_ROOT}/config/experiments/{experiment_name}/policy.yaml"
    train_test_config_path = f"{RLAD_ROOT}/config/experiments/{experiment_name}/{train_test_config_name}"

    with open(observation_config_path) as f:
        observation_config = yaml.load(f, Loader=SafeLoader)
    with open(policy_config_path) as f:
        policy_config = yaml.load(f, Loader=SafeLoader)
    with open(train_test_config_path) as f:
        train_test_config = yaml.load(f, Loader=SafeLoader)

    # TODO: check if we have more than one training environment (Curriculum Learning).
    env_config_name = train_test_config[0]['env_name']
    env_config_path = f"{RLAD_ROOT}/config/envs/{env_config_name}"
    with open(env_config_path) as f:
        env_config = yaml.load(f, Loader=SafeLoader)[0]

    # set random seed
    seed = train_test_config[0].get('seed', None)
    set_random_seed(seed=seed)

    # Load Environment.
    carla_map = env_config['env_configs'].get('carla_map', None)
    carla_fps = env_config['env_configs'].get('carla_fps', None)
    host = train_test_config[0].get('host', None)
    port = train_test_config[0].get('port', None)
    tm_port = train_test_config[0].get('tm_port', None)
    num_zombie_vehicles = env_config['env_configs'].get(
        'num_zombie_vehicles', None)
    num_zombie_walkers = env_config['env_configs'].get(
        'num_zombie_walkers', None)
    route_description = env_config['env_configs'].get(
        'route_description', None)
    weather_group = env_config['env_configs'].get('weather_group', None)
    terminal_configs = env_config['env_configs'].get('terminal', None)
    reward_configs = env_config['env_configs'].get('reward', None)
    background_traffic = env_config['env_configs'].get(
        'background_traffic', None)
    task_type = env_config['env_configs'].get('task_type', None)
    routes_group = env_config['env_configs'].get('routes_group', None)

    # init agent.
    policy_config['kwargs']['experiment_path'] = experiment_path
    maximum_speed = env_config['env_configs']['reward']['kwargs']['maximum_speed']
    policy_config['kwargs']['maximum_speed'] = maximum_speed
    policy_config['kwargs']['init_memory'] = train
    policy_config['kwargs']['n_steps'] = int(train_test_config[0]['n_steps'])
    module_str, class_str = policy_config['entry_point'].split(':')
    _Class = getattr(import_module(module_str), class_str)
    agent = _Class(**policy_config.get('kwargs', {}))

    # load weights.
    if not train:
        agent.load_models()
        agent.set_eval_mode()

    if args['resume_training']:
        agent.load_models()

    # init logfile.
    logfile = LogFile(experiment_path=experiment_path, observation_config=observation_config,
                      policy_config=policy_config, train_test_config=train_test_config,
                      train_test_config_name=train_test_config_name, overwrite=args['overwrite'],
                      fps=carla_fps, resume_training=args['resume_training'],)

    if env_config['env_id'].split('-')[0] == 'NoCrash':
        env = NoCrashEnv(carla_map=carla_map, host=host, port=port, obs_configs=observation_config,
                         reward_configs=reward_configs, terminal_configs=terminal_configs,
                         weather_group=weather_group, route_description=route_description,
                         background_traffic=background_traffic, carla_fps=carla_fps, tm_port=tm_port, seed=seed)
    elif env_config['env_id'].split('-')[0] == 'Endless':
        env = EndlessEnv(carla_map=carla_map, host=host, port=port, obs_configs=observation_config,
                         terminal_configs=terminal_configs, reward_configs=reward_configs,
                         num_zombie_vehicles=num_zombie_vehicles, num_zombie_walkers=num_zombie_walkers,
                         weather_group=weather_group, carla_fps=carla_fps, tm_port=tm_port, seed=seed)
    elif env_config['env_id'].split('-')[0] == 'CoRL2017':
        env = CoRL2017Env(carla_map=carla_map, host=host, port=port, obs_configs=observation_config,
                          terminal_configs=terminal_configs, reward_configs=reward_configs,
                          weather_group=weather_group, route_description=route_description, task_type=task_type,
                          carla_fps=carla_fps, tm_port=tm_port, seed=seed)
    elif env_config['env_id'].split('-')[0] == 'LeaderBoard':
        env = LeaderboardEnv(carla_map=carla_map, host=host, port=port, obs_configs=observation_config,
                             reward_configs=reward_configs, terminal_configs=terminal_configs, weather_group=weather_group,
                             routes_group=routes_group, carla_fps=carla_fps, tm_port=tm_port, seed=seed)
    else:
        env = None
        raise Exception('Unknown Environment!')

    if env_config['env_id'].split('-')[0] == 'Endless' and not train:
        raise Exception('Endless Environment cannot be use for test.')

    # Init Training/Testing.
    maximum_steps = train_test_config[0].get('n_steps', None)
    warmup_steps = train_test_config[0].get('warmup_steps', None)
    start_remembering = train_test_config[0].get('start_remembering')
    eval_freq = train_test_config[0].get('evaluate_frequency_steps', None)
    eval_average_episodes = train_test_config[0].get(
        'evaluate_average_episodes', None)

    # if n_episodes==-1 means we want to test all tasks of the environment.
    if maximum_steps == -1:
        n_episodes = len(env._all_tasks)
    else:
        n_episodes = int(10e10)

    steps = 0
    steps_eval = 0
    eval_idx = 0
    scores, steps_array = [], []
    best_score = -np.inf

    for ep in range(n_episodes):

        # evaluate agent.
        if (not train) or (steps_eval > eval_freq) or (steps == 0):
            num_episodes = eval_average_episodes if train else n_episodes
            eval_idx, best_score, scores, steps_array = evaluate_agent(env=env, agent=agent, logfile=logfile, step_train=steps, episode_train=ep-1,
                                                                       num_episodes=num_episodes, eval_idx=eval_idx, best_score=best_score, steps_array=steps_array, scores_eval=scores, maximum_speed=maximum_speed, train=train)
            if not train:
                break
            steps_eval = 0

        remember_cnt = 0
        done = False
        observation = env.reset()
        agent.reset(raw_state=observation)

        score = 0
        while not done:

            if steps < warmup_steps:
                action, controls = agent.random_action(raw_state=observation)
            else:
                action, controls = agent.choose_action(
                    raw_state=observation, step=steps)

            observation_, reward, done, info = env.step(controls)
            score += reward

            if remember_cnt > start_remembering:
                agent.remember(raw_state=observation, action=action,
                               reward=reward, next_raw_state=observation_, done=done)
            else:
                remember_cnt += 1

            _ = agent.learn(step=steps)

            if args['visualize_monitor']:
                image = observation_['monitor']['data']
                cv2.imshow('monitor', image)
                cv2.waitKey(1)

            observation = observation_
            steps += 1
            steps_eval += 1

        print(
            f" {Fore.BLUE} episode: {ep}, steps: {steps}, score: {score:.1f} {Fore.RESET}")

        if steps >= maximum_steps:
            break


if __name__ == '__main__':
    main()
