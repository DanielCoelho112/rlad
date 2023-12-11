import torch
import torch.nn.functional as F
import numpy as np
import random
import cv2
import os

from colorama import Fore
from importlib import import_module
from collections import deque

from agents.models.memory.memory import ReplayBufferStorage, make_replay_loader
from agents.models.image.image import ParameterizedReg, LocalSignalMixing, custom_parameterized_aug_optimizer_builder
from utilities.controls import carla_control, PID
from utilities.networks import update_target_network, RandomShiftsAug


class Agent():
    def __init__(self, augmentation_config, vehicle_measurements_config, waypoints_config, traffic_light_config, image_config, agent_config, critic_config, actor_config, memory_config, maximum_speed, experiment_path, init_memory, n_steps):

        self.maximum_speed = maximum_speed
        self.experiment_path = experiment_path
        self.device = torch.device(agent_config['device'])

        # agent config
        self.gamma = agent_config['gamma']
        self.alpha = agent_config['alpha']
        self.automatic_entropy_tuning = agent_config['automatic_alpha']
        self.latent_route_planner_size = agent_config['latent_route_planner_size']
        self.latent_image_size = agent_config['latent_image_size']
        self.latent_vehcicle_measurement_size = agent_config['latent_vehicle_measurement_size']
        self.target_update_interval = agent_config['target_update_interval']
        self.batch_size = agent_config['batch_size']
        self.raw_state_info = self.parse_raw_state_info(
            memory_config['raw_state_info'])
        self.repeat_action = agent_config['repeat_action']
        self.n_step = agent_config['n_step']
        self.deque_size = agent_config['deque_size']
        self.num_workers = memory_config['num_workers']

        self.use_aug = augmentation_config['use_aug']
        
        self.traffic_light_update_freq = traffic_light_config[
            'update_frequency']

        # waypoint config
        self.num_waypoints = waypoints_config['num_waypoints']

        # critic config
        self.critic_tau = critic_config['tau']
        self.critic_update_freq = critic_config['update_frequency']

        # encoder config
        self.image_size = image_config['image_size']

        # actor config
        self.actor_update_freq = actor_config['update_frequency']

        if self.use_aug:
            self.aug = RandomShiftsAug(pad=augmentation_config['pad'])


        self.action_ctn = 0
        self.prev_action = None
        self.state_size = self.latent_image_size + \
            self.latent_route_planner_size + self.latent_vehcicle_measurement_size

        self.learn_ctn = 0

        if init_memory:
            experiment_name = experiment_path.split('/')[-1]
            replay_dir = f"{os.getenv('HOME')}/memory/{experiment_name}"
            self.replay_storage = ReplayBufferStorage(
                raw_state_info=self.raw_state_info, replay_dir=replay_dir)

            self.replay_loader = make_replay_loader(replay_dir=replay_dir, raw_state_info=self.raw_state_info, max_size=memory_config['capacity'], batch_size=self.batch_size,
                                                    num_workers=memory_config['num_workers'], nstep=agent_config['n_step'], discount=self.gamma)
            self._replay_iter = None

        else:
            self._replay_iter = None

        # image encoder
        aug = LocalSignalMixing(pad=2, fixed_batch=True)
        encoder_aug = ParameterizedReg(
            aug=aug, parameter_init=0.5, param_grad_fn='alix_param_grad', param_grad_fn_args=[3, 0.535, 1e-20])
        module_str, class_str = image_config['entry_point'].split(
            ':')
        _Class = getattr(import_module(module_str), class_str)
        self.image_encoder = _Class(
            input_channels=self.deque_size*3, latent_size=self.latent_image_size, aug=encoder_aug, target=False, checkpoint_dir=self.experiment_path, device=self.device)


        image_encoder_optim_builder = custom_parameterized_aug_optimizer_builder(
            encoder_lr=image_config['lr'], lr=2e-3, betas=[0.5, 0.999])
        self.image_encoder_optim = image_encoder_optim_builder(
            self.image_encoder)

        # traffic light branch
        module_str, class_str = traffic_light_config['entry_point'].split(
            ':')
        _Class = getattr(import_module(module_str), class_str)
        self.traffic_light_decoder = _Class(traffic_light_config=traffic_light_config, image_latent_size=self.latent_image_size,
                                                                 device=self.device, checkpoint_dir=self.experiment_path)

        # waypoint encoder
        module_str, class_str = waypoints_config['entry_point'].split(
            ':')
        _Class = getattr(import_module(module_str), class_str)
        self.waypoints_encoder = _Class(lr=waypoints_config['lr'], num_waypoints=waypoints_config['num_waypoints'], fc_dims=waypoints_config['fc_dims'],
                                       out_dims=waypoints_config['out_dims'], weight_decay=waypoints_config['weight_decay'], device=self.device, target=False, checkpoint_dir=self.experiment_path)

        # vehicle measurement encoder
        module_str, class_str = vehicle_measurements_config['entry_point'].split(
            ':')
        _Class = getattr(import_module(module_str), class_str)
        self.vm_encoder = _Class(lr=vehicle_measurements_config['lr'], num_inputs=vehicle_measurements_config['num_inputs'] * self.deque_size, fc_dims=vehicle_measurements_config['fc_dims'],
                                       out_dims=vehicle_measurements_config['out_dims'], weight_decay=vehicle_measurements_config['weight_decay'], device=self.device, target=False, checkpoint_dir=self.experiment_path)

        # critic
        module_str, class_str = critic_config['entry_point'].split(':')
        _Class = getattr(import_module(module_str), class_str)
        self.critic = _Class(num_inputs=self.state_size, fc1_dims=critic_config['fc1_dims'], fc2_dims=critic_config[
            'fc2_dims'], lr=critic_config['lr'], device=self.device, checkpoint_dir=self.experiment_path, target=False)

        self.critic_target = _Class(num_inputs=self.state_size, fc1_dims=critic_config['fc1_dims'], fc2_dims=critic_config[
            'fc2_dims'], lr=critic_config['lr'], device=self.device, checkpoint_dir=self.experiment_path, target=True)

        # hard update using tau=1.
        update_target_network(self.critic_target, self.critic, tau=1)

        # actor
        module_str, class_str = actor_config['entry_point'].split(':')
        _Class = getattr(import_module(module_str), class_str)
        self.policy = _Class(num_inputs=self.state_size, fc1_dims=actor_config['fc1_dims'], fc2_dims=actor_config['fc2_dims'], lr=actor_config['lr'], device=self.device,
                                   checkpoint_dir=self.experiment_path, log_sig_min=actor_config['log_sig_min'], log_sig_max=actor_config['log_sig_max'], epsilon=actor_config['epsilon'])

        self.pid = PID(kp=agent_config['pid']['kp'], ki=agent_config['pid']['ki'],
                       kd=agent_config['pid']['kd'], dt=agent_config['pid']['dt'], maximum_speed=maximum_speed)

        # encoder tl optim.
        image_encoder_optim_builder = custom_parameterized_aug_optimizer_builder(
            encoder_lr=traffic_light_config['lr'], lr=2e-3, betas=[0.5, 0.999])
        self.tl_image_encoder_optim = image_encoder_optim_builder(
            self.image_encoder)
        self.checkpoint_tl_image_encoder_optim = f"{self.experiment_path}/weights/optimizers/tl_image_encoder.pt"
   
        if self.automatic_entropy_tuning:
            self.target_entropy = - \
                torch.prod(torch.Tensor([2]).to(self.device)).item()
            self.log_alpha = torch.tensor(np.log(agent_config['alpha']), requires_grad=True, device=self.device)
            self.alpha_optim = torch.optim.Adam(
                [self.log_alpha], lr=agent_config['lr_alpha'])
     
    @property
    def replay_iter(self):
        if self._replay_iter is None:
            self._replay_iter = iter(self.replay_loader)
        return self._replay_iter

    @staticmethod
    def parse_raw_state_info(raw_state_info):
        for state_key, state_value in raw_state_info.items():
            for key, value in state_value.items():
                raw_state_info[state_key][key] = eval(value)
        return raw_state_info

    def encode(self, raw_state, detach=False, target=False):
        image = raw_state['central_rgb'] 

        vm_data = raw_state['vehicle_measurements']

        route_plan = raw_state['route_plan']  

        if target:
            with torch.no_grad():
                route_plan = self.waypoints_encoder_target(route_plan)
        else:
            route_plan = self.waypoints_encoder(route_plan)

        if target:
            with torch.no_grad():
                state_image = self.image_encoder_target(image) 
        else:
            state_image = self.image_encoder(image)
            
        if target:
            with torch.no_grad():
                state_vm = self.vm_encoder_target(vm_data)
        else:
            state_vm = self.vm_encoder(vm_data)
            
        if detach:
            state_image = state_image.detach()
            route_plan = route_plan.detach()
            state_vm = state_vm.detach()

        state = torch.cat([state_image, state_vm,
                          route_plan], dim=1) 

        return state

    @torch.no_grad()
    def choose_action(self, raw_state, step, greedy=False):

        current_velocity = self.get_current_speed(raw_state=raw_state)

        if self.action_ctn % self.repeat_action == 0:
            raw_state = self.filter_raw_state(raw_state=raw_state)
            raw_state = self.convert_raw_state_into_torch(
                raw_state=raw_state, unsqueeze=True)

            state = self.encode(raw_state=raw_state, detach=True)

            if greedy is False:
                action, _, _ = self.policy.sample(state)
            else:
               _, _, action = self.policy.sample(state)

            action = action.detach().cpu().numpy()[0]

        else:
            action = self.prev_action

        controls = carla_control(self.pid.get(
            action=action, velocity=current_velocity))

        self.action_ctn += 1
        self.prev_action = action

        return action, controls

    def random_action(self, raw_state):

        current_velocity = self.get_current_speed(raw_state=raw_state)

        if self.action_ctn % self.repeat_action == 0:

            action = np.asarray([random.uniform(0, 1), random.uniform(-1, 1)])
        else:
            action = self.prev_action

        controls = carla_control(self.pid.get(
            action=action, velocity=current_velocity))

        self.action_ctn += 1
        self.prev_action = action

        return action, controls

    def filter_raw_state(self, raw_state):
        raw_state_ = {}

        resized_image = cv2.resize(
            raw_state['central_rgb']['data'], self.raw_state_info['central_rgb']['shape'][1:3], interpolation=cv2.INTER_AREA)

        raw_state_['central_rgb'] = np.einsum(
            'kij->jki', resized_image)

        raw_state_['route_plan'] = np.array(
            raw_state['route_plan']['location'])[0:self.num_waypoints, 0:2].reshape(self.num_waypoints, 2)

        raw_state_speed = np.array(
            raw_state['speed']['speed'][0] / self.maximum_speed, dtype=np.float32).reshape(1)

        raw_state_steer = np.array(
            raw_state['control']['steer'][0]).reshape(1)

        raw_state_['vehicle_measurements'] = np.concatenate([raw_state_speed, raw_state_steer]).reshape(2)
        
        raw_state_['traffic_light_state'] = np.array(
            raw_state['traffic_light']['state']).reshape(1)

        self.update_deque_raw_state(raw_state=raw_state_)
        
        return raw_state_

    def convert_raw_state_into_torch(self, raw_state, unsqueeze=False):
        for key, value in raw_state.items():
            if unsqueeze:
                raw_state[key] = torch.from_numpy(
                    value).to(self.device).unsqueeze(0)
            else:
                raw_state[key] = torch.from_numpy(value).to(self.device)
        return raw_state

    def convert_raw_state_into_device(self, raw_state, unsqueeze=False):
        for key, value in raw_state.items():
            if unsqueeze:
                raw_state[key] = value.to(self.device).unsqueeze(0)
            else:
                raw_state[key] = value.to(self.device)
        return raw_state

    def remember(self, raw_state, action, reward, next_raw_state, done):
        raw_state = None
        next_raw_state = self.filter_raw_state(next_raw_state)
        self.replay_storage.add(
            action=action, reward=reward, next_raw_state=next_raw_state, done=done)

    def augment_raw_state(self, raw_state):
        raw_state['central_rbg'] = self.aug(raw_state['central_rgb'].float())
        return raw_state

    def clone_raw_state(self, raw_state):
        raw_state_ = {}
        for key, value in raw_state.items():
            raw_state_[key] = value.clone()

        return raw_state_

    def learn(self, step):
        self.learn_ctn += 1

        if self.learn_ctn < 256 or self.replay_storage._num_episodes < self.num_workers:
            losses = {}
            return losses

        if self.learn_ctn % self.critic_update_freq != 0 and self.learn_ctn % self.actor_update_freq != 0 and self.learn_ctn % self.target_update_interval != 0 and self.learn_ctn % self.traffic_light_decoder_update_freq != 0:
            losses = {}
            return losses

        # sample batch from memory.
        raw_state_batch, action_batch, reward_batch, discount_batch, next_raw_state_batch, done_batch = tuple(
            next(self.replay_iter))

        raw_state_batch = self.convert_raw_state_into_device(
            raw_state_batch)
        next_raw_state_batch = self.convert_raw_state_into_device(
            next_raw_state_batch)
        
        
        raw_state_batch = self.augment_raw_state(
            raw_state=raw_state_batch)
        next_raw_state_batch = self.augment_raw_state(
            raw_state=next_raw_state_batch)


        action_batch = action_batch.to(self.device)
        reward_batch = reward_batch.to(self.device)
        discount_batch = discount_batch.to(self.device)
        done_batch = done_batch.to(self.device)

        ########################################################################
        #######################   critic networks  #############################
        ########################################################################

        if self.learn_ctn % self.critic_update_freq == 0:
            q_loss = self.update_critics(
                raw_state_batch=raw_state_batch, action_batch=action_batch, reward_batch=reward_batch, discount_batch=discount_batch,
                next_raw_state_batch=next_raw_state_batch, done_batch=done_batch, step=step)
        else:
            q_loss = None

        ########################################################################
        #######################    actor-network  ##############################
        ########################################################################

        if self.learn_ctn % self.actor_update_freq == 0:
            policy_loss, alpha_loss, alpha_logs = self.update_policy_and_alpha(
                raw_state_batch=raw_state_batch)
        else:
            policy_loss = None, None, None

        ########################################################################
        ########################    target update  #############################
        ########################################################################

        if self.learn_ctn % self.target_update_interval == 0:
            update_target_network(target=self.critic_target,
                                  source=self.critic, tau=self.critic_tau)

        ########################################################################
        #######################   traffic_light update  ########################
        ########################################################################

        if self.learn_ctn % self.traffic_light_update_freq == 0:
            self.traffic_light_decoder.update(
                raw_state_batch=raw_state_batch, image_encoder=self.image_encoder, image_encoder_optim=self.tl_image_encoder_optim)


        losses = {'q_loss': q_loss,
                  'policy_loss': policy_loss}

        return losses

    def update_critics(self, raw_state_batch, action_batch, reward_batch, discount_batch, next_raw_state_batch, done_batch, step):

        # get Q_target using Q(s) = r + gamma*Q(s').

        with torch.no_grad():
            next_state_batch = self.encode(next_raw_state_batch, target=False)
            next_state_action, next_state_log_prob, _ = self.policy.sample(
                next_state_batch)
            q1_next_target, q2_next_target = self.critic_target(
                next_state_batch, next_state_action)
            min_q_next_target = torch.min(
                q1_next_target, q2_next_target) - self.alpha * next_state_log_prob
            q_value_target = reward_batch + (torch.logical_not(done_batch)) * \
                discount_batch * (min_q_next_target)

        state_batch = self.encode(raw_state_batch)
        # two q-functions to mitigate positive bias in the policy update step.
        q1, q2 = self.critic(state_batch, action_batch)
        q1_loss = F.mse_loss(q1, q_value_target)
        q2_loss = F.mse_loss(q2, q_value_target)
        q_loss = q1_loss + q2_loss

        self.vm_encoder.optimizer.zero_grad(set_to_none=True)
        self.waypoints_encoder.optimizer.zero_grad(set_to_none=True)
        self.image_encoder_optim.zero_grad(set_to_none=True)
        self.critic.optimizer.zero_grad(set_to_none=True)

        q_loss.backward()

        self.vm_encoder.optimizer.step()
        self.waypoints_encoder.optimizer.step()
        self.image_encoder_optim.step()
        self.critic.optimizer.step()

        return round(q_loss.item(), 4)

    def update_policy_and_alpha(self, raw_state_batch):
    
        state_batch = self.encode(raw_state_batch, detach=True)

        actions, log_prob, _ = self.policy.sample(state_batch)
        q1, q2 = self.critic(state_batch, actions)
        min_q = torch.min(q1, q2)

     
        policy_loss = ((self.alpha * log_prob) - min_q).mean()

        self.policy.optimizer.zero_grad()
        policy_loss.backward()
        self.policy.optimizer.step()

        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_prob +
                           self.target_entropy).detach()).mean()
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()
            self.alpha = self.log_alpha.exp()

            alpha_logs = self.alpha.clone()  

        return round(policy_loss.item(), 4), round(alpha_loss.item(), 4), round(alpha_logs.item(), 4)
    
    
    @staticmethod
    def get_current_speed(raw_state):
        return raw_state['speed']['speed'][0]

    def update_deque_raw_state(self, raw_state):
        self.central_rgb_deque.append(raw_state['central_rgb'])
        raw_state['central_rgb'] = np.concatenate(list(self.central_rgb_deque), axis=0)

        self.vm_deque.append(raw_state['vehicle_measurements'])
        raw_state['vehicle_measurements'] = np.concatenate(list(self.vm_deque), axis=0)

    def reset(self, raw_state):
        self.pid.reset()
        
        self.central_rgb_deque = deque([], maxlen=self.deque_size)
        self.vm_deque = deque([], maxlen=self.deque_size)
        
        # make sure the deque is full.
        for _ in range(self.deque_size):
            __ = self.filter_raw_state(raw_state=raw_state)

    def set_train_mode(self):
        self.critic.train()
        self.critic_target.train()
        self.policy.train()
        self.image_encoder.train()
        self.waypoints_encoder.train()
        self.vm_encoder.train()
        self.traffic_light_decoder.train()

    def set_eval_mode(self):
        self.critic.eval()
        self.critic_target.eval()
        self.policy.eval()
        self.image_encoder.eval()
        self.waypoints_encoder.eval()
        self.vm_encoder.eval()
        self.traffic_light_decoder.eval()

    def save_models(self, save_memory=False):
        print(f'{Fore.GREEN} saving models... {Fore.RESET}')

        self.critic.save_checkpoint()
        self.critic_target.save_checkpoint()
        self.policy.save_checkpoint()
        self.image_encoder.save_checkpoint()
        self.waypoints_encoder.save_checkpoint()
        self.traffic_light_decoder.save_checkpoint()
        self.vm_encoder.save_checkpoint()
        torch.save(self.tl_image_encoder_optim.state_dict(), self.checkpoint_tl_image_encoder_optim)
        if self.automatic_entropy_tuning:
            torch.save(self.log_alpha,
                       f'{self.experiment_path}/weights/log_alpha.pt')
            torch.save(self.alpha_optim.state_dict(),
                       f'{self.experiment_path}/weights/optimizers/log_alpha.pt')
    
    def load_models(self, save_memory=False):
        print(f'{Fore.GREEN} loading models... {Fore.RESET}')

        self.critic.load_checkpoint()
        self.critic_target.load_checkpoint()
        self.policy.load_checkpoint()
        self.image_encoder.load_checkpoint()
        self.waypoints_encoder.load_checkpoint()
        self.traffic_light_decoder.load_checkpoint()
        self.vm_encoder.load_checkpoint()
        self.tl_image_encoder_optim.load_state_dict(torch.load(self.checkpoint_tl_image_encoder_optim, map_location=self.device))
        if self.automatic_entropy_tuning:
            self.log_alpha = torch.load(
                f'{self.experiment_path}/weights/log_alpha.pt')
            self.alpha_optim.load_state_dict(
                torch.load(f'{self.experiment_path}/weights/optimizers/log_alpha.pt'))
            self.alpha = self.log_alpha.exp()

