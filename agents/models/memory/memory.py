import datetime
import io
import random
import traceback
import numpy as np
import torch


from torch.utils.data import IterableDataset
from collections import defaultdict
from pathlib import Path

# inspired in: https://github.com/facebookresearch/drqv2/blob/main/replay_buffer.py

def episode_len(episode):
    # substract -1 because the dummy first transition.
    # iter transforms something into a iterator, then we can call next to get one element of the iterator.
    return next(iter(episode.values())).shape[0] - 1

def save_episode(episode, fn):
    with io.BytesIO() as bs:
        np.savez_compressed(bs, **episode)
        bs.seek(0)
        with fn.open('wb') as f:
            f.write(bs.read())

def load_episode(fn):
    with fn.open('rb') as f:
        episode = np.load(f)
        episode = {k: episode[k] for k in episode.keys()}
        return episode

class ReplayBufferStorage:
    def __init__(self, raw_state_info, replay_dir):
        self._raw_state_info = raw_state_info
        self._replay_dir = Path(replay_dir)
        self._replay_dir.mkdir(exist_ok=True)
        self._current_episode = defaultdict(list)
        self._preload()
    
    def __len__(self):
        return self._num_transitions
    
    def add(self, action, reward, next_raw_state, done):
        
        action_np = np.full((2,), action, np.float32)
        reward_np = np.full((1,), reward, np.float32)
        done_np = np.full((1,), done, np.bool8)
        
        self._current_episode['action'].append(action_np)
        self._current_episode['reward'].append(reward_np)
        self._current_episode['done'].append(done_np)
        
        for state_key, state_value in next_raw_state.items():
            value = np.full(self._raw_state_info[state_key]['shape'], state_value, self._raw_state_info[state_key]['dtype'])
            self._current_episode[state_key].append(value)
        
        if done_np[0] == True:
            episode = dict()
            episode['action'] = np.array(self._current_episode['action'], np.float32)
            episode['reward'] = np.array(self._current_episode['reward'], np.float32)
            episode['done'] = np.array(self._current_episode['done'], np.bool8)
            
            for state_key, state_data in self._raw_state_info.items():
                episode[state_key] = np.array(self._current_episode[state_key], state_data['dtype'])
            
            self._current_episode = defaultdict(list)
            self._store_episode(episode)
    
    def _preload(self):
        self._num_episodes = 0
        self._num_transitions = 0
        for fn in self._replay_dir.glob('*.npz'):
            _, _, eps_len = fn.stem.split('_')
            self._num_episodes += 1
            self._num_transitions += int(eps_len)
    
    def _store_episode(self, episode):
        eps_idx = self._num_episodes
        eps_len = episode_len(episode)   
        self._num_episodes += 1
        self._num_transitions += eps_len
        ts = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
        eps_fn = f'{ts}_{eps_idx}_{eps_len}.npz'
        save_episode(episode, self._replay_dir / eps_fn)


class ReplayBuffer(IterableDataset):
    def __init__(self, replay_dir, raw_state_info, capacity, num_workers, nstep, discount,
                 fetch_every):
        self._replay_dir = Path(replay_dir)
        self._raw_state_info = raw_state_info
        self._size = 0
        self._max_size = capacity
        self._num_workers = max(1, num_workers)
        self._episode_fns = []
        self._episodes = dict()
        self._nstep = nstep
        self._discount = discount
        self._fetch_every = fetch_every
        self._samples_since_last_fetch = fetch_every
        self._save_snapshot = False
    
    def _sample_episode(self):
        eps_fn = random.choice(self._episode_fns)
        return self._episodes[eps_fn]
    
    def _store_episode(self, eps_fn):
        """
        load episode into memory.
        """
        try:
            episode = load_episode(eps_fn)
        except:
            return False 
        eps_len = episode_len(episode)
        while eps_len + self._size > self._max_size:
            early_eps_fn = self._episode_fns.pop(0)
            early_eps = self._episodes.pop(early_eps_fn)
            self._size -= episode_len(early_eps)
            if early_eps_fn.exists():
                early_eps_fn.unlink()
        self._episode_fns.append(eps_fn)
        self._episode_fns.sort()
        self._episodes[eps_fn] = episode
        self._size += eps_len
        if not self._save_snapshot:
            if eps_fn.exists():
                eps_fn.unlink()
        
        return True
    
    def _try_fetch(self):
        if self._samples_since_last_fetch < self._fetch_every:
            return
        self._samples_since_last_fetch = 0
        try:
            worker_id = torch.utils.data.get_worker_info().id
        except:
            worker_id = 0
        eps_fns = sorted(self._replay_dir.glob('*.npz'), reverse=True)
        fetched_size = 0
        for eps_fn in eps_fns:
            eps_idx, eps_len = [int(x) for x in eps_fn.stem.split('_')[1:]]
            if eps_idx % self._num_workers != worker_id:
                continue
            if eps_fn in self._episodes.keys():
                break
            if fetched_size + eps_len > self._max_size:
                break
            fetched_size += eps_len
            if not self._store_episode(eps_fn):
                break
    
    def _sample(self):
        try:
            self._try_fetch()
        except:
            traceback.print_exc()
        self._samples_since_last_fetch += 1
        # sample episode from memory.
        episode = self._sample_episode()
        # add +1 for the first dummy transition
        idx = np.random.randint(0, episode_len(episode) - self._nstep + 1) + 1
        
        raw_state = {}
        next_raw_state = {}
        for state_key, state_value in self._raw_state_info.items():
            raw_state[state_key] = episode[state_key][idx - 1]
            next_raw_state[state_key] = episode[state_key][idx + self._nstep - 1]
            
        action = episode['action'][idx]
        done = episode['done'][idx]
        
        reward = np.zeros(shape=(1,), dtype=np.float32)
        discount = np.ones(shape=(1,), dtype=np.float32)
        
        for i in range(self._nstep):
            step_reward = episode['reward'][idx + i]
            reward += discount * step_reward
            discount *=  self._discount
        
        return (raw_state, action, reward, discount, next_raw_state, done)
    
    def __iter__(self):
        while True:
            yield self._sample()

def _worker_init_fn(worker_id):
    seed = np.random.get_state()[1][0] + worker_id
    np.random.seed(seed)
    random.seed(seed)


def make_replay_loader(replay_dir, raw_state_info, max_size, batch_size, num_workers, nstep, discount):
    
    max_size_per_worker = max_size // max(1, num_workers)

    iterable = ReplayBuffer(replay_dir,
                            raw_state_info,
                            max_size_per_worker,
                            num_workers,
                            nstep,
                            discount,
                            fetch_every=1000)

    loader = torch.utils.data.DataLoader(iterable,
                                         batch_size=batch_size,
                                         num_workers=num_workers,
                                         pin_memory=True,
                                         worker_init_fn=_worker_init_fn)
    return loader

        