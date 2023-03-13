import numpy as np
import random
import copy


# based on: https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/labml_nn/rl/dqn/replay_buffer.py
class PrioritizedExperienceReplay():
    def __init__(self, capacity, alpha, beta0, beta_inc, raw_state_info, checkpoint_dir):
        self.capacity = capacity
        self.alpha = alpha 
        self.mem_cntr = 0
        self.beta = beta0
        self.beta_inc = beta_inc
        
        self.filename = f"{checkpoint_dir}/weights/memory.pickle"
        n_actions = 2
       
        self.raw_state_info = raw_state_info
        
        self.priority_sum = [0 for _ in range(2 * self.capacity)]
        self.priority_min = [float('inf') for _ in range(2 * self.capacity)]
        
        # current max priority to be assigned to new transitions.
        self.max_priority = 1.
        
        self.data = {
            'raw_state': {},
            'action': np.empty((self.capacity, n_actions), dtype=np.float32),
            'reward': np.empty((self.capacity, 1), dtype=np.float32),
            'next_raw_state': {},
            'done': np.empty((self.capacity, 1), dtype=np.bool8)
        }
        
        for state_key, state_value in self.raw_state_info.items():
            self.data['raw_state'][state_key] = np.empty((self.capacity, *state_value['shape']), dtype=state_value['dtype'])
        
        self.data['next_raw_state'] = copy.deepcopy(self.data['raw_state'])
        
        self.size = 0
        
        self.filename = f"{checkpoint_dir}/weights/memory.pickle"
           
    
    def push(self, raw_state, action, reward, next_raw_state, done):
        
        # get next available index.
        idx = self.mem_cntr
        
        # store data.
        self.data['action'][idx] = action
        self.data['reward'][idx] = reward
        self.data['done'][idx] = done
        
        for state_key, state_value in raw_state.items():
            self.data['raw_state'][state_key][idx] = state_value
        
        for state_key, state_value in next_raw_state.items():
            self.data['next_raw_state'][state_key][idx] = state_value
        
        # increment next available slot.
        self.mem_cntr = (idx + 1) % self.capacity
        
        # compute new size.
        self.size = min(self.capacity, self.size + 1)
        
        # $p_i^\alpha$, new samples get max_priority to make sure all samples are seen at least once.
        priority_alpha = self.max_priority ** self.alpha
        
        # Update the two segment trees for sum and minimum
        self._set_priority_min(idx, priority_alpha)
        self._set_priority_sum(idx, priority_alpha)
    
    def _set_priority_min(self, idx, priority_alpha):
        """
        #### Set priority in binary segment tree for minimum
        """

        # Leaf of the binary tree
        idx += self.capacity
        self.priority_min[idx] = priority_alpha

        # Update tree, by traversing along ancestors.
        # Continue until the root of the tree.
        while idx >= 2:
            # Get the index of the parent node
            idx //= 2
            
            # Value of the parent node is the minimum of it's two children
            self.priority_min[idx] = min(self.priority_min[2 * idx], self.priority_min[2 * idx + 1])

    def _set_priority_sum(self, idx, priority):
        """
        #### Set priority in binary segment tree for sum
        """

        # Leaf of the binary tree
        idx += self.capacity
        # Set the priority at the leaf
        self.priority_sum[idx] = priority

        # Update tree, by traversing along ancestors.
        # Continue until the root of the tree.
        while idx >= 2:
            # Get the index of the parent node
            idx //= 2
            # Value of the parent node is the sum of it's two children
            self.priority_sum[idx] = self.priority_sum[2 * idx] + self.priority_sum[2 * idx + 1]

    def _sum(self):
        """
        #### $\sum_k p_k^\alpha$
        """

        # The root node keeps the sum of all values
        return self.priority_sum[1]

    def _min(self):
        """
        #### $\min_k p_k^\alpha$
        """

        # The root node keeps the minimum of all values
        return self.priority_min[1]

    def find_prefix_sum_idx(self, prefix_sum):
        """
        #### Find largest $i$ such that $\sum_{k=1}^{i} p_k^\alpha  \le P$
        """

        # Start from the root
        idx = 1
        while idx < self.capacity:
            # If the sum of the left branch is higher than required sum
            if self.priority_sum[idx * 2] > prefix_sum:
                # Go to left branch of the tree
                idx = 2 * idx
            else:
                # Otherwise go to right branch and reduce the sum of left
                #  branch from required sum
                prefix_sum -= self.priority_sum[idx * 2]
                idx = 2 * idx + 1

        # We are at the leaf node. Subtract the capacity by the index in the tree
        # to get the index of actual value
        return idx - self.capacity

    def sample(self, batch_size):
        """
        ### Sample from buffer
        """
        weights = np.zeros(shape=batch_size, dtype=np.float32)
        idxs = np.zeros(shape=batch_size, dtype=np.int32)
        
        # Get sample indexes
        for i in range(batch_size):
            p = random.random() * self._sum()
            idx = self.find_prefix_sum_idx(p)
            idxs[i] = idx

        # $\min_i P(i) = \frac{\min_i p_i^\alpha}{\sum_k p_k^\alpha}$
        prob_min = self._min() / self._sum()
        # $\max_i w_i = \bigg(\frac{1}{N} \frac{1}{\min_i P(i)}\bigg)^\beta$
        max_weight = (prob_min * self.size) ** (-self.beta)

        for i in range(batch_size):
            idx = idxs[i]
            # $P(i) = \frac{p_i^\alpha}{\sum_k p_k^\alpha}$
            prob = self.priority_sum[idx + self.capacity] / self._sum()
            # $w_i = \bigg(\frac{1}{N} \frac{1}{P(i)}\bigg)^\beta$
            weight = (prob * self.size) ** (-self.beta)
            # Normalize by $\frac{1}{\max_i w_i}$,
            #  which also cancels off the $\frac{1}{N}$ term
            weights[i] = weight / max_weight

        # Get samples data
        actions = self.data['action'][idxs]
        rewards = self.data['reward'][idxs]
        dones = self.data['done'][idxs]
        
        raw_states = {}
        for state_key, state_value in self.data['raw_state'].items():
            raw_states[state_key] = state_value[idxs]
        
        next_raw_states = {}
        for state_key, state_value in self.data['next_raw_state'].items():
            next_raw_states[state_key] = state_value[idxs]
            
        self.beta = min(1., self.beta + self.beta_inc)
        
        return raw_states, actions, rewards, next_raw_states, dones, weights, idxs

    def update_priorities(self, indexes, priorities):
        """
        ### Update priorities
        """

        for idx, priority in zip(indexes, priorities):            
            # Set current max priority
            self.max_priority = max(self.max_priority, priority)

            # Calculate $p_i^\alpha$
            priority_alpha = priority ** self.alpha
            # Update the trees
            self._set_priority_min(idx, priority_alpha)
            self._set_priority_sum(idx, priority_alpha)

    def is_full(self):
        """
        ### Whether the buffer is full
        """
        return self.capacity == self.size

    def save_memory(self):
        print('save_memory not implemented')
        
    def load_memory(self):
        print('load_memory not implemented.')
            
    

# if __name__ == '__main__':
    
#     with open('../../simpleSAC/policy.yaml') as f:
#             policy_config = yaml.load(f, Loader=SafeLoader)
            
#     raw_state_info = policy_config['kwargs']['memory_config']['raw_state_info'] 
    
#     memory = ReplayMemory(capacity=3, raw_state_info=raw_state_info, checkpoint_dir=f'{os.getenv("HOME")}')
    
#     state = {'central_rgb' : np.zeros((3,360,640)),
#              'route_plan' : np.zeros((2))}

#     next_state = {'central_rgb' : np.zeros((3,360,640))+1,
#              'route_plan' : np.zeros((2))+1}
    
#     memory.push(raw_state=state, action=np.array([0,1,2]), reward=0, next_raw_state=next_state, done=0)
    
    
#     state = {'central_rgb' : np.zeros((3,360,640)) +2,
#              'route_plan' : np.zeros((2)) +2}

#     next_state = {'central_rgb' : np.zeros((3,360,640))+3,
#              'route_plan' : np.zeros((2))+3}
    
#     memory.push(raw_state=state, action=np.array([0,1,2]), reward=0, next_raw_state=next_state, done=0)


#     state = {'central_rgb' : np.zeros((3,360,640)) +4,
#              'route_plan' : np.zeros((2)) +5}

#     next_state = {'central_rgb' : np.zeros((3,360,640))+6,
#              'route_plan' : np.zeros((2))+7}
    
#     memory.push(raw_state=state, action=np.array([0,1,2]), reward=0, next_raw_state=next_state, done=0)
    

    
#     print('raw_state')

    
    
#     state, action, reward, next_state, done = memory.sample(2)
    
#     memory.load_memory()
#     # print(state)
#     # print('---')
#     # print(action)
    
#     # print(action.shape)
    
#     # a.save_buffer()
    
