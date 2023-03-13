import numpy as np
import copy
import pickle


class ReplayMemory():
    def __init__(self, capacity, raw_state_info):
        self.capacity = capacity
        self.mem_cntr = 0
        n_actions = 2
        
        self.action_memory = np.empty((self.capacity, n_actions), dtype=np.float32)
        self.reward_memory = np.empty((self.capacity, 1), dtype=np.float32)
        self.terminal_memory = np.empty((self.capacity, 1), dtype=np.bool8)
        
        self.raw_state_info = raw_state_info
        
    
        self.raw_state_memory = {}
        
        for state_key, state_value in self.raw_state_info.items():
            self.raw_state_memory[state_key] = np.empty((self.capacity, *state_value['shape']), dtype=state_value['dtype'])
        
        self.next_raw_state_memory = copy.deepcopy(self.raw_state_memory)
            
    
    def push(self, raw_state, action, reward, next_raw_state, done):
        # circular buffer.
        index = self.mem_cntr % self.capacity
        
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done 
        
        for state_key, state_value in raw_state.items():
            self.raw_state_memory[state_key][index] = state_value
        
        for state_key, state_value in next_raw_state.items():
            self.next_raw_state_memory[state_key][index] = state_value
            
        self.mem_cntr += 1
            
            
    def sample(self, batch_size):
        max_mem = min(self.mem_cntr, self.capacity)
        
        batch = np.random.choice(max_mem, batch_size)
        
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        dones = self.terminal_memory[batch]
        
        raw_states = {}
        for state_key, state_value in self.raw_state_memory.items():
            raw_states[state_key] = state_value[batch] 

        next_raw_states = {}
        for state_key, state_value in self.next_raw_state_memory.items():
            next_raw_states[state_key] = state_value[batch] 
            
        
        return raw_states, actions, rewards, next_raw_states, dones
        
    def __len__(self):
        return len(self.capacity)


        
            
    

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
    
