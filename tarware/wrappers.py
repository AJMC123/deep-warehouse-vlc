import numpy as np
import gymnasium as gym
from gymnasium import spaces

class TupleToMultiDiscrete(gym.Wrapper):
    """
    Converte um env multi-agente (obs=Tuple([...]), act=Tuple(Discrete(...)))
    para um env single-agent com:
      - observation_space: Box (concatena obs de todos os agentes)
      - action_space: MultiDiscrete([action_size]*num_agents)
    """

    def __init__(self, env):
        super().__init__(env)
        self.unwrapped_env = env.unwrapped

        # --- ação ---
        # original: spaces.Tuple( [spaces.Discrete(action_size)] * num_agents )
        # novo:     spaces.MultiDiscrete([action_size]*num_agents)
        self.num_agents = self.unwrapped_env.num_agents
        self.action_size = self.unwrapped_env.action_size
        self.action_space = spaces.MultiDiscrete([self.action_size] * self.num_agents)

        # --- observação ---
        # original: spaces.Tuple([...]) (um space por agente)
        # novo:     spaces.Box concat de todos (low/ high concatenados)
        orig = self.unwrapped_env.observation_space
        assert isinstance(orig, spaces.Tuple), "Esperava Tuple de observações"
        lows = []
        highs = []
        sizes = []
        for sp in orig.spaces:
            # assumimos Box para cada agente (é o caso do mapper global)
            if not isinstance(sp, spaces.Box):
                raise TypeError(f"Esperava Box por agente, obtive {type(sp)}")
            lows.append(sp.low.reshape(-1))
            highs.append(sp.high.reshape(-1))
            sizes.append(sp.shape[0] if len(sp.shape)==1 else int(np.prod(sp.shape)))
        self._obs_splits = np.array(sizes, dtype=int)
        low = np.concatenate(lows, axis=0).astype(np.float32)
        high = np.concatenate(highs, axis=0).astype(np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

    def _flatten_obs(self, obs_tuple):
        # obs_tuple: tuple/list de np.ndarrays (por agente)
        flats = [np.asarray(o, dtype=np.float32).reshape(-1) for o in obs_tuple]
        return np.concatenate(flats, axis=0)

    def reset(self, **kwargs):
        res = self.env.reset(**kwargs)
        if isinstance(res, tuple) and len(res) == 2 and isinstance(res[1], dict):
            obs, info = res
        else:
            obs, info = res, {}
        return self._flatten_obs(obs), info

    def step(self, action):
        # action: np.ndarray shape=(num_agents,) com inteiros
        if isinstance(action, (list, tuple)):
            macro = list(map(int, action))
        else:
            action = np.asarray(action).astype(np.int64).ravel()
            assert action.shape[0] == self.num_agents, f"Esperava {self.num_agents} ações, recebi {action.shape}"
            macro = [int(a) for a in action]

        step_res = self.env.step(macro)
        if isinstance(step_res, tuple) and len(step_res) == 5:
            obs, rewards, terminateds, truncateds, info = step_res
        elif isinstance(step_res, tuple) and len(step_res) == 4:
            obs, rewards, dones, info = step_res
            terminateds = dones
            truncateds = dones
        else:
            raise RuntimeError("Formato inesperado de step()")

        # rewards e dones são por agente → agregamos para single-agent
        if isinstance(rewards, (list, np.ndarray)):
            reward = float(np.sum(rewards))
            terminated = bool(np.all(terminateds)) if isinstance(terminateds, (list, np.ndarray)) else bool(terminateds)
            truncated = bool(np.all(truncateds)) if isinstance(truncateds, (list, np.ndarray)) else bool(truncateds)
        else:
            reward = float(rewards)
            terminated = bool(terminateds)
            truncated = bool(truncateds)

        return self._flatten_obs(obs), reward, terminated, truncated, info
