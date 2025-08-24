import numpy as np

class CentralAgent:
    """
    Planeador central melhorado:
    - Usa as máscaras do env para ações válidas.
    - Prefere o alvo válido mais próximo (objetivo ou shelf) por agente.
    - Evita atribuir o mesmo alvo a vários agentes no mesmo step.
    """

    def __init__(self, vlc):
        self.vlc = vlc
        self.metrics = {}

    def reset_metrics(self):
        self.metrics = {
            'shelf_deliveries': 0,
            'clashes': 0,
            'stucks': 0,
            'agvs_distance_travelled': 0,
            'pickers_distance_travelled': 0,
            'agvs_idle_time': 0,
            'pickers_idle_time': 0,
            'total_reward': 0.0,
        }

    def _manhattan(self, ax, ay, bx, by):
        return abs(ax - bx) + abs(ay - by)

    def plan(self, obs, masks, env_unwrapped):
        """
        masks: (num_agents, action_size) em {0,1}
        env_unwrapped: Warehouse (tem agents, action_id_to_coords_map, goals, etc.)
        """
        num_agents = masks.shape[0]
        # mapa de id de ação -> (y, x)
        id2yx = env_unwrapped.action_id_to_coords_map
        goals_len = len(env_unwrapped.goals)

        # Preparar lista de candidatos válidos por agente
        valid = {}
        for i in range(num_agents):
            valid[i] = np.nonzero(masks[i])[0].tolist()

        # Evitar duplicar alvos
        reserved_targets = set()

        macro_actions = []
        for i in range(num_agents):
            agent = env_unwrapped.agents[i]
            ax, ay = agent.x, agent.y

            # candidatos válidos não incluem 0 (NOOP) por preferência
            candidates = [a for a in valid.get(i, []) if a != 0]

            best_a = 0
            best_d = None

            for a in candidates:
                # ignora alvos já reservados (para reduzir colisões)
                if a in reserved_targets:
                    continue

                # ação -> coordenadas (y, x)
                if a in id2yx:
                    y, x = id2yx[a]
                else:
                    # se não tiver mapeamento (deveria ter), salta
                    continue

                d = self._manhattan(ax, ay, x, y)

                # heurística simples:
                # - Se o agente estiver a carregar (AGV), preferir goals (a <= goals_len)
                # - Se for picker, não há goals válidos (máscara trata disso); vai ao AGV/shelf alvo
                if best_d is None or d < best_d:
                    best_d = d
                    best_a = a

            # Se nada escolhido (sem candidatos úteis), NOOP
            macro_actions.append(int(best_a))
            if best_a != 0:
                reserved_targets.add(best_a)

        return macro_actions

    def update_metrics(self, info, total_reward):
        self.metrics['shelf_deliveries'] += int(info.get('shelf_deliveries', 0))
        self.metrics['clashes'] += int(info.get('clashes', 0))
        self.metrics['stucks'] += int(info.get('stucks', 0))
        self.metrics['agvs_distance_travelled'] += int(info.get('agvs_distance_travelled', 0))
        self.metrics['pickers_distance_travelled'] += int(info.get('pickers_distance_travelled', 0))
        self.metrics['agvs_idle_time'] += int(info.get('agvs_idle_time', 0))
        self.metrics['pickers_idle_time'] += int(info.get('pickers_idle_time', 0))
        self.metrics['total_reward'] += float(total_reward)

    def get_metrics(self):
        return self.metrics
