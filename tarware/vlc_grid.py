import numpy as np

class VLCGrid:
    """
    Abstração simples de grelha VLC.
    Para já, apenas filtramos usando a máscara devolvida pelo ambiente.
    """
    def filter_valid_actions(self, masks):
        """
        masks: numpy array (num_agents, action_size) de {0,1}
        retorna: dict {agent_index: [lista de ações macro válidas]}
        """
        valid = {}
        if isinstance(masks, np.ndarray):
            for i in range(masks.shape[0]):
                valid_actions = np.nonzero(masks[i])[0].tolist()
                valid[i] = valid_actions
        else:
            # fallback super defensivo
            valid = {}
        return valid
