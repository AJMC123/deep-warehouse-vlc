# Deep Warehouse — VLC & Centralized RL for AGVs

Este repositório contém um simulador leve de armazém com **AGVs** e **coordenação centralizada**, com suporte a **VLC (Visible Light Communication)** para posicionamento/logística e **heurística** vs **PPO** para tomada de decisão.  

## Instalação rápida
```bash
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
.venv\Scripts\activate    # Windows

pip install -r requirements.txt
```

⚠️ Nota: manter `numpy==1.26.4` no requirements.txt para evitar conflitos.

## Execução rápida
### Heurística
```bash
python scripts/run_heuristic.py --episodes 100 --output logs/heuristic.json
```

### Central Agent (Heurística)
```bash
python scripts/run_centralagent.py --episodes 1000 --output logs/central_heur.json
```

### Treino PPO
```bash
python scripts/run_centralagent_ppo.py --env tarware-tiny-3agvs-2pickers-globalobs-v1 --train_steps 200000
```

### Avaliar modelo PPO
```bash
python scripts/eval_saved_ppo.py --model logs/ppo_run/ppo_central_final.zip --episodes 50
```

---

## 📄 License
This project is based on the original [Task Assignment Robotic Warehouse](https://github.com/uoe-agents/task-assignment-robotic-warehouse) environment.  
The original LICENSE (included in this repository) applies.  

Modifications and extensions — such as VLC-based positioning, Centralized PPO training, enhanced heuristic logic, and metrics logging — were developed by **Antonio Carvalho** as part of the Master’s Thesis at **Instituto Superior de Engenharia de Lisboa (ISEL)**.
