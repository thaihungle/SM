# Source code for paper Beyond Surprise: Improving Exploration Through Surprise Novelty
preprint version:  https://arxiv.org/abs/2308.04836  
code reference for RND algoritms https://github.com/jcwleo/random-network-distillation-pytorch  


# Setup  
```
pip install -r requirements.txt
mkdir models
```
Install CUDA libraries (if possible) and other required packages
Please check the hyperparameter list in main.py

# Run baseline PPO (no intrinisc reward) 
```
CUDA_VISIBLE_DEVICES=0 python main.py  --EnvID MontezumaRevengeNoFrameskip-v4 --TrainMethod RND --IntCoef 0  --Seed 0 --PPOEps 0.1  --MaxEnvStep 50000000
```

# Run baseline RND 
```
CUDA_VISIBLE_DEVICES=0 python main.py  --EnvID MontezumaRevengeNoFrameskip-v4 --TrainMethod RND --IntCoef 1  --Seed 0 --PPOEps 0.1  --MaxEnvStep 50000000
```

# Run our method RND+SM 
```
CUDA_VISIBLE_DEVICES=0 python main.py  --EnvID MontezumaRevengeNoFrameskip-v4 --TrainMethod RND_SM --MemType full --IntCoef 1  --Seed 0 --PPOEps 0.1  --MaxEnvStep 50000000
```
