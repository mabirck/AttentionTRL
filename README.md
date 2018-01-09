# AttentionTRL
Attention Applied to Reinforcement

#Asynchornous Advantage Actor Critic(a3c/a2c) with Support to multi-tasking and Spatial Mechanism
#UNFINISHED

#Comands,(use args to find out functionalities):

python3 main.py --env-name "PongNoFrameskip-v4" "QbertNoFrameskip-v4" --act_func "relu" --recurrent-policy --att "spatial"

#PUT THE ENVS YOU WANT TO TRAIN MULTITASKING AFTER ENV NAME ARG

REFERENCES:

https://github.com/ikostrikov/pytorch-a2c-ppo-acktr

https://github.com/chasewind007/Attention-DQN

https://arxiv.org/pdf/1602.01783.pdf
https://arxiv.org/pdf/1707.04175.pdf
https://arxiv.org/pdf/1512.01693.pdf

https://blog.openai.com/baselines-acktr-a2c/

half_path to:

https://arxiv.org/pdf/1707.04175.pdf

https://www.researchgate.net/publication/320237340_Multi-Task_reinforcement_learning_An_hybrid_A3C_domain_approach
