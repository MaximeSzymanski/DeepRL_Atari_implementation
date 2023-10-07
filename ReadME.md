
## Implemented Algorithms ##
- [DQN](https://arxiv.org/abs/1312.5602)
- [PPO](https://arxiv.org/abs/1707.06347)
- [DDPG](https://arxiv.org/abs/1509.02971)

## Results ##

### Lunar Lander (DQN, PPO, DDPG) ###
#### Reward Curves ####
| Algorithm         | Reward                                                                                                                                                                   |  
|-------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| DQN               | <img src=readme_files/reward_curves/DQN/LunarLanderv2/DQN_lunar_lander.svg width=100%>                                                                                   |    
| PPO               | <img src=readme_files/reward_curves/PPO/LunarLanderv2/PPO_lunar_lander.svg width=100%>  |   
| DDPG (Continuous) | <img src=readme_files/reward_curves/DDPG/LunarLanderv2/DDPG_lunar_lander.svg width=100%>                                                                                 |   

#### Training Video ####
 <img src="readme_files/video/ppo/LunarLanderv2/PPO_lunarlander.gif" width=100%> 


### Atari Pong (DQN, PPO) ###
#### Reward Curves ####
| Algorithm         | Reward                                                                       |
|-------------------|------------------------------------------------------------------------------|
| DQN               | <img src=readme_files/reward_curves/DQN/Pong/Reward_pong_DQN.png width=100%> |
| PPO               | <img src=readme_files/reward_curves/PPO/Pong/PPO_pong.svg width=100%>        |



#### Training video ####
| Step 0                                             | Step 900 000                                            | Step 1 500 000 (final)                                   |
|----------------------------------------------------|---------------------------------------------------------|----------------------------------------------------------|
| <img src="readme_files/video/ppo/Pong/step_0.gif"> | <img src="readme_files/video/ppo/Pong/step_900_000.gif"> |  <img src="readme_files/video/ppo/Pong/step_final.gif">  | 

### Atari Breakout (PPO) ###
| Step 0                                             | Step 900 000                                                 | Step 1 500 000 (final)                                   |
|----------------------------------------------------|--------------------------------------------------------------|----------------------------------------------------------|
| <img src="readme_files/video/ppo/Breakout/step_0.gif"> | <img src="readme_files/video/ppo/Breakout/step_900_000.gif"> |  <img src="readme_files/video/ppo/Breakout/step_final.gif">  |
