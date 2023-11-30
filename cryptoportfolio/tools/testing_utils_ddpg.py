from stable_baselines3 import DDPG
from cryptoportfolio.rlagent.rlagent_ddpg import RLAgent as ddpg_agent
from cryptoportfolio.rlagent.network import CustomCNN_DDPG
import datetime, time
import pandas as pd
import numpy as np
import torch.nn.functional as F
import torch

"""
def sample_ddpg_params(trial: optuna.Trial) -> Dict[str, Any]:

    gamma = trial.suggest_categorical("gamma", [0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999])
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 100, 128, 256, 512, 1024, 2048])
    buffer_size = trial.suggest_categorical("buffer_size", [int(1e4), int(1e5), int(1e6)])
    # Polyak coeff
    tau = trial.suggest_categorical("tau", [0.001, 0.005, 0.01, 0.02, 0.05, 0.08])

    train_freq = trial.suggest_categorical("train_freq", [1, 4, 8, 16, 32, 64, 128, 256, 512])
    gradient_steps = train_freq

    noise_type = trial.suggest_categorical("noise_type", ["ornstein-uhlenbeck", "normal", None])
    noise_std = trial.suggest_uniform("noise_std", 0, 1)

    # NOTE: Add "verybig" to net_arch when tuning HER (see TD3)
    net_arch = trial.suggest_categorical("net_arch", ["small", "medium", "big"])
    # activation_fn = trial.suggest_categorical('activation_fn', [nn.Tanh, nn.ReLU, nn.ELU, nn.LeakyReLU])

    net_arch = {
        "small": [64, 64],
        "medium": [256, 256],
        "big": [400, 300],
    }[net_arch]

    hyperparams = {
        "gamma": gamma,
        "tau": tau,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "buffer_size": buffer_size,
        "train_freq": train_freq,
        "gradient_steps": gradient_steps,
        "policy_kwargs": dict(net_arch=net_arch),
    }

    if noise_type == "normal":
        hyperparams["action_noise"] = NormalActionNoise(
            mean=np.zeros(trial.n_actions), sigma=noise_std * np.ones(trial.n_actions)
        )
    elif noise_type == "ornstein-uhlenbeck":
        hyperparams["action_noise"] = OrnsteinUhlenbeckActionNoise(
            mean=np.zeros(trial.n_actions), sigma=noise_std * np.ones(trial.n_actions)
        )

    if trial.using_her_replay_buffer:
        hyperparams = sample_her_params(trial, hyperparams)

    return hyperparams

"""


def train_ddpg(window_size, buffer_size, batch_size, learning_starts, gradient_steps, 
          total_iterations, learning_rate, year, tb_log_name, features, 
          model=None, synthetic=False, split="whole"):

        agent = ddpg_agent(lookback_window_size=window_size, features=features)
        agent.generate_portfolio(year=year, synthetic=synthetic, split=split)
        agent.set_batch_size(batch_size)
        
        #n_steps_per_episode = agent._crash_length - agent._lookback_window_size
        #u = int(total_timesteps / n_steps_per_episode) + 1 
        #buffer_size = u * n_steps_per_episode
        agent.set_weight_storage(buffer_size)
        total_timesteps = total_iterations * agent._crash_length

        timer = int(time.time())
        logdir = f"logs/ddpg/{timer}"

        if model is None:
            
            policy_kwargs = dict(
                features_extractor_class=CustomCNN_DDPG,
                features_extractor_kwargs=dict(features_dim=13, agent_env=agent),
            )
            
            model = DDPG(policy="CnnPolicy", 
                            env=agent,
                            device="auto",
                            buffer_size=buffer_size,
                            learning_rate=learning_rate,
                            batch_size=batch_size,
                            learning_starts=learning_starts,
                            gradient_steps=gradient_steps,
                            policy_kwargs=policy_kwargs,
                            seed=0,
                            verbose=1,
                            #tensorboard_log=logdir
                            )
        else:
            # Necessary so that agent._weight_storage and the 
            # indices from the samples during training are correct
            model.replay_buffer = model.replay_buffer_class(
                model.buffer_size,
                model.observation_space,
                model.action_space,
                device=model.device,
                n_envs=model.n_envs,
                optimize_memory_usage=model.optimize_memory_usage,
                **model.replay_buffer_kwargs,
            )
            # Model exists and is trained on new data set
            model.env.envs[0].env = agent
        
        # Starts the timer
        start = datetime.datetime.now() 
        model.learn(total_timesteps, tb_log_name=tb_log_name)#, reset_num_timesteps=True)
        runtime = (datetime.datetime.now() - start).total_seconds() 

        return model, runtime

def get_hyperparams_ddpg():

    hyperparameters_ddpg = {"window_size":[20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200],
                    "batch_size":[10,20,30,40,50,60,70,80,90,100,110,120,130,140,150],
                    "learning_rate":[1e-05,2e-05,3e-05,4e-05,5e-05,6e-05,7e-05,8e-05,9e-05,
                                        1e-04,2e-04,3e-04,4e-04,5e-04,6e-04,7e-04,8e-04,9e-04, 
                                        1e-03,2e-03,3e-03,4e-03,5e-03,6e-03,7e-03,8e-03,9e-03,
                                        1e-02,2e-02,3e-02,4e-02,5e-02,6e-02,7e-02,8e-02,9e-02],
                    "gradient_steps":[100,200,500,600,700,800,900,1000,
                                        1500,2000,2500,3000],#,3000,3500,4000],
                    "learning_starts":[10,20,30,40,50,60,70,80,90,100,110,120,130,140,150],
                    "buffer_size":[1000,2000,3000,4000,5000,10000,15000,20000],
                    "features":[
                        [],
                        ["RSI"],
                        ["mcd", "mcd_signal"],
                        ["RSI", "mcd", "mcd_signal"],
                        ["RSI", "SMA_50", "SMA_200"],
                        ["stoch_oszillator", "stoch_oszillator_signal"],
                        ["RSI", "stoch_oszillator", "stoch_oszillator_signal"],
                        ["Bollinger_middle" ,"Bollinger_low" ,"Bollinger_high"],
                        ["SMA_50", "SMA_200"],
                        ["mcd", "mcd_signal", "Bollinger_middle" ,"Bollinger_low" ,"Bollinger_high"],
                        ["RSI", "SMA_50", "SMA_200", "Bollinger_middle" ,"Bollinger_low" ,"Bollinger_high"],
                        ["mcd", "mcd_signal", "stoch_oszillator", "stoch_oszillator_signal"],
                        ["RSI", "SMA_50", "SMA_200", "Bollinger_middle" ,"Bollinger_low" ,"Bollinger_high", 
                            "mcd", "mcd_signal", "stoch_oszillator", "stoch_oszillator_signal"]
                        ]
                    }

    for key, value in hyperparameters_ddpg.items():
            if key == "window_size":
                window_size = np.random.choice(value, 1)[0]
                #print("window size:", window_size)
            elif key == "learning_rate":
                learning_rate = np.random.choice(value, 1)[0]
                #print("learning rate:", learning_rate)
            elif key == "batch_size":
                batch_size = np.random.choice(value, 1)[0]
                #print("batch size:", batch_size)
            elif key == "gradient_steps":
                gradient_steps = np.random.choice(value, 1)[0]
                #print("gradient_steps:", gradient_steps)
            elif key == "buffer_size":
                #buffer_size = test_agent._crash_length - window_size + 1 
                buffer_size = np.random.choice(value, 1)[0]
                #print("buffer_size:", buffer_size)
            elif key == "learning_starts":
                learning_starts = np.random.choice(value, 1)[0]
                #print("learning_starts:", learning_starts)
            elif key == "features":
                features = np.random.choice(value, 1)[0]
                #print("features:", features) 
                
    return window_size, learning_rate, batch_size, gradient_steps, buffer_size, learning_starts, features

def record_values_ddpg(agent, model, runtime, tb_log_name, total_iterations, year, deterministic, 
                       batch_size, learning_rate, buffer_size, window_size, features, gradient_steps, 
                       learning_starts, layers, layer_size, synthetic, time_stamp):
    """ Records all relevant information incl. performance measures """
        
    values = {"runtime[s]":runtime,
              "tb_log_name":tb_log_name, 
              "total_iterations":total_iterations, 
              "year":year,
              "deterministic":deterministic,
              "learning_rate":learning_rate,
              "batch_size":batch_size, 
              "buffer_size":buffer_size,
              "window_size":window_size,
              "features":features,
              "gradient_steps":gradient_steps,
              "learning_starts":learning_starts,
              "seed":model.seed, 
              "device":model.device,
              "layers":layers,
              "layer_size":layer_size,
              "synthetic":synthetic,
              "time_stamp":time_stamp
    }
    
    values_from_agent = agent.performance_measures()
    values.update(values_from_agent)
    
    # Reading in the header
    df = pd.read_csv("logs/ddpg_logs_new.csv")   
    header = list(df.columns)
    
    data = open("logs/ddpg_logs_new.csv", 'a')   
    data.write("\n")
    for index, element in enumerate(header):
        if element == "features":
            string = ""
            if len(values[element]) == 0:
                string = "None"
            else:
                for j, feature in enumerate(values[element]):
                    if j < len(values[element]) - 1:
                        string += feature + "|"
                    else:
                        string += feature
            data.write(string)
        if element != "features" and element != "train_freq":
            data.write(str(values[element]))
        if index < len(header) - 1:
            data.write(",")
    data.close()
  
def backtest_ddpg(scenario, batch_size, learning_rate, buffer_size, window_size, 
                 gradient_steps, learning_starts, features, layers, layer_size, 
                 total_iterations):
    
    years = ["2018", "2021"]
    if scenario == "scenario5":
        years = ["scenario5"] 
        synthetic = False
    elif scenario == "scenario3":
        synthetic = True
    elif scenario == "scenario2":
        synthetic = False
    else:
        synthetic = None

    tb_log_name = f"{scenario}_backtest"
    
    model = None
    deterministic = True
    total_runtime = 0
    time_stamp = time.time()
    
    # Training the model on subset of real crashes
    if scenario == "scenario4":
        for year in years:
            for state in [True, False]:
                model, runtime = train_ddpg(window_size=window_size, buffer_size=buffer_size, batch_size=batch_size,
                                learning_starts=learning_starts, total_iterations=total_iterations, learning_rate=learning_rate, 
                                year=year, tb_log_name=tb_log_name, features=features, gradient_steps=gradient_steps,
                                model=model, split="train", synthetic=synthetic)
                total_runtime += runtime
    else:
         for year in years:
            model, runtime = train_ddpg(window_size=window_size, buffer_size=buffer_size, batch_size=batch_size,
                                learning_starts=learning_starts, total_iterations=total_iterations, learning_rate=learning_rate, 
                                year=year, tb_log_name=tb_log_name, features=features, gradient_steps=gradient_steps,
                                model=model, split="train", synthetic=synthetic)
            total_runtime += runtime   
    runtime = total_runtime
    
    # Testing the agent on the real 2022 crash
    year = "2022"
    agent = ddpg_agent(lookback_window_size=window_size, features=features)
    agent.generate_portfolio(year=year, synthetic=False)
    agent.set_weight_storage(agent._crash_length)
    #agent.set_batch_size(batch_size)
        
    actions = []
    deterministic = True
    done = False
    obs = agent.reset()
    while done is False:
        w, _ = model.predict(torch.from_numpy(obs).float(), deterministic=deterministic)
        obs, reward, done, info = agent.step(w)
        action = F.softmax(torch.from_numpy(w), dim=0).numpy()
        actions.append(action)
    agent._weight_storage = actions
    agent.render()

    record_values_ddpg(agent, model, runtime, tb_log_name, total_iterations, year, deterministic, 
                        batch_size, learning_rate, buffer_size, window_size, features, gradient_steps, 
                        learning_starts, layers, layer_size, synthetic, time_stamp)
    