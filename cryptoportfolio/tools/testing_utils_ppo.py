from stable_baselines3 import PPO
from cryptoportfolio.rlagent.rlagent_ppo import RLAgent as ppo_agent
from cryptoportfolio.rlagent.network import CustomCNN_PPO
from cryptoportfolio.rlagent.network import CustomActorCriticPolicy 
import datetime, time
import pandas as pd
import numpy as np
import torch

from stable_baselines3.common.buffers import RolloutBuffer

""""""""""""""""""""" PPO """""""""""""""""""""


window_sizes_ppo = {"scenario4":[30, 31, 32, 33, 35, 39, 40, 41, 42, 43, 45, 46, 47, 48, 50, 52, 53, 55, 57, 58, 60, 
                             61, 62, 63, 64, 65, 66, 69, 70, 71, 72, 75, 77, 79, 80, 81, 82, 83, 84, 85, 86, 88, 
                             89, 90, 91, 92, 93, 95, 97, 99, 100, 101, 105, 108, 109, 110, 111, 112, 113, 115, 116, 
                             117, 119, 120, 122, 123, 125, 126, 129, 130, 132, 134, 135, 136, 138, 139, 140, 141, 
                             143, 145, 146, 148, 149, 150]}

def get_hyperparams_ppo():
    
    hyperparameters_ppo = {
                   "learning_rate":[1e-05,2e-05,3e-05,4e-05,5e-05,6e-05,7e-05,8e-05,9e-05,
                                    1e-04,2e-04,3e-04,4e-04,5e-04,6e-04,7e-04,8e-04,9e-04, 
                                    1e-03,2e-03,3e-03,4e-03,5e-03,6e-03,7e-03,8e-03,9e-03,
                                    1e-02,2e-02,3e-02,4e-02,5e-02,6e-02,7e-02,8e-02,9e-02],
                   "n_epochs":[5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80],
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
    
    windows_batches = {"128": [14, 16, 112, 56, 28] ,
                        "130": [10] ,
                        "132": [12] ,
                        "135": [105, 35, 21, 15] ,
                        "138": [18] ,
                        "140": [10, 20] ,
                        "142": [14] ,
                        "144": [16, 24, 48, 12] ,
                        "150": [10, 30, 15] ,
                        "30": [35, 126, 70, 105, 10, 42, 45, 14, 15, 18, 210, 21, 90, 30, 63] ,
                        "32": [16] ,
                        "36": [12] ,
                        "40": [40, 10, 20] ,
                        "44": [28, 14] ,
                        "45": [15] ,
                        "48": [36, 72, 12, 16, 48, 18, 144, 24] ,
                        "50": [10] ,
                        "51": [21] ,
                        "58": [14] ,
                        "60": [10, 12, 15, 20, 60, 30] ,
                        "64": [16] ,
                        "65": [35] ,
                        "66": [18] ,
                        "70": [10] ,
                        "72": [168, 42, 12, 14, 84, 21, 24, 56, 28] ,
                        "75": [45, 15] ,
                        "80": [40, 10, 16, 80, 20] ,
                        "84": [18, 12, 36] ,
                        "86": [14] ,
                        "90": [10, 30, 15] ,
                        "93": [21, 63] ,
                        "96": [16, 24, 48, 12] ,
                        "100": [35, 70, 10, 140, 14, 20, 28] ,
                        "102": [18] ,
                        "105": [15] ,
                        "108": [12] ,
                        "110": [10] ,
                        "112": [16] ,
                        "114": [42, 21, 14] ,
                        "120": [36, 40, 72, 10, 12, 45, 120, 15, 18, 20, 180, 24, 90, 60, 30]
    }
    
    window_size = np.random.choice([int(window) for window in windows_batches.keys()], 1)[0]
    batch_size = np.random.choice(windows_batches[str(window_size)], 1)[0]
    print("window_size:", window_size)
    print("batch_size:", batch_size)
    
    for key, value in hyperparameters_ppo.items():
        if key == "learning_rate":
            learning_rate = np.random.choice(value, 1)[0]
            print("learning rate:", learning_rate)
        elif key == "n_epochs":
            n_epochs = np.random.choice(value, 1)[0]
            print("n_epochs:", n_epochs)
        elif key == "features":
            features = np.random.choice(value, 1)[0]
            features = ["close", "low", "high"] + features
            print("features:", features)
    
    return window_size, batch_size, learning_rate, n_epochs, features
    
def get_hyperparams_ppo_scenario5():
    
    hyperparameters_ppo = {
                   "learning_rate":[1e-05,2e-05,3e-05,4e-05,5e-05,6e-05,7e-05,8e-05,9e-05,
                                    1e-04,2e-04,3e-04,4e-04,5e-04,6e-04,7e-04,8e-04,9e-04, 
                                    1e-03,2e-03,3e-03,4e-03,5e-03,6e-03,7e-03,8e-03,9e-03,
                                    1e-02,2e-02,3e-02,4e-02,5e-02,6e-02,7e-02,8e-02,9e-02],
                   "n_epochs":[5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80],
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
    
    for key, value in hyperparameters_ppo.items():
        if key == "learning_rate":
            learning_rate = np.random.choice(value, 1)[0]
            #print("learning rate:", learning_rate)
        elif key == "n_epochs":
            n_epochs = np.random.choice(value, 1)[0]
            #print("n_epochs:", n_epochs)
        elif key == "features":
            features = np.random.choice(value, 1)[0]
            features = ["close", "high", "low"] + features 
            print("features:", features) 
    
    return learning_rate, n_epochs, features

def get_batch_size(n_steps):
    print("n_steps:", n_steps)
    batches = []
    for i in range(10,300):
        if n_steps % i == 0:
            batches.append(i)
    try:
        batch_size = np.random.choice(batches, 1)[0]
    except:
        raise ValueError("No batch could be found.")
    print("batch size:", batch_size)
    return batch_size
                
def train_ppo(window_size, n_epochs, total_iterations, learning_rate, batch_size, year, tb_log_name, features, 
              model=None, synthetic=False, split="whole"):

        agent = ppo_agent(lookback_window_size=window_size, features=features,
                          batch_size=batch_size)
        agent.generate_portfolio(year=year, synthetic=synthetic, split=split)
        n_steps = int(agent._crash_length - agent._lookback_window_size)
        
        #timer = int(time.time()) # Starts the timer
        #logdir = f"logs/ppo/{timer}"

        if model is None:
            
            policy_kwargs = dict(
                features_extractor_class=CustomCNN_PPO,
                features_extractor_kwargs=dict(features_dim=13, agent_env=agent),
                layer_size = 200,
                n_layers = 1,
            )
                    
            model = PPO(policy=CustomActorCriticPolicy, 
                            env=agent,
                            device="auto",
                            n_epochs=n_epochs,
                            n_steps=n_steps,
                            learning_rate=learning_rate,
                            batch_size=batch_size,
                            policy_kwargs=policy_kwargs,
                            seed=0,
                            verbose=1,
                            #tensorboard_log=logdir
                        )
        else:
            # Model exists and now is trained on different data set
            #model.batch_size = batch_size#; print("model.batch_size: ", model.batch_size)
            model.n_steps = n_steps
            model.env.envs[0].env = agent
            
            model.rollout_buffer = RolloutBuffer(
                    model.n_steps,
                    model.observation_space,
                    model.action_space,
                    device=model.device,
                    gamma=model.gamma,
                    gae_lambda=model.gae_lambda,
                    n_envs=model.n_envs,
        )

        total_timesteps = total_iterations * n_steps

        start = datetime.datetime.now() # Starts the timer
        model.learn(total_timesteps, tb_log_name=tb_log_name)#, progress_bar=True)#, reset_num_timesteps=True)
        runtime = (datetime.datetime.now() - start).total_seconds() 

        """
        ############################
        
        print("Runtime for agent step_phase1:", agent.runtime_step_phase1, "s")
        print("Runtime for agent step_perf_measures:", agent.runtime_step_perf_measures, "s")
        print("Runtime for agent step_obs_tensor:", agent.runtime_step_obs_tensor, "s")
        print("Total runtime for agent:", agent.runtime_step_phase1+agent.runtime_step_perf_measures+agent.runtime_step_obs_tensor)
        
        ############################
        """
        
        return model, runtime
    
def train_ppo_scenario5(n_epochs, total_iterations, learning_rate, year, tb_log_name, features, 
              model=None, synthetic=False, split="whole", batch_size=None, window_size=None):

        if window_size is None:
            window_size = np.random.choice(window_sizes_ppo[year], 1)[0]
        print("window size:", window_size)
        agent = ppo_agent(lookback_window_size=window_size, features=features)
        agent.generate_portfolio(year=year, synthetic=synthetic, split=split)
        n_steps = int(agent._crash_length - agent._lookback_window_size)
        total_timesteps = total_iterations * n_steps
        
        if batch_size is None:
            batch_size = get_batch_size(n_steps)
        agent.set_batch_size(batch_size)
        
        timer = int(time.time()) # Starts the timer
        logdir = f"logs/ppo/{timer}"

        if model is None:
            
            policy_kwargs = dict(
                features_extractor_class=CustomCNN_PPO,
                features_extractor_kwargs=dict(features_dim=13, agent_env=agent),
            )
                    
            model = PPO(policy=CustomActorCriticPolicy, 
                            env=agent,
                            device="auto",
                            n_epochs=n_epochs,
                            n_steps=n_steps,
                            learning_rate=learning_rate,
                            batch_size=batch_size,
                            policy_kwargs=policy_kwargs,
                            seed=0,
                            verbose=1,
                            tensorboard_log=logdir
                        )
        else:
            # Model exists and now is trained on different data set
            model.n_steps = n_steps
            model.env.envs[0].env = agent
            
            model.rollout_buffer = RolloutBuffer(
            model.n_steps,
            model.observation_space,
            model.action_space,
            device=model.device,
            gamma=model.gamma,
            gae_lambda=model.gae_lambda,
            n_envs=model.n_envs,
        )
        #print("model.batchsize:", model.batch_size)            
        start = datetime.datetime.now() # Starts the timer
        model.learn(total_timesteps, tb_log_name=tb_log_name)#, reset_num_timesteps=True)
        runtime = (datetime.datetime.now() - start).total_seconds() 

        return model, runtime, window_size
  
def record_values_ppo(agent, model, runtime, tb_log_name, total_iterations, year, deterministic, 
                       batch_size, learning_rate, n_epochs, window_size, features, layers, layer_size, 
                       synthetic, time_stamp):
    """ Records all relevant information incl. performance measures """
    
    values = {"runtime[s]":runtime,
              "total_iterations":total_iterations, 
              "tb_log_name":tb_log_name, 
              "year":year,
              "deterministic":deterministic,
              "learning_rate":learning_rate,
              "batch_size":batch_size, 
              "n_epochs":n_epochs,
              "features":features,
              "window_size":window_size,
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
    df = pd.read_csv("logs/ppo_logs_new.csv")   
    header = list(df.columns)
    
    data = open("logs/ppo_logs_new.csv", 'a')   
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
        if element != "features":
            data.write(str(values[element]))
        if index < len(header) - 1:
            data.write(",")
    data.close()

def backtest_ppo(scenario, batch_size, learning_rate, n_epochs, window_size, 
                 features, layers, layer_size, total_iterations):

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

    # Initial model is None. Is overwritten later when existing model is trained on new crash
    model = None
    deterministic = True
    total_runtime = 0
    time_stamp = time.time()

    # Training the model on subset of real crashes
    for year in years:
        if scenario == "scenario4":
            for state in [True, False]:
                model, runtime = train_ppo(window_size=window_size, batch_size=batch_size, n_epochs=n_epochs,
                                        total_iterations=total_iterations, learning_rate=learning_rate, 
                                        year=year, tb_log_name=tb_log_name, features=features,
                                        model=model, synthetic=state, split="train")
                total_runtime += runtime
        elif scenario == "scenario5":
            model, runtime, window_size = train_ppo_scenario5(window_size=window_size, batch_size=batch_size, n_epochs=n_epochs,
                                    total_iterations=total_iterations, learning_rate=learning_rate, 
                                    year=year, tb_log_name=tb_log_name, features=features,
                                    model=model, synthetic=synthetic, split="train")
            total_runtime += runtime
        else:
            model, runtime = train_ppo(window_size=window_size, batch_size=batch_size, n_epochs=n_epochs,
                                    total_iterations=total_iterations, learning_rate=learning_rate, 
                                    year=year, tb_log_name=tb_log_name, features=features,
                                    model=model, synthetic=synthetic, split="train")
            total_runtime += runtime   
    runtime = total_runtime

    # Testing the model on the 2022 real crash
    year = "2022"
    agent = ppo_agent(lookback_window_size=window_size, features=features)
    agent.generate_portfolio(year=year, synthetic=False)

    deterministic = True
    done = False
    obs = agent.reset()
    while done is False:
        w, _ = model.predict(torch.from_numpy(obs).float(), deterministic=deterministic)
        obs, reward, done, info = agent.step(w)
    agent.render()

    record_values_ppo(agent, model, runtime, tb_log_name, total_iterations, year, deterministic, 
                        batch_size, learning_rate, n_epochs, window_size, features, layers, layer_size, 
                        synthetic, time_stamp)

def hyper_tuning_ppo(scenario, total_iterations, layers = 1, layer_size = 100):
    
    tb_log_name = f"hypertuning_{scenario}"
    model = None
    time_stamp = time.time()
    

    runtimes = {"2018":0, "2021":0}
    years = ["2018", "2021"]
    if scenario == "scenario4":
        synthetic = None
        runtimes = {"True":{"2018":0, "2021":0}, "False":{"2018":0, "2021":0}}
    elif scenario == "scenario2" or scenario == "scenario5":
        synthetic = False
    elif scenario == "scenario5":
        years = ["scenario5"]
        learning_rate, n_epochs, features = get_hyperparams_ppo_scenario5()    
    elif scenario == "scenario3":
        synthetic = True
    else:
        synthetic = None
                
    # Training the model on subset of real crashes
    if scenario == "scenario4":
        for year in years:
            for state in [True, False]:
                model, runtime = train_ppo(window_size=window_size, batch_size=batch_size, n_epochs=n_epochs,
                                        total_iterations=total_iterations, learning_rate=learning_rate, 
                                        year=year, tb_log_name=tb_log_name, features=features,
                                        model=model, synthetic=state, split="train")
                runtimes[str(state)][year] = runtime
                
    else:
        # Training the model on the respective crashes
        for year in years:
            model, runtime = train_ppo(window_size=window_size, batch_size=batch_size, n_epochs=n_epochs,
                                total_iterations=total_iterations, learning_rate=learning_rate, 
                                year=year, tb_log_name=tb_log_name, features=features,
                                model=model, synthetic=synthetic, split="train")
            runtimes[year] = runtime
        
        # Testing on the remaining year's validation data
        for year in years:
            agent = ppo_agent(lookback_window_size=window_size, features=features)
            agent.generate_portfolio(year=year, synthetic=synthetic, split="validation")
            agent.new_parameters(features, window_size)

            deterministic = True
            done = False
            obs = agent.reset()
            while done is False:
                w, _ = model.predict(torch.from_numpy(obs).float(), deterministic=deterministic)
                obs, reward, done, info = agent.step(w)
            agent.render()
            
            runtime = runtimes[year]
            record_values_ppo(agent, model, runtime, tb_log_name, total_iterations, year, deterministic, 
                            batch_size, learning_rate, n_epochs, window_size, features, layers, layer_size, 
                            synthetic, time_stamp)


    
    # Training the model on subset of real crashes
    for year in years:
        model, runtime = train_ppo(window_size=window_size, n_epochs=n_epochs,
                                total_iterations=total_iterations, learning_rate=learning_rate, 
                                year=year, tb_log_name=tb_log_name, features=features,
                                model=model, synthetic=synthetic, split="train", batch_size=batch_size)  
        runtimes[year] = runtime
    
    # Testing on the remaining year's validation data
    for year in years:
        agent = RLAgent()
        agent.generate_portfolio(year=year, synthetic=synthetic, split="validation")
        agent.new_parameters(features, window_size)

        deterministic = True
        done = False
        obs = agent.reset()
        while done is False:
            w, _ = model.predict(torch.from_numpy(obs).float(), deterministic=deterministic)
            obs, reward, done, info = agent.step(w)
        agent.render()
        
        runtime = runtimes[year]
        record_values_ppo(agent, model, runtime, tb_log_name, total_iterations, year, deterministic, 
                        batch_size, learning_rate, n_epochs, window_size, features, layers, layer_size, 
                       synthetic, time_stamp)
    return