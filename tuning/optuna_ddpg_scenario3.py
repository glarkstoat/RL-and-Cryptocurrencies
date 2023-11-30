import numpy as np
from stable_baselines3 import DDPG
import torch.nn as nn
import torch
import optuna
from typing import Any, Dict


def sample_ddpg_params(trial: optuna.Trial) -> Dict[str, Any]:
    """
    Sampler for DDPG hyperparams.
    
    :param trial: Optuna trial object
    :return: The sampled hyperparameters for the given trial
    """
    
    gamma = trial.suggest_categorical("gamma", [0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999])
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 0.1, log=False)
    buffer_size = trial.suggest_categorical("buffer_size", [1000,2000,3000,4000,5000,10000,15000])
    tau = trial.suggest_categorical("tau", [0.001,0.005,0.01,0.02,0.05,0.08])
    learning_starts = trial.suggest_categorical("learning_starts", [10,20,30,40,50,60,70,80,90,100,110,120,130,140,150])
    gradient_steps = trial.suggest_categorical("gradient_steps", [100,200,500,600,700,800,900,1000,1500,2000,2500,3000])
    net_arch = trial.suggest_categorical("net_arch", ["tiny", "small", "small+", "small++", "small+++", "medium", "large"])
    net_arch = {"tiny":[10,10], "small":[100,100], "small+++":[126,256], "small++":[256,126], "small+":[400,300], "medium":[300,300], "large":[500,500]}[net_arch]
    
    activation_fn = trial.suggest_categorical("activation_fn", ["tanh", "relu", "elu", "leaky_relu"])
    activation_fn = {"tanh": nn.Tanh, "relu": nn.ReLU, "elu": nn.ELU, "leaky_relu": nn.LeakyReLU}[activation_fn]

    features = trial.suggest_categorical("features", ["None", "RSI", "mcd", "SMA", "RSI_mcd",
                "RSI_SMA", "OSZ", "RSI_OSZ", "Bollinger", "Bollinger_mcd", "RSI_SMA_Bollinger", 
                "mcd_OSZ", "Everything"])
    features_list = {"None": [],
        "RSI": ["RSI"],
        "mcd": ["mcd", "mcd_signal"],
        "SMA": ["SMA_50", "SMA_200"],
        "RSI_mcd": ["RSI", "mcd", "mcd_signal"],
        "RSI_SMA": ["RSI", "SMA_50", "SMA_200"],
        "OSZ": ["stoch_oszillator", "stoch_oszillator_signal"],
        "RSI_OSZ": ["RSI", "stoch_oszillator", "stoch_oszillator_signal"],
        "Bollinger": ["Bollinger_middle" ,"Bollinger_low" ,"Bollinger_high"],
        "Bollinger_mcd": ["mcd", "mcd_signal", "Bollinger_middle" ,"Bollinger_low" ,"Bollinger_high"],
        "RSI_SMA_Bollinger": ["RSI", "SMA_50", "SMA_200", "Bollinger_middle" ,"Bollinger_low" ,"Bollinger_high"],
        "mcd_OSZ": ["mcd", "mcd_signal", "stoch_oszillator", "stoch_oszillator_signal"],
        "Everything": ["RSI", "SMA_50", "SMA_200", "Bollinger_middle" ,"Bollinger_low" ,"Bollinger_high", "mcd", "mcd_signal", "stoch_oszillator", "stoch_oszillator_signal"]
    }
    features = ["close", "low", "high"] + features_list[features]
    window_size = trial.suggest_int("window_size", 130, 180)
    batch_size = trial.suggest_int("batch_size", 10, 150)
    seed = trial.suggest_int("seed", 0, 3000)
    
    hyperparams = {
        "features" : features,
        "window_size" : window_size,
        "gamma": gamma,
        "tau": tau,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "learning_starts" : learning_starts,
        "buffer_size": buffer_size,
        "gradient_steps": gradient_steps,
        "seed" : seed,
        "policy_kwargs": dict(
            net_arch=net_arch,
            activation_fn=activation_fn,
        )
    }
    return hyperparams


from stable_baselines3 import DDPG
from cryptoportfolio.rlagent.rlagent_ddpg import RLAgent as ddpg_agent
from cryptoportfolio.rlagent.network import CustomCNN_DDPG
import pprint
import torch.nn.functional as F

def objective(trial: optuna.Trial) -> float:
    """
    Objective function using by Optuna to evaluate
    one configuration (i.e., one set of hyperparameters).

    Given a trial object, it will sample hyperparameters,
    evaluate it and report the result (mean episodic reward after training)

    :param trial: Optuna trial object
    :return: Mean episodic reward after training
    """

    # Contains the returns of the individual evaluations of the 
    # validation sets
    returns = []
    total_iterations = 1

    hyperparams = {
    }
    
    # Sample the hyperparameters
    hyperparams.update(sample_ddpg_params(trial))
    pp = pprint.PrettyPrinter()
    pp.pprint(hyperparams)
    
    """ First iteration """
    agent = ddpg_agent(lookback_window_size=hyperparams["window_size"],
                      features=hyperparams["features"])
    agent.generate_portfolio(year="2018", synthetic=False, split="train")
    agent.set_weight_storage(hyperparams["buffer_size"])
    agent.set_batch_size(hyperparams["batch_size"])
    
    hyperparams.update({"env": agent})
    hyperparams["policy_kwargs"].update(
        {"features_extractor_class": CustomCNN_DDPG,
        "features_extractor_kwargs": dict(features_dim=13, agent_env=agent)}        
    )
    
    model = DDPG(
        policy = "CnnPolicy", 
        env = hyperparams["env"],
        learning_rate = hyperparams["learning_rate"],
        buffer_size = hyperparams["buffer_size"],
        batch_size = hyperparams["batch_size"],
        learning_starts = hyperparams["learning_starts"],
        gamma = hyperparams["gamma"],
        tau = hyperparams["tau"],
        gradient_steps = hyperparams["gradient_steps"],
        policy_kwargs = hyperparams["policy_kwargs"],
        seed = hyperparams["seed"],
    )

    total_timesteps = total_iterations * agent._crash_length
    model.learn(total_timesteps)
    
    """ Second iteration """
    agent = ddpg_agent(lookback_window_size=hyperparams["window_size"],
                      features=hyperparams["features"])
    agent.generate_portfolio(year="2021", synthetic=False, split="train")
    agent.set_weight_storage(hyperparams["buffer_size"])
    agent.set_batch_size(hyperparams["batch_size"])
    
    model.replay_buffer = model.replay_buffer_class(
                model.buffer_size,
                model.observation_space,
                model.action_space,
                device=model.device,
                n_envs=model.n_envs,
                optimize_memory_usage=model.optimize_memory_usage,
                **model.replay_buffer_kwargs,
    )
    # Overwriting the model parameters with the new ones
    model.env.envs[0].env = agent

    total_timesteps = total_iterations * agent._crash_length
    model.learn(total_timesteps)
    
    """ Third iteration """
    agent = ddpg_agent(lookback_window_size=hyperparams["window_size"],
                      features=hyperparams["features"])
    agent.generate_portfolio(year="2018", synthetic=True, split="train")
    agent.set_weight_storage(hyperparams["buffer_size"])
    agent.set_batch_size(hyperparams["batch_size"])
    
    model.replay_buffer = model.replay_buffer_class(
                model.buffer_size,
                model.observation_space,
                model.action_space,
                device=model.device,
                n_envs=model.n_envs,
                optimize_memory_usage=model.optimize_memory_usage,
                **model.replay_buffer_kwargs,
    )
    # Overwriting the model parameters with the new ones
    model.env.envs[0].env = agent

    total_timesteps = total_iterations * agent._crash_length
    model.learn(total_timesteps)
    
    """ Fourth iteration """
    agent = ddpg_agent(lookback_window_size=hyperparams["window_size"],
                      features=hyperparams["features"])
    agent.generate_portfolio(year="2021", synthetic=True, split="train")
    agent.set_weight_storage(hyperparams["buffer_size"])
    agent.set_batch_size(hyperparams["batch_size"])
    
    model.replay_buffer = model.replay_buffer_class(
                model.buffer_size,
                model.observation_space,
                model.action_space,
                device=model.device,
                n_envs=model.n_envs,
                optimize_memory_usage=model.optimize_memory_usage,
                **model.replay_buffer_kwargs,
    )
    # Overwriting the model parameters with the new ones
    model.env.envs[0].env = agent

    total_timesteps = total_iterations * agent._crash_length
    model.learn(total_timesteps)
    
    # Backtest
    test_agent = ddpg_agent(lookback_window_size=hyperparams["window_size"],
                      features=hyperparams["features"])
    test_agent.generate_portfolio(year="2022", synthetic=False, split="whole")

    done = False
    obs = test_agent.reset()
    while done is False:
        w, _ = model.predict(torch.from_numpy(obs).float(), deterministic=True)
        obs, reward, done, info = test_agent.step(w)
    
    print("fPV:", test_agent.performance_measures()["fPV"])
    return test_agent.performance_measures()["return"]


from optuna.samplers import TPESampler
from optuna.visualization import plot_optimization_history, plot_param_importances
import torch as th
import pickle

# Set pytorch num threads to 1 for faster training
th.set_num_threads(1)

# Unique identifier of the study
study_name = "scenario3-ddpg-cnn_nosoft-cheating"  
storage_name = "sqlite:///{}.db".format(study_name)

try:
    restored_sampler = pickle.load(open(f"{study_name}.pkl", "rb"))
    study = optuna.create_study(study_name=study_name, storage=storage_name, load_if_exists=True, sampler=restored_sampler, direction="maximize")
except:
    # Select the sampler, can be random, TPESampler, CMAES, ...
    sampler = TPESampler(n_startup_trials=0, seed=0)
    study = optuna.create_study(study_name=study_name, sampler=sampler, direction="maximize", storage=storage_name)#, pruner=pruner
    
for _ in range(100):
    try:
        study.optimize(objective, n_trials=1)
    except:
        # Reseed so that the sampler doesn't pick the faulty hyperparams again
        study.sampler.reseed_rng()
        print("Trial failed ...")
        
    # Write report
    study.trials_dataframe().to_csv(f"{study_name}.csv")

    # Save the sampler with pickle to be loaded later.
    with open(f"{study_name}.pkl", "wb") as fout:
        pickle.dump(study.sampler, fout)    


print("Number of finished trials: ", len(study.trials))

print("Best trial:")
trial = study.best_trial

print(f"  Value: {trial.value}")

print("  Params: ")
for key, value in trial.params.items():
    print(f"    {key}: {value}")

try:
    fig1 = plot_optimization_history(study)
    fig2 = plot_param_importances(study)
    fig1.show()
    fig2.show()
except:
    pass
