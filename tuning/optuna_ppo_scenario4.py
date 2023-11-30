import numpy as np
from stable_baselines3 import PPO
import torch.nn as nn
import torch
import optuna
from typing import Any, Dict

def get_batches(n_steps):
    batches = []
    for i in range(10,300):
        if n_steps % i == 0:
            batches.append(i)
    return batches

def sample_ppo_params(trial: optuna.Trial) -> Dict[str, Any]:
    """
    Sampler for PPO hyperparams.
    
    :param trial: Optuna trial object
    :return: The sampled hyperparameters for the given trial
    """
    """learning_rate = trial.suggest_categorical("learning_rate", [1e-05,2e-05,3e-05,4e-05,5e-05,6e-05,7e-05,8e-05,9e-05,
                                    1e-04,2e-04,3e-04,4e-04,5e-04,6e-04,7e-04,8e-04,9e-04, 
                                    1e-03,2e-03,3e-03,4e-03,5e-03,6e-03,7e-03,8e-03,9e-03,
                                    1e-02,2e-02,3e-02,4e-02,5e-02,6e-02,7e-02,8e-02,9e-02])"""
    learning_rate = trial.suggest_float("learning_rate", 0.00000001, 0.001)
    n_epochs = trial.suggest_categorical("n_epochs", [5,10,15,20,25,30])
    gamma = trial.suggest_categorical("gamma", [0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999])
    ent_coef = trial.suggest_float("ent_coef", 0.00000001, 0.1, log=True)
    clip_range = trial.suggest_categorical("clip_range", [0.1, 0.2, 0.3, 0.4])
    gae_lambda = trial.suggest_categorical("gae_lambda", [0.8, 0.9, 0.92, 0.95, 0.98, 0.99, 1.0])
    max_grad_norm = trial.suggest_categorical("max_grad_norm", [0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 5])
    vf_coef = trial.suggest_float("vf_coef", 0, 1)
    ortho_init = False

    layer_size = trial.suggest_categorical("layer_size", [8, 16, 32, 64, 128, 256])
    n_layers = trial.suggest_categorical("n_layers", [1, 2, 3])

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

    activation_fn = trial.suggest_categorical("activation_fn", ["tanh", "relu"])
    activation_fn = {"tanh": nn.Tanh, "relu": nn.ReLU}[activation_fn]

    window_sizes = [130, 132, 134, 135, 136, 138, 139, 140, 141, 143, 145, 146, 148, 149, 150]
    window_size = trial.suggest_categorical("window_size", window_sizes)
    n_steps = int(28125 - window_size) # 28125: crash_length for scenario5 train set
    batches = get_batches(n_steps)
    batch_size = np.random.choice(batches, 1)[0]
    seed = trial.suggest_int("seed", 0, 3000)
    
    return {
        "window_size": window_size,
        "features": features,
        "batch_size": batch_size,
        "gamma": gamma,
        "learning_rate": learning_rate,
        "ent_coef": ent_coef,
        "clip_range": clip_range,
        "n_epochs": n_epochs,
        "gae_lambda": gae_lambda,
        "max_grad_norm": max_grad_norm,
        "vf_coef": vf_coef,
        "seed" : seed,
        "policy_kwargs": dict(
            layer_size=layer_size,
            n_layers=n_layers,
            activation_fn=activation_fn,
            ortho_init=ortho_init,
        )
    }


from cryptoportfolio.rlagent.rlagent_ppo import RLAgent as ppo_agent
from cryptoportfolio.rlagent.network import CustomCNN_PPO
from cryptoportfolio.rlagent.network import CustomActorCriticPolicy 
import pprint


def objective(trial: optuna.Trial) -> float:
    """
    Objective function using by Optuna to evaluate
    one configuration (i.e., one set of hyperparameters).

    Given a trial object, it will sample hyperparameters,
    evaluate it and report the result (mean episodic reward after training)

    :param trial: Optuna trial object
    :return: Mean episodic reward after training
    """

    total_iterations = 1

    hyperparams = {
        "policy": CustomActorCriticPolicy, 
    }
    
    # Sample the hyperparameters
    hyperparams.update(sample_ppo_params(trial))
    pp = pprint.PrettyPrinter(indent=0)
    pp.pprint(hyperparams)
    
    data = open("scenario4-ppo-nosoft_mlp_cnn-cheating-new1-batches.csv", "a")
    data.write("\n")
    data.write(str(hyperparams["window_size"])); data.write(",")
    data.write(str(hyperparams["batch_size"]))
    data.close()
    
    agent = ppo_agent(lookback_window_size=hyperparams["window_size"],
                      features=hyperparams["features"],
                      batch_size=hyperparams["batch_size"])
    agent.generate_portfolio(year="scenario4", synthetic=False, split="train")
    n_steps = int(agent._crash_length - agent._lookback_window_size)
    hyperparams.update({"n_steps": n_steps})
    hyperparams.update({"env": agent})
    hyperparams["policy_kwargs"].update(
        {"features_extractor_class": CustomCNN_PPO,
        "features_extractor_kwargs": dict(features_dim=13, agent_env=agent)}        
    )
    
    model = PPO(
        policy = hyperparams["policy"],
        env = hyperparams["env"],
        learning_rate = hyperparams["learning_rate"],
        n_steps = hyperparams["n_steps"],
        batch_size = hyperparams["batch_size"],
        n_epochs = hyperparams["n_epochs"],
        gamma = hyperparams["gamma"],
        gae_lambda = hyperparams["gae_lambda"],
        clip_range = hyperparams["clip_range"],
        ent_coef = hyperparams["ent_coef"],
        vf_coef = hyperparams["vf_coef"],
        max_grad_norm = hyperparams["max_grad_norm"],
        policy_kwargs = hyperparams["policy_kwargs"],
        verbose = 1,
        seed = hyperparams["seed"],
    )

    total_timesteps = total_iterations * n_steps
    model.learn(total_timesteps)
    
    # Backtest
    test_agent = ppo_agent(lookback_window_size=hyperparams["window_size"],
                      features=hyperparams["features"],
                      batch_size=hyperparams["batch_size"])
    #test_agent.generate_portfolio(year="scenario4", synthetic=False, split="validation")
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
study_name = "scenario4-ppo-nosoft_mlp_cnn-cheating-lr"
storage_name = "sqlite:///{}.db".format(study_name)

try:
    restored_sampler = pickle.load(open(f"{study_name}.pkl", "rb"))
    study = optuna.create_study(
        study_name=study_name, storage=storage_name, load_if_exists=True, sampler=restored_sampler
    )
except:
    # Select the sampler, can be random, TPESampler, CMAES, ...
    sampler = TPESampler(n_startup_trials=0, seed=0)
    study = optuna.create_study(study_name=study_name, sampler=sampler, direction="maximize", storage=storage_name)#, pruner=pruner

for _ in range(200):
    try:
        study.optimize(objective, n_trials=1)
    except:
        study.sampler.reseed_rng()
        print("Trial failed ...")
        
    # Write report
    study.trials_dataframe().to_csv(f"{study_name}.csv")

    # Save the sampler with pickle to be loaded later.
    with open(f"{study_name}.pkl", "wb") as fout:
        pickle.dump(study.sampler, fout)