# launcher.py
from syne_tune.tuner import Tuner
from syne_tune.backend.local_backend import LocalBackend
from syne_tune.config_space import uniform, loguniform, choice
from syne_tune.optimizer.baselines import ASHA
from syne_tune import StoppingCriterion
import matplotlib.pyplot as plt
from syne_tune.experiments import load_experiment


# Define search space for mu and domain_width
config_space = {
    "bell_mu": uniform(0.0, 1.0),
    "bell_sigma": loguniform(1e-2, 0.8),
    "bell_peak": loguniform(1e-2, 1.0),
    "dataname": "MQ2008",
    "mode": "train",
    "closs_weight_schedule": "bell_like",
    "dim_t": choice([256]),
    "lr": loguniform(1e-6, 1e-3),
    "steps": 15000,
}


# Choose a scheduler/searcher
scheduler = ASHA(config_space=config_space, metric="loss/val_ndcg", time_attr="epoch", do_minimize=False, max_t=72000)

# Create Tuner
tuner = Tuner(
    trial_backend=LocalBackend(entry_point="main.py"),
    scheduler=scheduler,
    stop_criterion=StoppingCriterion(max_wallclock_time=30000),  # 20 hours
    n_workers=20,                  # number of parallel workers (adjust to your CPU/GPU)
)

tuner.run()   # returns when stopping criterion is met

results_df = tuner.tuning_status.get_dataframe()
# save dataframe
results_df.to_csv(f'hpo/output/{tuner.name}_results.csv', index=False)

# Plot results
e = load_experiment(tuner.name)  # name of the tuning run which is printed at the beginning of the run
e.plot_trials_over_time(metric_to_plot='loss/val_ndcg')
plt.savefig(f'hpo/output/{tuner.name}_trials_over_time.png')