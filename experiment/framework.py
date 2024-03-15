def warn(*args, **kwargs):
    pass

import warnings as wr

wr.warn = warn

import gc
import jstyleson
import os
import pathlib as ph
import dill as pk
import sys
import re
import traceback as tb
import itertools as it
import sqlite3 as sql
from datetime import datetime as dt
import time

from joblib import Parallel, delayed
import numpy as np
import pandas as pd 
from sklearn.utils import check_random_state as crs
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split as tts
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB, ComplementNB
from sklearn.linear_model import LogisticRegression

true_print = print

def skclone(estimator):
    # Create a new instance of the estimator
    new_estimator = estimator.__class__()
    # Copy the parameters
    new_estimator.set_params(**estimator.get_params(deep=True))
    return new_estimator


## Monkey patching print
def print(*args, **kwargs):
    """Print function that flushes the output."""
    true_print(*args, **kwargs)
    sys.stdout.flush()
    sys.stderr.flush()


# Open config file
try:
    with open("config.json") as f:
        config = jstyleson.load(f)
except FileNotFoundError:
    print(
        "ERROR -",
        "Config file not found. Please, create a config.json file.",
        file=sys.stderr,
    )
    sys.exit(1)

# Load libraries
for lib in config["global"]["libs"]:
    # Load the sslearn library and ubulearn library.
    # If is set as None it means that the library is already installed in the system
    if lib not in sys.modules and config["global"]["libs"][lib] is not None:
        sys.path.append(config["global"]["libs"][lib])

# Import from sslearn
from sslearn.datasets import read_csv, read_keel
from sslearn.model_selection import StratifiedKFoldSS, artificial_ssl_dataset
from sslearn.subview import SubViewClassifier
from sslearn.wrapper import (
    CoForest,
    CoTraining,
    CoTrainingByCommittee,
    DemocraticCoLearning,
    DeTriTraining,
    Rasco,
    RelRasco,
    SelfTraining,
    Setred,
    TriTraining,
)
from sslearn import __version__ as sslearn_version

# Import from ubulearn
from ubulearn.features import RFWClassifier, RFWTreeClassifier
from ubulearn.neighbors import DisturbingNeighborsClassifier
from ubulearn.rotation import (
    RotationForestClassifier,
    RotationTransformer,
    RotationTreeClassifier,
)

# Create session
if config["global"]["session"] is None:
    session = dt.now().strftime("%Y-%m")
    session = f"session_{session}_{sslearn_version}"
else:
    session = config["global"]["session"]

i = 0

# Ignore warnings
def ignore_warn():
    wr.simplefilter("ignore")


# Configuration parameters
print("DEBUG -", "Loading configuration parameters...")
seed = config["global"]["seed"]
clf_seed = config["global"]["clf_seed"]
n_splits = config["global"]["n_splits"]
n_repeats = config["global"]["n_repeats"]
n_jobs = config["global"]["n_jobs"]
error_seed = crs(42)

# Check n_jobs
label_rates = config["global"]["label_rates"]
relaunch = config["global"]["relaunch"]

checkpoints = config["global"]["checkpoints"]
done_path = config["global"]["done_path"]

if n_jobs == -1:
    n_jobs = os.cpu_count()
elif n_jobs is None:
    n_jobs = 1

print("DEBUG -", f"n_jobs: {n_jobs}")

JOBS_USED = 0

random_state = crs(seed)
print("DEBUG -", "Done")


def get_already_done():
    """Get the experiments already done.

    Returns
    -------
    sqlite3.Connection
        Connection to the database.
    """
    return sql.connect(done_path)


print("DEBUG -", "Loading dataset names...")
data_it = set(
    map(lambda x: x.split(".")[0], os.listdir(config["global"]["data"]["path"]))
)
conn = get_already_done()
c = conn.cursor()
available_datasets = c.execute("SELECT name FROM Datasets;").fetchall()
available_datasets = set(map(lambda x: x[0], available_datasets))
c.close()
conn.close()
ignored_datasets = set(config["global"]["data"]["ignored_datasets"])
only_datasets = config["global"]["data"]["only_datasets"]
data_it = data_it - ignored_datasets
data_it = data_it.intersection(available_datasets)
if only_datasets is not None:
    only_datasets = set(only_datasets)
    data_it = data_it.intersection(only_datasets)

data_it = sorted(list(data_it))
print("DEBUG -", "Done")

# Functions
lrf = lambda x: f"{x:.2f}"


def load_datasets(data_it):
    """Get datasets from files.

    Parameters
    ----------
    data_it : collection
        List of datasets to load.

    Returns
    -------
    dict
        Dictionary with the datasets.
    """
    print("DEBUG -", "Loading datasets...")
    datasets = {}
    for file in data_it:
        dataset = pd.read_csv(
            os.path.join(config["global"]["data"]["path"], file + ".csv"), header=None
        )

        columns = []
        for i, tp in enumerate(dataset.dtypes):
            if not np.issubdtype(tp, np.number) and i != dataset.shape[1] - 1:
                columns.append(i)
                dataset[i] = dataset[i].astype("|S")

        y = dataset.iloc[:, -1]
        if np.issubdtype(y, np.number):
            y = y + 2
        X = dataset.iloc[:, :-1]
        if len(columns) > 0:
            elements = [X[X.columns.difference(columns)]]

            for col in columns:
                elements.append(pd.get_dummies(X[col]))

            concatenated_data = pd.concat(elements, axis=1)
            X = concatenated_data
        datasets[file.split(".")[0]] = (X.to_numpy(), y.to_numpy())
    print("DEBUG -", "Done")
    return datasets


def set_seed_to_model(model, iteration):
    """Set the seed to the model in all levels.

    Parameters
    ----------
    model : Estimator
        Model to set the seed.
    iteration : int
        Iteration number to set the seed.
    """
    local_random_state = crs(clf_seed + iteration)
    # Based on _set_random_state from sklearn
    to_set = {}
    for key in sorted(model.get_params(deep=True)):
        if key == "random_state" or key.endswith("__random_state"):
            to_set[key] = local_random_state.randint(np.iinfo(np.int32).max)


def set_n_jobs_to_model(model, n_jobs):
    """Set the n_jobs to the model in all levels.

    Parameters
    ----------
    model : Estimator
        Model to set the n_jobs.
    n_jobs : int
        Number of jobs to set.
    """
    to_set = {}
    for key in sorted(model.get_params(deep=True)):
        if key == "n_jobs" or key.endswith("__n_jobs"):
            to_set[key] = 5

    model.set_params(**to_set)


def check_not_done(model, mode, dataset, label_rate):
    """Check if the experiment is already done.

    Parameters
    ----------
    model : str
        Model name.
    mode : str
        Mode name (especific experiment).
    dataset : str
        Dataset name.
    label_rate : float
        Label rate.

    Returns
    -------
    bool
        True if the experiment is not done, False otherwise.
    """
    if relaunch:
        return False
    conn = get_already_done()
    cur = conn.cursor()
    cur.execute(
        f"SELECT COUNT(*) FROM Experiments WHERE model='{model}' AND mode='{mode}' AND dataset='{dataset}' AND labelrate='{lrf(label_rate)}';"
    )
    end = cur.fetchone()
    res = False
    if end[0] == 0:
        res = True
    cur.close()
    conn.close()
    if not res:
        print(
            "INFO -",
            f"Skipping: {model}, {mode}, {dataset}, {lrf(label_rate)}, already done.",
        )
    return res


def save_new_state(model, mode, dataset, label_rate, checkpoint_name):
    """Save the new state of the experiment.

    Parameters
    ----------
    model : str
        Model name.
    mode : str
        Mode name (especific experiment).
    dataset : str
        Dataset name.
    label_rate : float
        Label rate.
    checkpoint_name : str
        Checkpoint name.
    """
    print("INFO -", f"Saving new state: {model}, {mode}, {dataset}, {lrf(label_rate)}")
    local_save = relaunch
    conn = None
    tries = 0
    total_tries = 10
    error = None
    while tries < total_tries:
        try:
            conn = get_already_done()
            cur = conn.cursor()
            if local_save:
                cur.execute(
                    f"DELETE FROM Experiments WHERE model='{model}' AND mode='{mode}' AND dataset='{dataset}' AND labelrate='{lrf(label_rate)}';"
                )
            cur.execute(
                f"INSERT INTO Experiments (model, mode, dataset, labelrate, checkpoint, version) VALUES ('{model}', '{mode}', '{dataset}', '{lrf(label_rate)}', '{checkpoint_name}', '{sslearn_version}');"
            )
            conn.commit()
            break
        except sql.Error as error:
            # wait 3 second and try again
            if "UNIQUE" in str(error):  # Unique constraint failed
                # If the experiment is already done but was done, then delete and save again
                local_save = True
            error = error
            if cur:
                cur.close()
            if conn:
                conn.rollback()
                conn.close()
            time.sleep(error_seed.randint(3, 15))

            tries += 1
    if tries == total_tries:
        print("ERROR -", f"Failed to save new state: {error}.")
        print(
            "DEBUG -",
            f"Data: model='{model}', mode='{mode}', dataset='{dataset}', labelrate='{lrf(label_rate)}', checkpoint='{checkpoint_name}', version='{sslearn_version}'",
        )
        raise Exception("Failed to save new state.")

    if conn:
        cur.close()
        conn.close()


# Create directory
print("DEBUG -", "Creating directories...")
if checkpoints["enabled"]:
    checkpoint_dir = ph.Path(checkpoints["path"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

session_dir = ph.Path(session)
session_dir.mkdir(parents=True, exist_ok=True)
print("DEBUG -", "Done")

# Load datasets
datasets = load_datasets(data_it)


# Full experiment check
def check_full_experiment_done(mode, label_rate):
    """Check if the full experiment is already done.

    Parameters
    ----------
    mode : str
        Mode name (especific experiment).
    label_rate : float
        Label rate.
    """
    conn = get_already_done()
    cur = conn.cursor()
    name = mode["name"]
    cur.execute(
        f"SELECT dataset FROM Experiments WHERE mode='{name}' AND labelrate='{lrf(label_rate)}';"
    )
    end = cur.fetchall()
    res = False
    if len(set(data_it).difference(set([d[0] for d in end]))) == 0:
        res = True

    cur.close()
    conn.close()

    return res


def load_checkpoint(mode, dataset, label_rate):
    """Load the checkpoint.

    Parameters
    ----------
    mode : str
        Mode name (especific experiment).
    dataset : str
        Dataset name.
    label_rate : float
        Label rate.

    Returns
    -------
    dict
        Dictionary with the checkpoint.
    """
    conn = get_already_done()
    cur = conn.cursor()
    cur.execute(
        f"SELECT checkpoint FROM Experiments WHERE mode='{mode}' AND dataset='{dataset}' AND labelrate='{lrf(label_rate)}';"
    )
    end = cur.fetchone()
    res = None
    if end is not None:
        res = end[0]
    else:
        print(
            f"ERROR -",
            f"Checkpoint not found for {mode}, {dataset}, {lrf(label_rate)}",
        )
        return dict(), False
    cur.close()
    conn.close()

    try:
        with open(ph.Path(checkpoints["path"], res), "rb") as f:
            data = pk.load(f)
    except Exception as e:
        print("ERROR -", f"Failed to load checkpoint: {e}")
        return dict(), False
    print("INFO -", f"Loaded checkpoint: {ph.Path(checkpoints['path'], res)}")
    return data, True


# Create experiment executer
def create_experiments(model, dataset):
    """Create the experiments to run.

    Parameters
    ----------
    model : str
        Model name.
    dataset : str
        Dataset name.

    Returns
    -------
    list
        List of experiments to run.
    """
    # Load classifier
    experiments = list()

    clf = config["models"][model]
    model_class = eval(clf["model"])
    kind = clf["kind"]
    learners = {}
    names = []
    for mode in clf["modes"]:
        params = mode["params"]
        for p, v in params.items():
            if re.match(r".+_estimator$", p) is not None and type(v) is str:
                params[p] = eval(v)
        learners[mode["name"]] = eval("model_class(**params)")
        names.append(mode["name"])
    for name, label_rate in it.product(names, label_rates):
        experiments.append(
            {
                "model": model,
                "learner": learners[name],
                "dataset": dataset,
                "label_rate": label_rate,
                "kind": kind,
                "name": name,
            }
        )
    return experiments


def executer(model, learner, dataset, label_rate, kind, name):
    """Executer of the experiment.

    Parameters
    ----------
    learner : Estimator
        Model to use.
    dataset : str
        Dataset name.
    label_rate : float
        Label rate.
    kind : str
        Kind of the model (supervised or semi-supervised).
    name : str
        Name of the experiment.
    """
    print("INFO -", f"Starting: {name}, {dataset}, {lrf(label_rate)}, {kind}")
    repetition_results = []
    for r in range(n_repeats):
        local_seed = seed * r
        result = {"seed": local_seed, "split_result": list()}
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=local_seed)
        X, y = datasets[dataset]

        for train, test in skf.split(X, y):
            learner = skclone(learner)
            set_seed_to_model(learner, r)
            set_n_jobs_to_model(learner, n_jobs)
            local_result = dict()

            X_train, y_train = X[train], y[train]
            X_test, y_test = X[test], y[test]

            # Create the artificial dataset
            try:
                X_train, y_train, X_unlabel, y_true = artificial_ssl_dataset(
                    X_train, y_train, label_rate, random_state=local_seed
                )
            except Exception as _:
                print(f"Failed in dataset: {dataset}", file=sys.stderr)
                raise

            try:
                if kind == "semi-supervised":
                    learner.fit(X_train, y_train)
                else:
                    learner.fit(
                        X_train[y_train != y.dtype.type(-1), :],
                        y_train[y_train != y.dtype.type(-1)],
                    )
            except Exception as _:
                print(
                    "ERROR -",
                    f"Error: {name}, {dataset}, {lrf(label_rate)}, {kind}. Repeat: {r}, seed: {local_seed}. Skipping...",
                )
                print(
                    "ERROR -",
                    f"Error: {name}, {dataset}, {lrf(label_rate)}, {kind}. Repeat: {r}, seed: {local_seed}. Skipping...",
                    file=sys.stderr,
                )
                tb.print_exc()
                return None

            local_result["y_pred_trans"] = learner.predict(X_unlabel)
            local_result["y_pred_ind"] = learner.predict(X_test)
            local_result["acc_trans"] = learner.score(X_unlabel, y_true)
            local_result["acc_ind"] = learner.score(X_test, y_test)
            result["split_result"].append(local_result)

            # Clean memory
            del X_train, y_train, X_unlabel, y_true, X_test, y_test

        repetition_results.append(result)
        gc.collect()
    print("SUCCESS -", f"Done: {name}, {dataset}, {lrf(label_rate)}, {kind}")
    result = repetition_results
    return result


def merge_to_results(global_results, experiment_result):
    """Merge the results of the experiment to the global results.

    Parameters
    ----------
    global_results : dict
        Global results.
    experiment_result : dict
        Results of the experiment.
    """
    for model, datasets in experiment_result.items():  # Only one model
        if model not in global_results:
            global_results[model] = {}
        for dataset, label_rates in datasets.items():  # Only one dataset
            if dataset not in global_results[model]:
                global_results[model][dataset] = {}
            for label_rate, results in label_rates.items():  # Only one label rate
                global_results[model][dataset][label_rate] = results


def execute_a_set_of_experiments(experiments, label_rate, name):
    """Execute a set of experiments. According to the label rate and the name.

    Parameters
    ----------
    experiments : list
        List of experiments to execute.
    label_rate : float
        Label rate.
    name : str
        Name of the experiment.
    """
    ignore_warn()
    global JOBS_USED
    print(
        "INFO -",
        f"Executing experiments with label rate: {lrf(label_rate)} and name: {name}",
    )
    global_results = {}
    local_experiments_to_do = []
    for experiment in experiments:
        if experiment["label_rate"] != label_rate or experiment["name"] != name:
            continue
        if not check_not_done(
            experiment["model"], name, experiment["dataset"], label_rate
        ):
            # checkpoint_dict = load_checkpoint(model=experiment["model"], dataset=experiment["dataset"], label_rate=label_rate, mode=name)
            # merge_to_results(global_results, checkpoint_dict)
            continue
        local_experiments_to_do.append(experiment)
    try:
        posible_jobs = max((n_jobs - JOBS_USED) // len(local_experiments_to_do), 0)
    except ZeroDivisionError:
        posible_jobs = 1
    if posible_jobs == 0:
        posible_jobs = 1

    if len(local_experiments_to_do) > 0:
        Parallel(n_jobs=min(posible_jobs, len(local_experiments_to_do)), verbose=10)(
            delayed(execute_jobs)(**experiment)
            for experiment in local_experiments_to_do
        )

    # Load all checkpoints and merge to global results
    global_results, load_correct = load_all_checkpoints(
        name, label_rate
    )
    if len(global_results) == 0:
        print(
            "INFO -",
            f"Nothing to do with {lrf(label_rate)}, {name}, the experiments are already done.",
        )
        return
    if not load_correct:
        print(
            "ERROR -",
            f"Error: {name}, {lrf(label_rate)}. Some checkpoints are not correct. The global results will not be saved.",
        )
        JOBS_USED -= 1
        return

    save_for_all_datasets(global_results, label_rate, name)
    JOBS_USED -= 1
    print("SUCCESS -", f"Done: All experiments with {lrf(label_rate)}, {name}")


def load_all_checkpoints(name, label_rate):
    """Load all checkpoints and merge to global results.

    Parameters
    ----------
    name : str
        Name of the experiment.
    label_rate : float
        Label rate.

    Returns
    -------
    dict
        Global results.
    """
    print(
        "INFO -",
        f"Loading checkpoints of {name} with label rate: {lrf(label_rate)}",
    )
    results = []
    global_results = {}
    for d in data_it:
        checkpoint_dict, result = load_checkpoint(
            dataset=d, label_rate=label_rate, mode=name
        )
        merge_to_results(global_results, checkpoint_dict)
        results.append(result)
    print(
        "INFO -",
        f"Done loading checkpoints of {name} with label rate: {lrf(label_rate)}",
    )
    return global_results, all(results)


def execute_jobs(**experiment):
    ignore_warn()
    results = executer(**experiment)
    label_rate = experiment["label_rate"]
    name = experiment["name"]
    if results is not None:
        result = {name: {experiment["dataset"]: {lrf(label_rate): results}}}
        save_checkpoint(
            result,
            model=experiment["model"],
            dataset=experiment["dataset"],
            label_rate=label_rate,
            name=name,
        )


def save_for_all_datasets(results, label_rate, name):
    # Check if file not exists, if exists create a new with a next number
    print("INFO -", f"Saving results: {lrf(label_rate)}, {name}")
    path = session_dir / f"{name}_{lrf(label_rate)}.pkl"
    if path.exists():
        print("INFO -", f"File already exists: {path}")
        del results
        gc.collect()
        return
        i = 1
        while True:
            path = session_dir / f"{name}_{lrf(label_rate)}_{i}.pkl"
            if not path.exists():
                break
            i += 1
    with open(path, "wb") as f:
        pk.dump(results, f)
    if not checkpoints["enabled"]:
        for dataset in datasets:
            model = "Not defined"
            for m in config["models"]:
                if name in config["models"][m]["modes"]:
                    model = m
                    break
            save_new_state(model, name, dataset, label_rate, path.name)
    print("SUCCESS -", f"Done: Saving results: {lrf(label_rate)}, {name}")
    # Clean memory
    del results
    gc.collect()
    # Remove all checkpoints
    if checkpoints["enabled"]:
        remove_checkpoints(name, label_rate)


def remove_checkpoints(name, label_rate):
    # Version 2 - No remove checkpoints after the final file, keep this for future use.
    return
    # Version 1 - Remove all checkpoints after the final file.
    final_file = session_dir / f"{name}_{lrf(label_rate)}.pkl"

    if os.path.exists(final_file):
        print("INFO -", f"Removing checkpoints: {name}, {lrf(label_rate)}")
        conn = get_already_done()
        c = conn.cursor()
        c.execute(
            f'SELECT checkpoint FROM Experiments WHERE mode="{name}" AND labelrate="{lrf(label_rate)}"'
        )
        checkpoints = c.fetchall()
        for checkpoint in checkpoints:
            # Delete file
            path = os.path.join(checkpoint_dir, checkpoint[0])
            if os.path.exists(path):
                os.remove(path)
        conn.commit()
        conn.close()
    else:
        print(
            "ERROR -",
            f"Final file not exists: {name}, {lrf(label_rate)} - {final_file}. Therefore, checkpoints not removed",
        )


def save_checkpoint(results, model, dataset, label_rate, name):
    if checkpoints["enabled"]:
        print(
            "INFO -",
            f"Saving checkpoint: {model}, {dataset}, {lrf(label_rate)}, {name}",
        )
        checkpoint_name = f"{name}_{dataset}_{lrf(label_rate)}.pkl"
        with open(os.path.join(checkpoint_dir, f"{checkpoint_name}"), "wb") as f:
            pk.dump(results, f)
        save_new_state(model, name, dataset, label_rate, checkpoint_name)


def load_experiments(to_ignore=[]):
    experiments = []
    for model in config["models"]:
        for dataset in datasets:
            experiments.extend(create_experiments(model, dataset))

    to_delete = []
    for i, exp in enumerate(experiments):
        proof = (exp["model"], exp["name"], exp["label_rate"])
        if proof in to_ignore:
            to_delete.append(i)
    for i in reversed(to_delete):
        del experiments[i]

    return experiments


def clean_done_experiments():
    to_delete = set()
    for model in config["models"]:
        for i, mode in enumerate(config["models"][model]["modes"]):
            for lr in label_rates:
                if check_full_experiment_done(mode, lr):
                    load_checkpoints_for_done_experiments(mode["name"], lr)
                    remove_checkpoints(mode["name"], lr)
                    to_delete.add((model, mode["name"], lr))
    return to_delete


def load_checkpoints_for_done_experiments(mode, label_rate):
    # Ensure final file exists
    final_file = session_dir / f"{mode}_{lrf(label_rate)}.pkl"
    # If not exists, load the checkpoints and merge them
    if not os.path.exists(final_file):
        print("INFO -", f"Final file not exists {final_file}: {mode}, {lrf(label_rate)}")
        global_results, results = load_all_checkpoints(mode, label_rate)
        if results:
            save_for_all_datasets(global_results, label_rate, mode)
    else:
        print("INFO -", f"Final file exists: {mode}, {lrf(label_rate)} - {final_file}")


def execute_all_experiments(exps):
    global JOBS_USED
    names = set(map(lambda x: x["name"], exps))
    # Parallelize the execution of the experiments
    all_experiments_pairs = list(it.product(label_rates, names))
    local_jobs = min(len(all_experiments_pairs), n_jobs)
    Parallel(n_jobs=local_jobs, backend="loky")(
        delayed(execute_a_set_of_experiments)(exps, label_rate, name)
        for label_rate, name in all_experiments_pairs
    )


if __name__ == "__main__":
    with wr.catch_warnings():
        wr.simplefilter("ignore")
        to_ignore = clean_done_experiments()
        experiments = load_experiments(to_ignore)
        execute_all_experiments(experiments)
