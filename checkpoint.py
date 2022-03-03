import numpy as np
import time

class Checkpoint:
    def __init__(self, filename, save_interval=60*60):
        self.filename = filename
        self.state = dict()
        self.objective = np.nan
        self.objective_history = []
        self.save_interval = save_interval
        self.last_saved = np.nan

    def restore(self):
        data = np.load(self.filename, allow_pickle=True)
        self.state = tuple(data['state'][()])
        self.objective_history = list(data['objective_history'])
        self.last_saved = time.time()

    def save_state(self, *state):
        self.state = np.array(state, dtype=object)
        self.state[5] = np.copy(self.state[5])
        self.objective_history.append(self.objective)
        self.persist()

    def persist(self, force=False):
        current_time = time.time()
        if not force and current_time - self.last_saved < self.save_interval:
            return
        np.savez(self.filename, state=self.state, objective_history=self.objective_history)
        self.last_saved = current_time
