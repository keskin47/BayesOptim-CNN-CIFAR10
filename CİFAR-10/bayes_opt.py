import os
import json
import numpy as np
from datetime import datetime
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern

import torch


class BayesOptimizer:
    def __init__(self, config, trainer_cls, model_cls):
        self.config = config
        self.trainer_cls = trainer_cls
        self.model_cls = model_cls
        self.search_space = config.search_space

        # -----------------------------------------------------
        # Log/State Paths
        # -----------------------------------------------------
        os.makedirs("./results", exist_ok=True)

        self.state_path = "./results/bo_state.json"
        self.results_path = "./results/bo_results.json"

        # -----------------------------------------------------
        # Internal BO state
        # -----------------------------------------------------
        self.results = []
        self.gp = None
        self.current_iter = 0
        self.warmup_done = False

        # -----------------------------------------------------
        # Try loading previous state
        # -----------------------------------------------------
        if os.path.exists(self.state_path):
            self._load_state()
            print("[BayesOpt] Previous state loaded. Resuming...")
        else:
            self._init_gp()
            print("[BayesOpt] Fresh start (no saved state).")

        print("[BayesOpt] Search space:", self.search_space)

    # =============================================================
    #                GP + STATE HANDLING
    # =============================================================

    def _init_gp(self):
        kernel = Matern(nu=2.5)
        self.gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-4,
            normalize_y=True,
            n_restarts_optimizer=5
        )

    def _save_state(self):
        state = {
            "results": self.results,
            "current_iter": self.current_iter,
            "warmup_done": self.warmup_done
        }
        with open(self.state_path, "w") as f:
            json.dump(state, f, indent=4)

    def _load_state(self):
        with open(self.state_path, "r") as f:
            data = json.load(f)

        self.results = data["results"]
        self.current_iter = data["current_iter"]
        self.warmup_done = data["warmup_done"]
        self._init_gp()

    # =============================================================
    #                PARAM ENCODE / DECODE
    # =============================================================

    def _encode_params(self, params):
        vector = []

        for key, space in self.search_space.items():
            value = params[key]

            if isinstance(space, list):
                vector.append(float(space.index(value)))

            elif isinstance(space, tuple):
                vector.append(float(value))

        return np.array(vector)

    def _decode_params(self, vector):
        decoded = {}
        i = 0

        for key, space in self.search_space.items():
            if isinstance(space, list):   # categorical
                idx = int(round(vector[i]))
                idx = max(0, min(idx, len(space) - 1))
                decoded[key] = space[idx]

            elif isinstance(space, tuple):
                low, high = space
                val = vector[i]

                if isinstance(low, int) and isinstance(high, int):
                    decoded[key] = int(np.clip(round(val), low, high))
                else:
                    decoded[key] = float(np.clip(val, low, high))

            i += 1

        return decoded

    # =============================================================
    #               RANDOM SAMPLING
    # =============================================================

    def _sample_random_params(self):
        params = {}

        for key, space in self.search_space.items():
            if isinstance(space, list):
                params[key] = np.random.choice(space)

            elif isinstance(space, tuple):
                low, high = space
                if isinstance(low, int):
                    params[key] = np.random.randint(low, high + 1)
                else:
                    params[key] = float(np.random.uniform(low, high))

        return params

    def _sample_random_vector(self):
        params = self._sample_random_params()
        vec = self._encode_params(params)
        return params, vec

    # =============================================================
    #               GP UPDATE + EI
    # =============================================================

    def _update_gp(self):
        if len(self.results) < 2:
            return

        try:
            X = np.array([r["vector"] for r in self.results])
            y = np.array([r["loss"] for r in self.results])
            self.gp.fit(X, y)
        except Exception as e:
            print("[BayesOpt] GP update failed:", e)

    def _ei(self, X):
        mu, sigma = self.gp.predict(X, return_std=True)
        sigma = np.maximum(sigma, 1e-9)

        y_best = min(r["loss"] for r in self.results)
        Z = (y_best - mu) / sigma
        ei = (y_best - mu) * norm.cdf(Z) + sigma * norm.pdf(Z)

        return ei

    def _suggest_next(self):
        self._update_gp()

        candidates = []
        for _ in range(300):
            _, v = self._sample_random_vector()
            candidates.append(v)
        candidates = np.array(candidates)

        ei_scores = self._ei(candidates)
        idx = np.argmax(ei_scores)
        best_vec = candidates[idx]
        best_params = self._decode_params(best_vec)

        return best_params, best_vec

    # =============================================================
    #               MODEL EVALUATION
    # =============================================================

    def _evaluate(self, params, vector):
        print(f"[BO] Evaluating params: {params}")

        model = self.model_cls(**params)
        trainer = self.trainer_cls(model, self.config)

        val_acc = trainer.train()
        loss = 1 - val_acc

        # Full detail logging entry
        result = {
            "iter": self.current_iter,
            "params": params,
            "vector": vector.tolist(),
            "loss": float(loss),
            "val_acc": float(val_acc),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "log_file": trainer.log_path,
            "checkpoint": trainer.best_ckpt_path
        }

        self.results.append(result)
        self._save_results()

        print(f"[BO] Result → val_acc: {val_acc:.4f}, loss: {loss:.4f}")
        return loss

    def _save_results(self):
        with open(self.results_path, "w") as f:
            json.dump(self.results, f, indent=4)

    # =============================================================
    #               MAIN BO LOOP
    # =============================================================

    def run(self, n_iterations=20, warmup=5):
        try:
            # Warm Up Phase
            if not self.warmup_done:
                print(f"[BO] Warm-up phase: {warmup}")
                for i in range(warmup):
                    self.current_iter += 1
                    p, v = self._sample_random_vector()
                    loss = self._evaluate(p, v)
                    self._save_state()

                self.warmup_done = True
                self._save_state()

            # Main BO iterations
            print(f"[BO] Starting main phase: {n_iterations - warmup} iterations")

            for _ in range(n_iterations - warmup):
                self.current_iter += 1

                p, v = self._suggest_next()
                self._evaluate(p, v)
                self._save_state()

            # Best result
            best = min(self.results, key=lambda r: r["loss"])
            print("\n[BO] Finished.")
            print("[BO] Best params:", best["params"])
            print("[BO] Best validation accuracy:", best["val_acc"])

            return best

        except KeyboardInterrupt:
            print("\n[BO] INTERRUPTED — Saving state safely...")
            self._save_state()
            print("[BO] State saved. Goodbye!")
            exit()
