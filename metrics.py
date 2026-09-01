"""Tiny SQLite run/metrics DB for Wyrmling training. Zero deps (stdlib sqlite3).

Records each training run's config + per-eval loss curve, so progress is queryable
instead of scrolled-past in logs:

  sqlite3 out/runs.db "SELECT step, val_loss, val_bpb FROM steps ORDER BY step"
"""
import json
import sqlite3
import time


class RunDB:
    def __init__(self, path):
        self.db = sqlite3.connect(path)
        self.db.execute("""CREATE TABLE IF NOT EXISTS runs(
            id INTEGER PRIMARY KEY AUTOINCREMENT, config TEXT, params_m REAL,
            started REAL, hparams TEXT)""")
        self.db.execute("""CREATE TABLE IF NOT EXISTS steps(
            run_id INTEGER, step INTEGER, train_loss REAL, val_loss REAL,
            train_bpb REAL, val_bpb REAL, lr REAL, elapsed_s REAL)""")
        self.db.commit()
        self.run_id = None

    def start(self, config, params_m, hparams: dict):
        cur = self.db.execute(
            "INSERT INTO runs(config,params_m,started,hparams) VALUES(?,?,?,?)",
            (config, params_m, time.time(), json.dumps(hparams)))
        self.db.commit()
        self.run_id = cur.lastrowid
        return self.run_id

    def log(self, step, train_loss, val_loss, lr, elapsed_s):
        ln2 = 0.6931471805599453
        self.db.execute(
            "INSERT INTO steps VALUES(?,?,?,?,?,?,?,?)",
            (self.run_id, step, train_loss, val_loss,
             (train_loss / ln2 if train_loss == train_loss else None),
             (val_loss / ln2 if val_loss == val_loss else None), lr, elapsed_s))
        self.db.commit()
