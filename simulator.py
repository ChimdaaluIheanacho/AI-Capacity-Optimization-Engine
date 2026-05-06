import pandas as pd
import random


class Parcel:
    def __init__(self, parcel_id, priority):
        self.parcel_id = parcel_id
        self.priority = priority
        self.rejections = 0
        self.status = "waiting"


class IndustrialProcessSimulator:
    def __init__(
        self,
        incoming_rate=25,
        conveyor_capacity=20,
        rejection_probability=0.03,
        simulation_time=120,
        seed=None,  # ✅ NEW: seed for repeatable randomness
    ):
        self.incoming_rate = incoming_rate
        self.conveyor_capacity = conveyor_capacity
        self.rejection_probability = rejection_probability
        self.simulation_time = simulation_time

        # ✅ NEW: each simulator instance gets its own RNG
        self.rng = random.Random(seed)

        self.time = 0
        self.parcel_counter = 0
        self.queue = []
        self.problem_solve = []
        self.completed = []
        self.log = []

    def generate_parcels(self):
        for _ in range(self.incoming_rate):
            priority = self.rng.choice(["normal", "urgent"])  # ✅ use self.rng
            parcel = Parcel(self.parcel_counter, priority)
            self.queue.append(parcel)
            self.parcel_counter += 1

    def process_conveyor(self):
        processed_this_tick = 0

        while self.queue and processed_this_tick < self.conveyor_capacity:
            parcel = self.queue.pop(0)

            # ✅ use self.rng so Monte Carlo seeding works
            if self.rng.random() < self.rejection_probability:
                parcel.rejections += 1

                if parcel.rejections >= 2:
                    parcel.status = "problem_solve"
                    self.problem_solve.append(parcel)
                else:
                    self.queue.append(parcel)
            else:
                parcel.status = "completed"
                self.completed.append(parcel)

            processed_this_tick += 1

    def log_kpis(self):
        self.log.append({
            "time": self.time,
            "queue_length": len(self.queue),
            "completed_total": len(self.completed),
            "problem_solve_total": len(self.problem_solve)
        })

    def run(self):
        for minute in range(self.simulation_time):
            self.time = minute
            self.generate_parcels()
            self.process_conveyor()
            self.log_kpis()

        return pd.DataFrame(self.log)
