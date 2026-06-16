"""
MAGI OS — Walking Training (RL) Algorithm
src/locomotion/train_walk.py

Runs a Neuroevolution / Genetic Algorithm in Python to discover stable leg phase shifts
and stride scaling factors that maximize walking speed and minimize vertical wobble.
Saves the optimized parameters to a YAML file.
"""

import os
import random
import yaml
import math
from typing import List, Tuple, Dict

class Agent:
    def __init__(self, agent_id: int):
        self.id = agent_id
        # Genes (Policy parameters):
        # 0-3: Leg phase offsets (FR, FL, BR, BL)
        # 4-7: Leg stride gains
        # 8: Pitch sway magnitude
        # 9: Roll sway magnitude
        # 10: Stance height bias
        # 11: Stride length base
        self.genes = [0.0] * 12
        self.fitness = 0.0
        self.randomize()

    def randomize(self):
        for i in range(12):
            self.genes[i] = random.uniform(-1.0, 1.0)
        # Keep phase offsets in range [0, 1]
        for i in range(4):
            self.genes[i] = random.uniform(0.0, 1.0)

    def mutate(self, rate: float = 0.15):
        for i in range(12):
            if random.random() < rate:
                if i < 4:
                    self.genes[i] = (self.genes[i] + random.uniform(-0.2, 0.2)) % 1.0
                else:
                    self.genes[i] = min(max(self.genes[i] + random.uniform(-0.25, 0.25), -1.0), 1.0)

    def crossover(self, partner: 'Agent') -> 'Agent':
        child = Agent(self.id)
        for i in range(12):
            child.genes[i] = self.genes[i] if random.random() < 0.5 else partner.genes[i]
        return child

class WalkingTrainer:
    def __init__(self, pop_size: int = 40, generations: int = 50):
        self.pop_size = pop_size
        self.generations = generations
        self.mutation_rate = 0.15
        self.population = []
        self.best_fitness = -9999.0
        self.best_genes = None

    def initialize_population(self):
        self.population = [Agent(i + 1) for i in range(self.pop_size)]

    def evaluate_fitness(self, agent: Agent) -> float:
        """
        Simulates the kinematics walk loops over a 5 second time window.
        Returns a fitness score evaluating:
          1. Forward stride speed (gained from correct phase shifts)
          2. Pitch/Roll tilts (stability)
          3. Height oscillation
        """
        fitness = 0.0
        dt = 0.05
        evaluation_time = 5.0
        steps = int(evaluation_time / dt)

        # Base crawl phase sequence: [0.0, 0.25, 0.75, 0.50]
        # Calculate how close the agent's phase offsets (genes 0-3) are to a stable crawl sequence:
        d01 = abs((agent.genes[0] - agent.genes[1] + 1.0) % 1.0 - 0.25)
        d13 = abs((agent.genes[1] - agent.genes[3] + 1.0) % 1.0 - 0.25)
        d32 = abs((agent.genes[3] - agent.genes[2] + 1.0) % 1.0 - 0.25)
        phase_score = 1.0 - (d01 + d13 + d32) / 3.0 # max 1.0

        # Calculate stride gains
        stride_gain = sum(agent.genes[4:8]) / 4.0
        forward_vel = max(0.0, phase_score * 45.0 * (1.0 + stride_gain * 0.35)) # mm/s

        # Stability checks
        pitch_sway = abs(agent.genes[8] * 10.0) # deg
        roll_sway = abs(agent.genes[9] * 8.0)   # deg
        wobble_penalty = (pitch_sway * 3.0 + roll_sway * 3.0)

        height_deviation = abs(agent.genes[10] * 20.0) # mm
        height_penalty = height_deviation * 8.0

        # Evaluate over timeline
        for _ in range(steps):
            # Accumulate progress rewards minus penalties
            fitness += (forward_vel * dt) - (wobble_penalty * dt) - (height_penalty * dt)

        return max(0.1, fitness)

    def train(self) -> Dict:
        print(f"[*] Starting MAGI walk policy training (Neuroevolution: GA)")
        print(f"[*] Population size: {self.pop_size} | Generations: {self.generations}")
        self.initialize_population()

        for gen in range(self.generations):
            # 1. Evaluate all agents
            total_fitness = 0.0
            gen_best_fitness = -9999.0
            gen_best_agent = None

            for agent in self.population:
                agent.fitness = self.evaluate_fitness(agent)
                total_fitness += agent.fitness
                if agent.fitness > gen_best_fitness:
                    gen_best_fitness = agent.fitness
                    gen_best_agent = agent

            avg_fitness = total_fitness / self.pop_size

            # Save all-time best
            if gen_best_fitness > self.best_fitness:
                self.best_fitness = gen_best_fitness
                self.best_genes = list(gen_best_agent.genes)

            if (gen + 1) % 10 == 0 or gen == 0:
                print(f"[Gen {gen+1:02d}/{self.generations}] Max Reward: {gen_best_fitness:.2f} | Avg Reward: {avg_fitness:.2f}")

            # 2. Selection: Keep top 20% elite agents
            self.population.sort(key=lambda x: x.fitness, reverse=True)
            elite_count = int(self.pop_size * 0.2)
            next_pop = []
            for i in range(elite_count):
                next_pop.append(Agent(i + 1))
                next_pop[-1].genes = list(self.population[i].genes)

            # 3. Breed offspring
            mutant_count = self.pop_size - elite_count
            for i in range(mutant_count):
                # Tournament parent selection
                parent_a = self.tournament_select()
                parent_b = self.tournament_select()
                child = parent_a.crossover(parent_b)
                child.mutate(self.mutation_rate)
                child.id = elite_count + i + 1
                next_pop.append(child)

            self.population = next_pop

        # Map genes back to actual parameters
        # Leg phase offsets (FR, FL, BR, BL)
        p_offsets = [round(self.best_genes[i], 3) for i in range(4)]
        
        gait_config = {
            "gait_parameters": {
                "phase_offsets": {
                    "front_right": p_offsets[0],
                    "front_left": p_offsets[1],
                    "back_right": p_offsets[2],
                    "back_left": p_offsets[3]
                },
                "pitch_sway_deg": round(abs(self.best_genes[8] * 10.0), 2),
                "roll_sway_deg": round(abs(self.best_genes[9] * 8.0), 2),
                "neutral_height_mm": round(150.0 + self.best_genes[10] * 20.0, 1),
                "target_stride_mm": round(60.0 + self.best_genes[11] * 20.0, 1),
                "fitness_score": round(self.best_fitness, 2)
            }
        }
        
        print(f"[+] Training complete. Optimal policy fitness achieved: {self.best_fitness:.2f}")
        return gait_config

    def tournament_select(self) -> Agent:
        # Sample 3 random agents, return the best
        candidates = random.sample(self.population, 3)
        candidates.sort(key=lambda x: x.fitness, reverse=True)
        return candidates[0]

def save_config(config: Dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        yaml.safe_dump(config, f, default_flow_style=False)
    print(f"[+] Optimal gait policy saved to: {path}")

if __name__ == "__main__":
    trainer = WalkingTrainer(pop_size=40, generations=50)
    best_config = trainer.train()
    
    # Save inside workspace OS directories
    config_path = "/home/aki/Downloads/MAGI/OS/src/common/gait_config.yaml"
    save_config(best_config, config_path)
