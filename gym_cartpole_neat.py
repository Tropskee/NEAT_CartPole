import gym
import neat 
import os
import numpy as np
import pickle

### User Params ###

GEN = 10 # max number of generations to evolve network
game_name = 'CartPole-v0' # name of game
STEPS = 1000 # max number of steps to take per genome
RENDER = True # renders game if true
CHECKPOINT = False # loads from checkpoint if true - must specify checkpoint later
EPISODES = 1 # Number of times to run a single genome. This takes the fitness score from the worst run
REPLAY = False # Replay game with best genome

### End User Params ###

def simulate_species(net, env, EPISODES, STEPS, RENDER):
  """
  Captures inputs from gym and passes them into the feed forward neural network. 
  Decides on action(s) which are calculated from the output(s).
  New inputs and rewards are then calculated which will be recycled back into the game loop.
  Fitnesses are then created for each species.
  :return: fitness
  """
  fitnesses = []
  for runs in range(EPISODES):
    inputs = my_env.reset()
    cum_reward = 0.0
    for step in range(STEPS):
      # Activation propagates the inputs through the entire network.
      outputs = net.activate(inputs)
      # Take the highest confidence output
      action = np.argmax(outputs)
      inputs, reward, done, _ = env.step(action)
      # Decide if you want to render the game or simply train
      if RENDER:
        env.render()
      if done:
        break
      cum_reward += reward
    
    fitnesses.append(cum_reward)
  
  fitness = np.array(fitnesses).mean()
  print("Species fitness: %s" % str(fitness))
  return fitness


def train_network(env, config_path):
  """
  Sets up configuration, and creates population.
  Contains evaulate_genome, eval_fitness, and replay_genome functions -- all utilize the 'env' param.
  Starts the gym simulation and captures the best species as 'winner'.
  Includes ability to save/load winning genome via pickle
  """
  
  def evaluate_genome(genome, config):
    """
    Creates a neural network using the passed genome and config file.
    Returns simulate_species which was passed the created feed forward 
    neural network.
    :return: species from simulate_species
    """
    # Creates feed forward neural network, and passes net to
    # simulate_species function
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    return simulate_species(net, env, EPISODES, STEPS, RENDER )


  def eval_fitness(genomes, config):
    """
    Captures fitness score from 'evaluate_genome' and assigns to current genome
    """
    # genomes contains genome_id and genome information.
    # we only need the genome information here.
    # The genome fitness is then calculated and applied
    for genome_id, genome in genomes:
      fitness = evaluate_genome(genome, config)
      genome.fitness = fitness


  # Simulation configuration
  config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                              neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)
  pop = neat.population.Population(config)

  # Start Simulation
  pop.add_reporter(neat.StdOutReporter(True))
  stats = neat.StatisticsReporter()
  pop.add_reporter(stats)

  # Captures winning genome (highest fitness)
  winner = pop.run(eval_fitness, GEN)

  # Save Best Network
  with open('winner.pkl', 'wb') as f:
    pickle.dump(winner, f, 1)
    f.close()

  # Outputs best genome in terminal
  print('\nBest genome:\n{!s}'.format(winner))


def replay_genome(env, config_path, genome_path="winner.pkl"):
  """
  Uses pickle to load best genome for replay or continued training/learning.
  Change REPLAY to false if training/learning for the first time.
  """
  if REPLAY:
    # Load requried NEAT config
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction, 
                                neat.DefaultSpeciesSet, neat.DefaultStagnation, 
                                config_path)

    # Unpickle saved winner
    with open(genome_path, "rb") as f:
        genomes = pickle.load(f)

    # Call game with only the loaded genome
    winner_net = neat.nn.FeedForwardNetwork.create(genomes, config)
    for i in range(10):
      simulate_species(winner_net, env, EPISODES, STEPS, RENDER)
      

if __name__ == "__main__":
  """
  Determination path to configuation file. This path manipulation
  is here so that the script will run successfully regardless of
  current working directory.
  """
  local_dir = os.path.dirname(__file__)
  config_path = os.path.join(local_dir, 'config_gym.txt')
  my_env = gym.make(game_name)

  # If you want to witness the best genome play the game, keep REPLAY as true. 
  # otherwise change to false
  if REPLAY:
    replay_genome(my_env, config_path)
  else:
    train_network(my_env, config_path)
