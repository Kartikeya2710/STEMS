from os.path import exists
from sumo_rl import SumoEnvironment
from sumo_rl.agents import QLAgent

import pickle
import matplotlib.pyplot as plt
from sumo_rl.exploration import EpsilonGreedy

def my_reward_fn(ts):
    return ts._queue_reward()

env = SumoEnvironment(
    net_file='simple.net.xml',
    route_file='trips.trips.xml',
    use_gui=True,
    num_seconds=1000,
    yellow_time=3,
    min_green=5,
    max_green=60,
    reward_fn=my_reward_fn
)

# agent = DQN(
#         policy = "MlpPolicy",
#         env = env,
#         learning_rate=0.001,
#         buffer_size=100,
#         learning_starts = 1,
#     )

episodes = 5

out_csv = 'outputs/q_learning'

rewards = []

for episode in range(1, episodes + 1):

    initial_states = env.reset()

    if exists('model.pkl'):
        ql_agents = pickle.load(open('model.pkl', 'rb'))
    else:
        ql_agents = {ts: QLAgent(
            starting_state=env.encode(initial_states[ts], ts),
            state_space=env.observation_space,
            action_space=env.action_space,
            exploration_strategy=EpsilonGreedy(initial_epsilon=0.5, min_epsilon=0.005, decay=0.9))
            for ts in env.ts_ids
        }

    done = {'__all__': False}
    infos = []
    episode_reward = 0.0

    while not done['__all__']:
        actions = {ts: ql_agents[ts].act() for ts in ql_agents.keys()}
        obs, reward, done, info = env.step(actions)
        episode_reward += sum(list(reward.values()))

        for agent_id in ql_agents.keys():
            ql_agents[agent_id].learn(next_state=env.encode(
                obs[agent_id], agent_id), reward=reward[agent_id])

    if len(rewards) == 0 or episode_reward >= max(rewards):
        pickle.dump(ql_agents, open('model.pkl', 'wb'))


    rewards.append(episode_reward)

plt.plot(rewards)
plt.show()

# for episode in range(1, episodes+1):
#     done = False
#     env.reset()


#     # The junction here is just the junction name, example "J10"

#     for junction in env.agent_iter():
#         observation, reward, termination, truncation, info = env.last()
#         done = termination or truncation

#         if done:
#             break
#         action = randint(0, 1)
#         env.step(action)

env.close()
