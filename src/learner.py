import tensorflow as tf
import numpy as np
import logging
from .utils import save_pickle, create_env, TimeSnippets
from .session_manager import session_manager
from .tree_actor import TreeActor
from .counters import TrainCounters, Ticker
from .agent import PolicyAgent, run_episodes
from .registry import get_lookahead, get_from_registry
from .data_structures import Dataset
from .settings import settings

logger = logging.getLogger(__name__)

class Learner:
    def __init__(self, log_dir='.'):
        self.log_dir = log_dir

        self.compute_summaries = session_manager.is_chief or settings["summaries"] == "all"
        self.report_logs = session_manager.is_chief or settings["logs"] == "all"

        self.env = create_env(settings["env_id"], frameskip=settings["frameskip"])

        self.set_seed()

        self.actor = TreeActor(self.env)
        
        TrainCounters.build_global_counters()
        self.build_summaries()

        session_manager.set_optimizer(self.get_optimizer())

        algorithm_class = get_from_registry("Algorithms", settings["algorithm"])
        self.algorithm = algorithm_class(actor=self.actor,
                                         obs_shape=self.env.observation_space.shape,
                                         n_actions=self.env.action_space.n)

        self.time_snippets = TimeSnippets()

        if log_dir and session_manager.is_chief:
            save_pickle(log_dir + '/args.pkl', settings)

            header = '=' * 10 + '\n' + ' ' * 5 + 'Arguments' + ' ' * 5 + '\n' + '=' * 10 + '\n'
            args_str = "\n".join(sorted([k + ": " + str(v) for k,v in settings.items()]))
            footer = '\n' + '=' * 39 + '\n'
            logger.info(header + args_str + footer)

    def set_seed(self):
        np.random.seed(settings["seed"])
        tf.set_random_seed(settings["seed"])
        self.env.seed(settings["seed"])
        logger.warning("Seed set to %s" % str(settings["seed"]))
        return settings["seed"]

    def save_model(self):
        self.algorithm.global_network.save(session_manager.session, self.log_dir + '/network_params_%i.pkl'%TrainCounters["global_interactions"])

    def train(self, batch):
        train_op, feed_dict = self.algorithm.get_train_op(batch)
        inc_counters_op, feed_dict = TrainCounters.get_inc_op(feed_dict = feed_dict)
        
        _, _, loss, train_summaries = session_manager.session.run([train_op, inc_counters_op, self.algorithm.loss, self.algorithm.summaries_op], feed_dict=feed_dict)
        TrainCounters.update_global_counters()

        #check for nan in loss
        if not np.isfinite(loss):
            raise Exception("Loss = %s" % str(loss))

        #compute summaries
        if train_summaries is not None and self.compute_summaries:
            session_manager.summary_writer.add_summary(train_summaries, TrainCounters["global_interactions"])
        return loss

    def run(self):
        try:
            TrainCounters.update_global_counters()
            if session_manager.is_chief:
                self.last_save_interactions = TrainCounters["global_interactions"]

            while not session_manager.session.should_stop() and TrainCounters["global_interactions"] < settings["max_train_interactions"]:
                self.algorithm.update() #sync global network
                with self.time_snippets.time_snippet("generate_batch"):
                    batch = self.generate_batch()
                with self.time_snippets.time_snippet("train"):
                    loss = self.train(batch)

                #maybe save and eval
                if session_manager.is_chief and settings["save_eval_interactions"] is not None:
                    if TrainCounters["global_interactions"] - self.last_save_interactions >= settings[
                        "save_eval_interactions"]:
                        if settings["save_model"]:
                            self.save_model()
                            logger.info("Global interaction %i. Network parameters saved!" % TrainCounters[
                                "global_interactions"])
                        self.last_save_interactions = TrainCounters["global_interactions"]
                        Ticker.tick("evaluate")

        finally:
            logger.info("DONE!")
            if session_manager.is_chief:
                self.save_model()
            self.env.close()

    def get_optimizer(self):
        from src.counters import TrainCounters
        def linear_annealing(initial_value, final_step, final_value=0):
            slope = (final_value - initial_value) / final_step

            def f(current_step):
                return tf.minimum(slope * tf.cast(current_step, tf.float32) + initial_value, final_value)

            return f

        if settings["learning_rate_annealing"] == "constant":
            learning_rate_fn = lambda x: settings["learning_rate"]
        elif settings["learning_rate_annealing"] == "linear":
            learning_rate_fn = linear_annealing(settings["learning_rate"], settings["max_train_interactions"])
        else:
            raise ValueError("learning_rate_annealing has to be either 'constant' or 'linear'")

        lr = learning_rate_fn(TrainCounters.counters["global_interactions"].var)  # TODO: this is a little bit hacky, set lr as placeholder and input it.

        if settings["optimizer"] in ('rmsprop', 'RMSprop'):
            optimizer = tf.train.RMSPropOptimizer(learning_rate=lr,
                                                  decay=settings["rmsprop_decay"],
                                                  momentum=settings["rmsprop_momentum"],
                                                  epsilon=settings["rmsprop_epsilon"],
                                                  use_locking=False,
                                                  # there's just one thread: server worker, so even if it is True there won't be any locking (From docs, I think this flag doesn't work for distributed tensorflow, but see this: https://stackoverflow.com/questions/43147435/how-does-asynchronous-training-work-in-distributed-tensorflow)
                                                  centered=settings["rmsprop_centered"])
        elif settings["optimizer"] in ('adam', 'Adam'):
            optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        else:
            raise NotImplementedError

        return optimizer

    def generate_batch(self):
        raise NotImplementedError("Child classes should implement this method.")

    def get_eval_actor(self):
        raise NotImplementedError("Child classes should implement this method.")
                
    def build_summaries(self):
        if session_manager.is_chief and settings["eval_episodes"] > 0:
            session_manager.add_scalar_summaries(['eval_reward_per_episode', 'eval_transitions_per_episode', 'eval_reward_per_transition'])
            
            def evaluate():
                eval_env = create_env(settings["env_id"], settings["frameskip"])
                actor = self.get_eval_actor()
                step_fn = actor.get_step_fn(eval_env)
                rewards, transitions = run_episodes(step_fn, num_episodes=settings["eval_episodes"])
                reward_per_episode = np.mean(rewards)
                transitions_per_episode = np.mean(transitions)
                reward_per_transitions = reward_per_episode/transitions_per_episode
                logger.info("Evaluation: Mean reward per episode %.2f.  Mean episode transitions %.2f. Mean reward per transition %.2f."% (reward_per_episode, transitions_per_episode, reward_per_transitions))
                session_manager.compute_summaries([('eval_reward_per_episode', reward_per_episode), ('eval_transitions_per_episode', transitions_per_episode), ('eval_reward_per_transition', reward_per_transitions)], TrainCounters["global_interactions"])
                return reward_per_episode, transitions_per_episode
            Ticker.add(evaluate, "evaluate")

        if self.compute_summaries or self.report_logs:
            session_manager.add_scalar_summaries(['train_reward_per_episode', 'train_transitions_per_episode', 'train_reward_per_transition'])
            def report_episode():
                episode_reward = TrainCounters["episode_reward"]
                episode_transitions = TrainCounters["episode_transitions"]
                reward_per_transition = episode_reward/episode_transitions
                if self.report_logs:
                    logger.info("Episode: %i. Global interactions %i. Global transitions %i.  Reward: %.2f.  Episode transitions: %i. Reward/transition: %.2f."
                                % (TrainCounters["local_episodes"], TrainCounters["global_interactions"], TrainCounters["global_transitions"], episode_reward, episode_transitions, reward_per_transition))
                if self.compute_summaries:
                    session_manager.compute_summaries([('train_reward_per_episode', episode_reward), ('train_transitions_per_episode', episode_transitions), ('train_reward_per_transition', reward_per_transition)], TrainCounters["global_interactions"])
            Ticker.add(report_episode, "episode")

            session_manager.add_scalar_summaries(['generate_batch_time', 'train_time'])
            def batch_train_times():
                session_manager.compute_summaries([("generate_batch_time", self.time_snippets.get_average_time("generate_batch")), ("train_time", self.time_snippets.get_average_time("train"))], TrainCounters["global_interactions"])
                self.time_snippets.reset(["generate_batch", "train_time"])
            Ticker.add(batch_train_times, "episode")



class LookaheadLearner(Learner):
    """
    Waits until an episode has ended before adding nodes to the dataset. The
    value is then learned from the episode return (no bootstrapping).
    Initializes the dataset with one episode. Then, for each step, it trains
    the neural network with a batch sampled from the dataset.
    """
    ALLOW_RESTORE = True

    def __init__(self, *args, **kwargs):
        Learner.__init__(self, *args, **kwargs)

        TrainCounters.new_counter("expanded_nodes", add_to_groups=["episode"])
        TrainCounters.new_counter("tree_nodes", add_to_groups=["episode"])
        TrainCounters.new_counter("tree_depth", add_to_groups=["episode"])

        NN_planner = settings["trainedNN_planner"] if settings[
            "trainedNN_planner"] else self.algorithm.global_network
        NN_features = settings["trainedNN_features"] if settings[
            "trainedNN_features"] else self.algorithm.global_network

        self.lookahead, feature_extractor = get_lookahead(actor=self.actor,
                                                          obs_shape=self.env.observation_space.shape,
                                                          n_actions=self.env.action_space.n,
                                                          lookahead_class=settings["lookahead"],
                                                          planner_class=settings["planner"],
                                                          features_class=settings["features"],
                                                          NN_planner=NN_planner,
                                                          NN_features=NN_features)

        assert settings[
                   "dataset_max_transitions"] is not None, "Specify a maximum of transitions to be stored in the dataset."
        assert settings["dataset_min_transitions"] <= settings["dataset_max_transitions"]

    def perform_step(self):
        trajectory, expanded_nodes = self.lookahead.step()
        parent_data, child_data = trajectory[-2], trajectory[-1]
        TrainCounters.increment("expanded_nodes", expanded_nodes)
        TrainCounters.increment("tree_nodes", len(self.actor.tree))
        TrainCounters.increment("tree_depth", self.actor.tree.max_depth)

        r, done = child_data["r"], child_data["done"]

        TrainCounters.increment("episode_reward", r)
        TrainCounters.increment("transitions")
        Ticker.tick("transition")
        if done:
            TrainCounters.increment("local_episodes")
            Ticker.tick("episode")
            TrainCounters.reset("episode")
        return self.algorithm.get_transition(parent_data, child_data), r, done

    def get_eval_actor(self):
        return PolicyAgent(self.algorithm.policy)

    def build_summaries(self):
        Learner.build_summaries(self)
        if self.compute_summaries or self.report_logs:
            session_manager.add_scalar_summaries(['expanded_nodes', 'tree_nodes', 'tree_depth'])

            def tree_statistics():
                episode_transitions = TrainCounters["episode_transitions"]
                expanded_nodes = TrainCounters["expanded_nodes"] / episode_transitions
                tree_nodes = TrainCounters["tree_nodes"] / episode_transitions
                tree_depth = TrainCounters["tree_depth"] / episode_transitions
                if self.report_logs:
                    logger.info("Avg expanded nodes per step: %.2f. Avg tree nodes per step: %.2f" % (
                    expanded_nodes, tree_nodes))
                if self.compute_summaries:
                    session_manager.compute_summaries(
                        [("expanded_nodes", expanded_nodes), ("tree_nodes", tree_nodes), ("tree_depth", tree_depth)],
                        TrainCounters["global_interactions"])

            Ticker.add(tree_statistics, "episode")

        if settings["save_model_episodes"]:
            Ticker.add(self.save_model, "episode")

    def _add_transitions_to_dataset(self, transitions, rewards):
        if self.add_returns:
            R = 0
            for i in range(len(transitions) - 1, -1, -1):
                R = rewards[i] + settings["discount_factor"] * R
                transitions[i]['return'] = R
        for t in transitions:
            self.dataset.add(t)

    def run(self):
        assert self.actor.tree is None

        self.algorithm.update()  # sync network
        self.actor.reset_env()

        logger.info("Initializing dataset.")
        t, r, done = self.perform_step()
        transitions = [t]
        rewards = [r]

        names = list(t.keys())

        if settings["planner"] == "MCTSAlphaZero":
            self.add_returns = True
            names.append("return")
        else:
            self.add_returns = False

        self.dataset = Dataset(names, max_len=settings["dataset_max_transitions"], info=settings)

        # initialize datset
        if done:
            self._add_transitions_to_dataset(transitions, rewards)
            transitions, rewards = list(), list()

        while len(self.dataset) < settings["local_batch_size"] or len(self.dataset) < settings[
            "dataset_min_transitions"]:
            # run_episode
            done = False
            while not done:
                t, r, done = self.perform_step()
                transitions.append(t)
                rewards.append(r)
            self._add_transitions_to_dataset(transitions, rewards)
            transitions, rewards = list(), list()

        logger.info("Starting with a dataset of %i transitions." % len(self.dataset))
        self.episode_transitions = list()
        self.episode_rewards = list()
        return Learner.run(self)

    def generate_batch(self):
        assert self.actor.tree is not None

        t, r, done = self.perform_step()
        self.episode_transitions.append(t)
        self.episode_rewards.append(r)
        if done:
            self._add_transitions_to_dataset(self.episode_transitions, self.episode_rewards)
            self.episode_transitions = list()
            self.episode_rewards = list()

        transitions = self.dataset.sample(settings["local_batch_size"])
        return transitions