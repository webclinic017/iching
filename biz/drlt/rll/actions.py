import numpy as np
from typing import Union


class ActionSelector:
    """
    Abstract class which converts scores to the actions
    """
    def __call__(self, scores):
        raise NotImplementedError


class ArgmaxActionSelector(ActionSelector):
    """
    Selects actions using argmax
    """
    def __call__(self, scores):
        assert isinstance(scores, np.ndarray)
        return np.argmax(scores, axis=1)


class EpsilonGreedyActionSelector(ActionSelector):
    def __init__(self, epsilon=0.05, selector=None):
        self.epsilon = epsilon
        self.selector = selector if selector is not None else ArgmaxActionSelector()
        print('epsilon: {0}; selector: {1};'.format(self.epsilon, self.selector))

    def __call__(self, scores):
        assert isinstance(scores, np.ndarray)
        self.epsilon = 0.7
        batch_size, n_actions = scores.shape
        print('batch_size={0}; n_actions={1};'.format(batch_size, n_actions))
        actions = self.selector(scores)
        print('actions: {0};'.format(actions))
        mask = np.random.random(size=batch_size) < self.epsilon
        print('mask: {0};'.format(mask))
        rand_actions = np.random.choice(n_actions, sum(mask))
        print('rand_actions: {0};'.format(rand_actions))
        print('mask type: {0}; {1};'.format(type(mask), mask[2]))
        actions[mask] = rand_actions
        print('final actions: {0};'.format(actions))
        return actions


class ProbabilityActionSelector(ActionSelector):
    """
    Converts probabilities of actions into action by sampling them
    """
    def __call__(self, probs):
        assert isinstance(probs, np.ndarray)
        actions = []
        for prob in probs:
            actions.append(np.random.choice(len(prob), p=prob))
        return np.array(actions)


class EpsilonTracker:
    """
    Updates epsilon according to linear schedule
    """
    def __init__(self, selector: EpsilonGreedyActionSelector,
                 eps_start: Union[int, float],
                 eps_final: Union[int, float],
                 eps_frames: int):
        self.selector = selector
        self.eps_start = eps_start
        self.eps_final = eps_final
        self.eps_frames = eps_frames
        self.frame(0)

    def frame(self, frame: int):
        eps = self.eps_start - frame / self.eps_frames
        self.selector.epsilon = max(self.eps_final, eps)
