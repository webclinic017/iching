import torch

from torch.nn.utils.convert_parameters import parameters_to_vector
from torch.distributions.kl import kl_divergence

from ann.dmrl.samplers import MultiTaskSampler
from ann.dmrl.metalearners.base import GradientBasedMetaLearner
from ann.dmrl.utils.torch_utils import (weighted_mean, detach_distribution,
                                       to_numpy, vector_to_parameters)
from ann.dmrl.utils.optimization import conjugate_gradient
from ann.dmrl.utils.reinforcement_learning import reinforce_loss


class MAMLTRPO(GradientBasedMetaLearner):
    """Model-Agnostic Meta-Learning (MAML, [1]) for Reinforcement Learning
    application, with an outer-loop optimization based on TRPO [2].

    Parameters
    ----------
    policy : `maml_rl.policies.Policy` instance
        The policy network to be optimized. Note that the policy network is an
        instance of `torch.nn.Module` that takes observations as input and
        returns a distribution (typically `Normal` or `Categorical`).

    fast_lr : float
        Step-size for the inner loop update/fast adaptation.

    num_steps : int
        Number of gradient steps for the fast adaptation. Currently setting
        `num_steps > 1` does not resample different trajectories after each
        gradient steps, and uses the trajectories sampled from the initial
        policy (before adaptation) to compute the loss at each step.

    first_order : bool
        If `True`, then the first order approximation of MAML is applied.

    device : str ("cpu" or "cuda")
        Name of the device for the optimization.

    References
    ----------
    .. [1] Finn, C., Abbeel, P., and Levine, S. (2017). Model-Agnostic
           Meta-Learning for Fast Adaptation of Deep Networks. International
           Conference on Machine Learning (ICML) (https://arxiv.org/abs/1703.03400)

    .. [2] Schulman, J., Levine, S., Moritz, P., Jordan, M. I., and Abbeel, P.
           (2015). Trust Region Policy Optimization. International Conference on
           Machine Learning (ICML) (https://arxiv.org/abs/1502.05477)
    """
    def __init__(self,
                 policy,
                 fast_lr=0.5,
                 first_order=False,
                 device='cpu'):
        super(MAMLTRPO, self).__init__(policy, device=device)
        self.fast_lr = fast_lr
        self.first_order = first_order

    async def adapt(self, train_futures, first_order=None):
        print('MAMLTRPO.adapt 1')
        if first_order is None:
            first_order = self.first_order
        print('MAMLTRPO.adapt 2')
        # Loop over the number of steps of adaptation
        params = None
        for futures in train_futures:
            print('    MAMLTRPO.adapt 3')
            inner_loss = reinforce_loss(self.policy,
                                        await futures,
                                        params=params)
            print('    MAMLTRPO.adapt 4')
            params = self.policy.update_params(inner_loss,
                                               params=params,
                                               step_size=self.fast_lr,
                                               first_order=first_order)
            print('    MAMLTRPO.adapt 5')
        return params

    def hessian_vector_product(self, kl, damping=1e-2):
        grads = torch.autograd.grad(kl,
                                    self.policy.parameters(),
                                    create_graph=True)
        flat_grad_kl = parameters_to_vector(grads)

        def _product(vector, retain_graph=True):
            grad_kl_v = torch.dot(flat_grad_kl, vector)
            grad2s = torch.autograd.grad(grad_kl_v,
                                         self.policy.parameters(),
                                         retain_graph=retain_graph)
            flat_grad2_kl = parameters_to_vector(grad2s)

            return flat_grad2_kl + damping * vector
        return _product

    async def surrogate_loss(self, train_futures, valid_futures, old_pi=None):
        print('MAMLTRPO.surrogate_loss 1')
        first_order = (old_pi is not None) or self.first_order
        print('MAMLTRPO.surrogate_loss 2')
        params = await self.adapt(train_futures,
                                  first_order=first_order)
        print('MAMLTRPO.surrogate_loss 3')
        with torch.set_grad_enabled(old_pi is None):
            valid_episodes = await valid_futures
            pi = self.policy(valid_episodes.observations, params=params)
            print('MAMLTRPO.surrogate_loss 4')

            if old_pi is None:
                old_pi = detach_distribution(pi)
            print('MAMLTRPO.surrogate_loss 5')
            log_ratio = (pi.log_prob(valid_episodes.actions)
                         - old_pi.log_prob(valid_episodes.actions))
            ratio = torch.exp(log_ratio)
            print('MAMLTRPO.surrogate_loss 6')
            losses = -weighted_mean(ratio * valid_episodes.advantages,
                                    lengths=valid_episodes.lengths)
            print('MAMLTRPO.surrogate_loss 7')
            kls = weighted_mean(kl_divergence(pi, old_pi),
                                lengths=valid_episodes.lengths)
            print('MAMLTRPO.surrogate_loss 8')
        return losses.mean(), kls.mean(), old_pi

    def step(self,
             train_futures,
             valid_futures,
             max_kl=1e-3,
             cg_iters=10,
             cg_damping=1e-2,
             ls_max_steps=10,
             ls_backtrack_ratio=0.5):
        num_tasks = len(train_futures[0])
        logs = {}
        print('MAMLTRPO.step 1 num_tasks={0};'.format(num_tasks))
        # Compute the surrogate loss
        old_losses, old_kls, old_pis = self._async_gather([
            self.surrogate_loss(train, valid, old_pi=None)
            for (train, valid) in zip(zip(*train_futures), valid_futures)])
        print('MAMLTRPO.step 2')
        logs['loss_before'] = to_numpy(old_losses)
        logs['kl_before'] = to_numpy(old_kls)
        print('MAMLTRPO.step 3')
        old_loss = sum(old_losses) / num_tasks
        print('MAMLTRPO.step 4')
        grads = torch.autograd.grad(old_loss,
                                    self.policy.parameters(),
                                    retain_graph=True)
        print('MAMLTRPO.step 5')
        grads = parameters_to_vector(grads)
        print('MAMLTRPO.step 6')
        # Compute the step direction with Conjugate Gradient
        old_kl = sum(old_kls) / num_tasks
        hessian_vector_product = self.hessian_vector_product(old_kl,
                                                             damping=cg_damping)
        print('MAMLTRPO.step 7')
        stepdir = conjugate_gradient(hessian_vector_product,
                                     grads,
                                     cg_iters=cg_iters)
        print('MAMLTRPO.step 8')
        # Compute the Lagrange multiplier
        shs = 0.5 * torch.dot(stepdir,
                              hessian_vector_product(stepdir, retain_graph=False))
        lagrange_multiplier = torch.sqrt(shs / max_kl)
        print('MAMLTRPO.step 9')
        step = stepdir / lagrange_multiplier

        # Save the old parameters
        old_params = parameters_to_vector(self.policy.parameters())
        print('MAMLTRPO.step 10')
        # Line search
        step_size = 1.0
        for _ in range(ls_max_steps):
            vector_to_parameters(old_params - step_size * step,
                                 self.policy.parameters())

            losses, kls, _ = self._async_gather([
                self.surrogate_loss(train, valid, old_pi=old_pi)
                for (train, valid, old_pi)
                in zip(zip(*train_futures), valid_futures, old_pis)])

            improve = (sum(losses) / num_tasks) - old_loss
            kl = sum(kls) / num_tasks
            if (improve.item() < 0.0) and (kl.item() < max_kl):
                logs['loss_after'] = to_numpy(losses)
                logs['kl_after'] = to_numpy(kls)
                break
            step_size *= ls_backtrack_ratio
        else:
            vector_to_parameters(old_params, self.policy.parameters())
        print('MAMLTRPO.step 12')

        return logs
