import torch
# print(torch.__version__)
import torch.nn as nn
from torch.distributions.normal import Normal
import numpy as np

from models.modulator import Modulator
from utils.util_meta import update_parameters
from utils.util_data import integers_to_symbols, symbols_to_integers, add_cartesian_awgn_tensor
from utils.util_data import torch_tensor_to_numpy as to_numpy
from utils.util_data import numpy_to_torch_tensor as to_tensor
from utils.util_lookup_table import BER_lookup_table
lookup_table = BER_lookup_table()


class MAMLModulatorGradientPassing(Modulator):
    """Gradient Passing updated modulator"""
    def __init__(self, *,
                 stepsize_meta: float = 1e-3,
                 stepsize_inner: float = 1e-1,
                 first_order: bool = False,
                 inner_steps: int = 10,
                 inner_batch_size: int = 16,
                 outer_batch_size: int = 32,
                 SNR_db: float = 10,
                 **kwargs
                 ):
        """
        For the MAML Modulator, the `optimizer` only defines the optimizer
        for the outer meta-learning loop. The inner fast adaptation loop
        is hand-written SGD because PyTorch does not support differentiation
        through parameter updates.
        """
        super(MAMLModulatorGradientPassing, self).__init__(**kwargs)
        self.stepsize_meta = torch.tensor(stepsize_meta).to(self.device)
        self.stepsize_inner = torch.tensor(stepsize_inner).to(self.device)
        self.first_order = first_order
        self.inner_steps = inner_steps
        self.inner_batch_size = inner_batch_size
        self.outer_batch_size = outer_batch_size
        self.SNR_db = SNR_db
        self.verbose = False
        if 'verbose' in kwargs:
            self.verbose = kwargs['verbose']

        # Override the base class's optimizer with meta-params and meta-lr
        optimizers = {
            'adam': torch.optim.Adam,
            'sgd': torch.optim.SGD,
        }
        if kwargs['optimizer'] and hasattr(self.model, 'named_parameters') and not hasattr(self.model, "update"):
            optimizer = optimizers[kwargs['optimizer'].lower()]
            self.param_dicts = [
                {'params': self.model.meta_parameters().values(), 'lr': stepsize_meta},
            ]
            self.optimizer = optimizer(self.param_dicts, lr=stepsize_meta)

    def update_maml(self, demods):
        model = self.model
        ntasks = len(demods)
        self.optimizer.zero_grad()
        outer_loss = torch.tensor(0.).to(self.device)
        outer_ber = 0.
        # Calculate inner losses and take fast steps
        for it in range(ntasks):
            demod = demods[it]
#             for demod in demods:
            # Generate random symbols to modulate
            meta_loss, inner_loss, meta_ber = self._update_inner(model, demod)

            outer_loss += meta_loss
            outer_ber += meta_ber

            if self.verbose:
                print("meta_ber:", meta_ber, "meta loss:", meta_loss.detach().cpu().numpy(), "inner_loss:", inner_loss)
            # Gradient update on mean task meta-loss
        outer_loss /= ntasks
        outer_ber /= ntasks
        outer_loss.backward()
        self.optimizer.step()
        return to_numpy(outer_loss), outer_ber

    def update_test(self, demod, nshots, batch_size=32, step_size=1e-2):
        model = self.model
        l1, l2 = [self.lambda_l1, self.lambda_l2]
        criterion = nn.CrossEntropyLoss(reduction='mean')
        bers = []
        params = model.meta_parameters()
        for _ in range(nshots):
            data_si = np.random.randint(0, 2**self.bits_per_symbol, size=[batch_size])
            data_sb = to_tensor(integers_to_symbols(data_si, self.bits_per_symbol)).float().to(self.device)
            data_c = model.forward(data_sb, params=params)
            data_c_noisy = add_cartesian_awgn_tensor(data_c, SNR_db=self.SNR_db, device=self.device)

            logits = demod.demodulate_tensor(data_c_noisy)
            labels_sb_g = demod.demodulate(to_numpy(data_c_noisy))
            # Mean bit errors
            ber = np.sum(np.abs(labels_sb_g - to_numpy(data_sb))) / labels_sb_g.size
            bers.append(ber)
            loss = criterion(logits, to_tensor(data_si).to(self.device).view([-1]).long())
            if l1 > 0:
                loss += l1 * model.l1_loss(params.values())
            if l2 > 0:
                loss += l2 * model.l2_loss(params.values())
            # Update params: performs one SGD step
            model.zero_grad()
            params = update_parameters(model, loss, params=params,
                                       step_size=step_size,
                                       first_order=True)
            # inner_loss = loss.detach().cpu().numpy()
        # Measure BER after final shot
        with torch.no_grad():
            data_si = np.random.randint(0, 2**self.bits_per_symbol, size=[batch_size])
            data_sb = to_tensor(integers_to_symbols(data_si, self.bits_per_symbol)).float().to(self.device)
            data_c = model.forward(data_sb, params=params)
            data_c_noisy = add_cartesian_awgn_tensor(data_c, SNR_db=self.SNR_db, device=self.device)
            labels_sb_g = demod.demodulate(to_numpy(data_c_noisy))
            # Mean bit errors
            ber = np.sum(np.abs(labels_sb_g - to_numpy(data_sb))) / labels_sb_g.size
            bers.append(ber)
        return bers

    def _update_inner(self, model, demod):
        params = model.meta_parameters()
        l1, l2 = [self.lambda_l1, self.lambda_l2]
        criterion = nn.CrossEntropyLoss(reduction='mean')
        inner_loss = 0.
        for _ in range(self.inner_steps):
            data_si = np.random.randint(0, 2**self.bits_per_symbol, size=[self.inner_batch_size])
            data_sb = to_tensor(integers_to_symbols(data_si, self.bits_per_symbol)).float().to(self.device)
            data_c = model.forward(data_sb, params=params)
            data_c_noisy = add_cartesian_awgn_tensor(data_c, SNR_db=self.SNR_db, device=self.device)
            logits = demod.demodulate_tensor(data_c_noisy)
            loss = criterion(logits, to_tensor(data_si).to(self.device).view([-1]).long())

            if l1 > 0:
                loss += l1 * model.l1_loss(params.values())
            if l2 > 0:
                loss += l2 * model.l2_loss(params.values())
            # Update params: performs one SGD step
            model.zero_grad()
            params = update_parameters(model, loss, params=params,
                                       step_size=self.stepsize_inner,
                                       first_order=self.first_order)
            inner_loss = loss.detach().cpu().numpy()
        # Calculate meta-loss for this task
        meta_data_si = np.random.randint(0, 2**self.bits_per_symbol, size=[self.outer_batch_size])
        meta_data_sb = to_tensor(integers_to_symbols(meta_data_si, self.bits_per_symbol)).float().to(self.device)
        meta_data_c = model.forward(meta_data_sb, params=params)
        meta_data_c_noisy = add_cartesian_awgn_tensor(meta_data_c, SNR_db=self.SNR_db, device=self.device)
        meta_logits = demod.demodulate_tensor(meta_data_c_noisy)
        meta_loss = criterion(meta_logits, to_tensor(meta_data_si).to(self.device).view([-1]).long())
#         if self.verbose:
#                 visualize_constellation(to_numpy(meta_data_c_noisy), meta_data_si)
        meta_labels_sb_g = demod.demodulate(to_numpy(meta_data_c_noisy))
        # BER mean
        meta_ber = np.sum(np.abs(meta_labels_sb_g - to_numpy(meta_data_sb))) / meta_labels_sb_g.size
        if l1 > 0:
            meta_loss += l1 * model.l1_loss(params.values)
        if l2 > 0:
            meta_loss += l2 * model.l2_loss(params.values)

        return meta_loss, inner_loss, meta_ber


class MAMLModulatorLossPassing(Modulator):
    def __init__(self, *,
                 stepsize_meta: float = 1e-3,
                 stepsize_inner: float = 1e-1,
                 first_order: bool = True,
                 inner_steps: int = 10,
                 inner_batch_size: int = 16,
                 outer_batch_size: int = 32,
                 SNR_db: float = 10,
                 std_explore: float = 1e-3,
                 **kwargs
                 ):
        """
        For the MAML Demodulator, the `optimizer` only defines the optimizer
        for the outer meta-learning loop. The inner fast adaptation loop
        is hand-written SGD because PyTorch does not support differentiation
        through parameter updates.
        """
        super(MAMLModulatorLossPassing, self).__init__(**kwargs)
        self.stepsize_meta = torch.tensor(stepsize_meta).to(self.device)
        self.stepsize_inner = torch.tensor(stepsize_inner).to(self.device)
        self.first_order = first_order
        self.inner_steps = inner_steps
        self.inner_batch_size = inner_batch_size
        self.outer_batch_size = outer_batch_size
        self.SNR_db = SNR_db
        self.std_explore = std_explore
        self.log_std = nn.Parameter(np.log(std_explore) * torch.ones(2).to(self.device))
        self.verbose = False
        if 'verbose' in kwargs:
            self.verbose = kwargs['verbose']

        # Override the base class's optimizer with meta-params and meta-lr
        optimizers = {
            'adam': torch.optim.Adam,
            'sgd': torch.optim.SGD,
        }
        if kwargs['optimizer'] and hasattr(self.model, 'named_parameters') and not hasattr(self.model, "update"):
            optimizer = optimizers[kwargs['optimizer'].lower()]
            self.param_dicts = [
                {'params': list(self.model.meta_parameters().values()), 'lr': stepsize_meta},
            ]
            self.optimizer = optimizer(self.param_dicts, lr=stepsize_meta)

    def update_maml(self, demods):
        # Calculate meta-loss for this task
        if self.loss_function == 'ppo':
            loss, ber, inner_loss = self._update_maml_ppo(demods)
        elif self.loss_function == 'vanilla_pg':
            loss, ber, inner_loss = self._update_maml_vanilla_pg(demods)
        else:
            raise NotImplementedError("unknown loss function %s" % self.loss_function)

        if self.verbose:
            print("meta_ber:", ber, "meta loss:", loss, "inner_loss:", inner_loss)
        return loss, ber

    def _update_maml_ppo(self, demods):
        model = self.model
        ntasks = len(demods)
        meta_data_si = np.random.randint(0, 2**self.bits_per_symbol, size=[self.outer_batch_size])
        meta_data_sb = to_tensor(integers_to_symbols(meta_data_si, self.bits_per_symbol)).float().to(self.device)
        prev_policies = []  # Needs to persist across PPO epochs, but be generated after the first one
        final_ber = 0.
        final_loss = 0.
        final_inner_loss = 0.
        for e in range(1):#self.ppo_epochs):
            self.optimizer.zero_grad()
            outer_loss = torch.tensor(0.).to(self.device)
            outer_ber = 0.
            inner_loss_avg = 0.
            # Calculate inner losses and take fast steps
            for it in range(ntasks):
                demod = demods[it]
                params = model.meta_parameters()
                #print("params sum", params['linear0.weight'].sum())
                params, inner_loss = self._update_inner(model, params, demod)
                inner_loss_avg += inner_loss / ntasks
                meta_means = model.forward(meta_data_sb, params=params)
                meta_policy = Normal(meta_means, self.std_explore)
                meta_data_c = meta_policy.sample()
                # Store prev_log_prob on first iteration
                if e == 0:
                    prev_policies.append(Normal(meta_means.detach(), self.std_explore))
                # Send through channel and get reward
                meta_data_c_noisy = add_cartesian_awgn_tensor(meta_data_c, SNR_db=self.SNR_db, device=self.device)
                meta_labels_sb_g = demod.demodulate(to_numpy(meta_data_c_noisy))
                meta_bit_loss = np.sum(np.abs(meta_labels_sb_g - to_numpy(meta_data_sb)), 1)
                meta_reward = torch.from_numpy(-meta_bit_loss).to(self.device)
                prev_log_prob = torch.log(prev_policies[it].log_prob(meta_data_c.detach()).exp() + self.lambda_prob).sum(dim=1)
                #print("prev_log_prob", prev_log_prob)
                meta_loss = self._loss_ppo(meta_policy, meta_reward, meta_data_c, prev_log_prob, params, verbose=True)
                outer_loss += meta_loss
                #print("Meta loss @{},{}: {}".format(e, it, meta_loss))
                #print()
                # For ber calculations
                with torch.no_grad():
                    meta_data_c_ber = meta_means
                    meta_data_c_noisy_ber = add_cartesian_awgn_tensor(meta_data_c_ber, SNR_db=self.SNR_db, device=self.device)
                    meta_labels_sb_g_ber = demod.demodulate(to_numpy(meta_data_c_noisy_ber))
                    meta_ber = np.mean(np.mean(np.abs(meta_labels_sb_g_ber - to_numpy(meta_data_sb)), 1), 0)
                    outer_ber += meta_ber
            # Gradient update on mean task meta-loss
            outer_loss /= ntasks
            outer_ber /= ntasks
            outer_loss.backward()
            self.optimizer.step()
            final_ber = outer_ber
            final_loss = outer_loss
            final_inner_loss = inner_loss_avg
        return to_numpy(final_loss), final_ber, final_inner_loss

    def _update_maml_vanilla_pg(self, demods):
        model = self.model
        ntasks = len(demods)
        self.optimizer.zero_grad()
        outer_loss = torch.tensor(0.).to(self.device)
        outer_ber = 0.
        inner_loss_avg = 0.
        # Calculate inner losses and take fast steps
        for it in range(ntasks):
            # Inner update for task
            demod = demods[it]
            params = model.meta_parameters()
            params, inner_loss = self._update_inner(model, params, demod)
            inner_loss_avg += inner_loss / ntasks
            # Meta update
            meta_data_si = np.random.randint(0, 2**self.bits_per_symbol, size=[self.outer_batch_size])
            meta_data_sb = to_tensor(integers_to_symbols(meta_data_si, self.bits_per_symbol)).float().to(self.device)
            meta_means = model.forward(meta_data_sb, params=params)
            meta_policy = Normal(meta_means, self.std_explore)
            meta_data_c = meta_policy.sample()
            meta_data_c_noisy = add_cartesian_awgn_tensor(meta_data_c, SNR_db=self.SNR_db, device=self.device)
            meta_labels_sb_g = demod.demodulate(to_numpy(meta_data_c_noisy))
            meta_bit_loss = np.sum(np.abs(meta_labels_sb_g - to_numpy(meta_data_sb)), 1)
            meta_reward = torch.from_numpy(-meta_bit_loss).to(self.device)
            meta_loss = self._loss_vanilla_pg(meta_reward, meta_data_c, meta_policy, params)
            outer_loss += meta_loss
            # For ber calculations
            with torch.no_grad():
                meta_data_c_ber = meta_means
                meta_data_c_noisy_ber = add_cartesian_awgn_tensor(meta_data_c_ber, SNR_db=self.SNR_db, device=self.device)
                meta_labels_sb_g_ber = demod.demodulate(to_numpy(meta_data_c_noisy_ber))
                outer_ber += np.mean(np.mean(np.abs(meta_labels_sb_g_ber - to_numpy(meta_data_sb)), 1), 0)
        # Gradient update on mean task meta-loss
        outer_loss /= ntasks
        outer_ber /= ntasks
        outer_loss.backward()
        self.optimizer.step()
        return to_numpy(outer_loss), outer_ber, inner_loss_avg

    def _update_inner(self, model, params, demod):
        # params = model.meta_parameters()
        # l1, l2 = [self.lambda_l1, self.lambda_l2]
        inner_loss = 0.

        for ib in range(self.inner_steps):
            # Act and get a reward
            data_si = np.random.randint(0, 2**self.bits_per_symbol, size=[self.inner_batch_size])
            data_sb = to_tensor(integers_to_symbols(data_si, self.bits_per_symbol)).float().to(self.device)
            means = model.forward(data_sb, params=params)
            policy = Normal(means, self.std_explore)
            data_c = policy.sample()
            data_c_noisy = add_cartesian_awgn_tensor(data_c, SNR_db=self.SNR_db, device=self.device)
            labels_sb_g = demod.demodulate(to_numpy(data_c_noisy))
            bit_loss = np.sum(np.abs(labels_sb_g - to_numpy(data_sb)), 1)
            reward = torch.from_numpy(-bit_loss).to(self.device)
            # Calculate loss and perform update
            loss = None  # Make sure loss is in outer scope for PPO
            if self.loss_function == 'ppo':
                prev_log_prob = torch.log(policy.log_prob(data_c).exp() + self.lambda_prob).sum(dim=1).detach()
                for e in range(self.ppo_epochs):
                    i0 = 0  # e * (len(reward) // self.ppo_epochs)
                    i1 = len(reward)  # i0 + len(reward) // self.ppo_epochs
                    mini_reward = reward[i0:i1]
                    mini_actions = data_c[i0:i1, :]
                    mini_prev_log_prob = prev_log_prob[i0:i1]
                    curr_policy = Normal(model.forward(data_sb[i0:i1, :], params), self.std_explore)
                    loss = self._loss_ppo(curr_policy, mini_reward, mini_actions, mini_prev_log_prob, params)
                    # print("Update {} loss {}".format(e, to_numpy(loss)))
                    # Update params: performs one SGD step
                    model.zero_grad()
                    params = update_parameters(model, loss, params=params,
                                               step_size=self.stepsize_inner,
                                               first_order=self.first_order)
            elif self.loss_function == 'vanilla_pg':
                loss = self._loss_vanilla_pg(reward, data_c, policy, params)
                # Update params: performs one SGD step
                model.zero_grad()
                params = update_parameters(model, loss, params=params,
                                           step_size=self.stepsize_inner,
                                           first_order=self.first_order)
            else:
                raise NotImplementedError("unknown loss function %s" % self.loss_function)
            inner_loss = to_numpy(loss)
        return params, inner_loss

    # Vanilla Policy Gradient loss, sufficient for simulation
    def _loss_vanilla_pg(self, reward, actions, policy, params):
        log_probs = torch.log(policy.log_prob(actions).exp() + self.lambda_prob).sum(dim=1)
        baseline = torch.mean(reward)
        loss = -torch.mean(log_probs * (reward - self.lambda_baseline * baseline))
        if self.lambda_center > 0:
            loss += self.lambda_center * self.model.location_loss(params)
        if self.lambda_l1 > 0:
            loss += self.lambda_l1 * self.model.l1_loss(params)
        if self.lambda_l2 > 0:
            loss += self.lambda_l2 * self.model.l2_loss(params)
        return loss

    # Proximal Policy Optimization loss, for efficiency when we are over the air...
    # we can train more on the single preamble we get. So sending a large preamble
    # and training mini-batches over it. Lots of overhead to sending over air!
    def _loss_ppo(self, policy, reward, actions, prev_log_prob, params, verbose=False):
        """https://arxiv.org/abs/1707.06347"""
        with torch.no_grad():
            baseline = torch.mean(reward)
        adv = reward# - self.lambda_baseline * baseline
        log_prob = torch.log(policy.log_prob(actions).exp() + self.lambda_prob).sum(dim=1)
        ratio = (log_prob - prev_log_prob).exp()
        if verbose:
            #print(ratio)
            pass
        min_adv = torch.where(adv > 0, (1 + self.ppo_clip_ratio) * adv,
                                       (1 - self.ppo_clip_ratio) * adv)
        loss = -torch.min(ratio * adv, min_adv).mean()
        if self.lambda_center > 0:
            loss += self.lambda_center * self.model.location_loss(params)
        if self.lambda_l1 > 0:
            loss += self.lambda_l1 * self.model.l1_loss(params)
        if self.lambda_l2 > 0:
            loss += self.lambda_l2 * self.model.l2_loss(params)
        return loss

    def update_test(self, demod, nshots, batch_size=32, step_size=1e-2):
        model = self.model
        l1, l2 = [self.lambda_l1, self.lambda_l2]
        bers = []
        params = model.meta_parameters()
        # Calculate BER before any updates
        with torch.no_grad():
            data_si = np.random.randint(0, 2**self.bits_per_symbol, size=[batch_size])
            data_sb = to_tensor(integers_to_symbols(data_si, self.bits_per_symbol)).float().to(self.device)
            data_c = model.forward(data_sb, params=params)
            data_c_noisy = add_cartesian_awgn_tensor(data_c, SNR_db=self.SNR_db, device=self.device)

            labels_sb_g = demod.demodulate(to_numpy(data_c_noisy))
            bit_loss = np.sum(np.abs(labels_sb_g - to_numpy(data_sb)), 1)
            bers.append(np.mean(bit_loss) / self.bits_per_symbol)
        # Do updates
        for i in range(nshots):
            # Act and get a reward
            data_si = np.random.randint(0, 2**self.bits_per_symbol, size=[batch_size])
            data_sb = to_tensor(integers_to_symbols(data_si, self.bits_per_symbol)).float().to(self.device)
            means = model.forward(data_sb, params=params)
            policy = Normal(means, self.std_explore)
            data_c = policy.sample()
            data_c_noisy = add_cartesian_awgn_tensor(data_c, SNR_db=self.SNR_db, device=self.device)
            labels_sb_g = demod.demodulate(to_numpy(data_c_noisy))
            bit_loss = np.sum(np.abs(labels_sb_g - to_numpy(data_sb)), 1)
            reward = torch.from_numpy(-bit_loss).to(self.device)
            # Calculate loss and perform update
            if self.loss_function == 'ppo':
                prev_log_prob = torch.log(policy.log_prob(data_c).exp() + self.lambda_prob).sum(dim=1).detach()
                for e in range(self.ppo_epochs):
                    i0 = 0  # e * (len(reward) // self.ppo_epochs)
                    i1 = len(reward)  # i0 + len(reward) // self.ppo_epochs
                    mini_reward = reward[i0:i1]
                    mini_actions = data_c[i0:i1, :]
                    mini_prev_log_prob = prev_log_prob[i0:i1]
                    curr_policy = Normal(model.forward(data_sb[i0:i1, :], params), self.std_explore)
                    loss = self._loss_ppo(curr_policy, mini_reward, mini_actions, mini_prev_log_prob, params)
                    # Update params: performs one SGD step
                    model.zero_grad()
                    params = update_parameters(model, loss, params=params,
                                               step_size=self.stepsize_inner,
                                               first_order=True)
            elif self.loss_function == 'vanilla_pg':
                loss = self._loss_vanilla_pg(reward, data_c, policy, params)
                # Update params: performs one SGD step
                model.zero_grad()
                params = update_parameters(model, loss, params=params,
                                           step_size=self.stepsize_inner,
                                           first_order=True)
            else:
                raise NotImplementedError("unknown loss function %s" % self.loss_function)
            # Test current BER
            with torch.no_grad():
                data_si = np.random.randint(0, 2**self.bits_per_symbol, size=[batch_size])
                data_sb = to_tensor(integers_to_symbols(data_si, self.bits_per_symbol)).float().to(self.device)
                data_c = model.forward(data_sb, params=params)
                data_c_noisy = add_cartesian_awgn_tensor(data_c, SNR_db=self.SNR_db, device=self.device)

                labels_sb_g = demod.demodulate(to_numpy(data_c_noisy))
                bit_loss = np.sum(np.abs(labels_sb_g - to_numpy(data_sb)), 1)
                bers.append(np.mean(bit_loss) / self.bits_per_symbol)
        return bers

    def update_test_dev(self, demod, nshots, train_batch_size=32, test_batch_size=32,
                        step_size=1e-2, test_bers=[1e-5, 1e-4, 1e-3, 1e-2]):
        test_snrs = [lookup_table.get_optimal_SNR_for_BER(test_ber, self.bits_per_symbol) for test_ber in test_bers]
        model = self.model
        l1, l2 = [self.lambda_l1, self.lambda_l2]
        bers = {}
        for test_SNR_db in test_snrs:
            bers[test_SNR_db] = []
        params = model.meta_parameters()
        # Calculate BER before any updates
        with torch.no_grad():
            for test_SNR_db in test_snrs:
                data_si = np.random.randint(0, 2**self.bits_per_symbol, size=[test_batch_size])
                data_sb = to_tensor(integers_to_symbols(data_si, self.bits_per_symbol)).float().to(self.device)
                data_c = model.forward(data_sb, params=params)
                data_c_noisy = add_cartesian_awgn_tensor(data_c, SNR_db=test_SNR_db, device=self.device)

                labels_sb_g = demod.demodulate(to_numpy(data_c_noisy))
                bit_loss = np.sum(np.abs(labels_sb_g - to_numpy(data_sb)), 1)
                bers[test_SNR_db].append(np.mean(bit_loss) / self.bits_per_symbol)
        # Do updates
        for i in range(nshots):
            data_si = np.random.randint(0, 2**self.bits_per_symbol, size=[train_batch_size])
            data_sb = to_tensor(integers_to_symbols(data_si, self.bits_per_symbol)).float().to(self.device)
            means = model.forward(data_sb, params=params)
            policy = Normal(means, self.std_explore)
            data_c = policy.sample()
            log_probs = torch.log(policy.log_prob(data_c).exp() + self.lambda_prob).sum(dim=1)
            data_c_noisy = add_cartesian_awgn_tensor(data_c, SNR_db=self.SNR_db, device=self.device)
            labels_sb_g = demod.demodulate(to_numpy(data_c_noisy))
            bit_loss = np.sum(np.abs(labels_sb_g - to_numpy(data_sb)), 1)
            reward = torch.from_numpy(-bit_loss).to(self.device)
            # Calculate loss and perform update
            if self.loss_function == 'ppo':
                prev_log_prob = torch.log(policy.log_prob(data_c).exp() + self.lambda_prob).sum(dim=1).detach()
                for e in range(self.ppo_epochs):
                    i0 = 0  # e * (len(reward) // self.ppo_epochs)
                    i1 = len(reward)  # i0 + len(reward) // self.ppo_epochs
                    mini_reward = reward[i0:i1]
                    mini_actions = data_c[i0:i1, :]
                    mini_prev_log_prob = prev_log_prob[i0:i1]
                    curr_policy = Normal(model.forward(data_sb[i0:i1, :], params), self.std_explore)
                    loss = self._loss_ppo(curr_policy, mini_reward, mini_actions, mini_prev_log_prob, params)
                    # Update params: performs one SGD step
                    model.zero_grad()
                    params = update_parameters(model, loss, params=params,
                                               step_size=self.stepsize_inner,
                                               first_order=True)
            elif self.loss_function == 'vanilla_pg':
                loss = self._loss_vanilla_pg(reward, data_c, policy, params)
                # Update params: performs one SGD step
                model.zero_grad()
                params = update_parameters(model, loss, params=params,
                                           step_size=self.stepsize_inner,
                                           first_order=True)
            else:
                raise NotImplementedError("unknown loss function %s" % self.loss_function)
            # Calculate current BER
            with torch.no_grad():
                for test_SNR_db in test_snrs:
                    data_si = np.random.randint(0, 2**self.bits_per_symbol, size=[test_batch_size])
                    data_sb = to_tensor(integers_to_symbols(data_si, self.bits_per_symbol)).float().to(self.device)
                    data_c = model.forward(data_sb, params=params)
                    data_c_noisy = add_cartesian_awgn_tensor(data_c, SNR_db=test_SNR_db, device=self.device)

                    labels_sb_g = demod.demodulate(to_numpy(data_c_noisy))
                    bit_loss = np.sum(np.abs(labels_sb_g - to_numpy(data_sb)), 1)
                    bers[test_SNR_db].append(np.mean(bit_loss) / self.bits_per_symbol)
        constellation = model.forward(to_tensor(self.all_symbols).to(self.device), params=params)
        return bers, to_numpy(constellation)
