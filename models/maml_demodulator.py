import torch
import torch.nn as nn
import numpy as np

from models.demodulator import Demodulator
from utils.util_meta import update_parameters
from utils.util_data import integers_to_symbols
from utils.util_data import torch_tensor_to_numpy as to_numpy
from utils.util_data import numpy_to_torch_tensor as to_tensor
from utils.util_data import add_cartesian_awgn_tensor
from utils.util_lookup_table import BER_lookup_table
lookup_table = BER_lookup_table()


class MAMLDemodulator(Demodulator):
    def __init__(self, *,
                 stepsize_meta: float = 1e-3,
                 stepsize_inner: float = 1e-3,
                 first_order: bool = False,
                 inner_steps: int = 1,
                 inner_batch_size: int = 32,
                 outer_batch_size: int = 32,
                 SNR_db: float = 10.,
                 **kwargs
                 ):
        """
        For the MAML Demodulator, the `optimizer` only defines the optimizer
        for the outer meta-learning loop. The inner fast adaptation loop
        is hand-written SGD because PyTorch does not support differentiation
        through parameter updates.
        """
        super(MAMLDemodulator, self).__init__(**kwargs)
        self.stepsize_meta = torch.tensor(stepsize_meta, device=self.device)
        self.stepsize_inner = torch.tensor(stepsize_inner, device=self.device)
        self.first_order = first_order
        self.inner_steps = inner_steps
        self.inner_batch_size = inner_batch_size
        self.outer_batch_size = outer_batch_size
        self.SNR_db = SNR_db

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

    def update_maml(self, mods):
        model = self.model
        ntasks = len(mods)
        self.optimizer.zero_grad()
        outer_loss = torch.tensor(0., device=self.device)
        avg_ber = 0.
        # Calculate inner losses and take fast steps
        for it in range(ntasks):
            mod = mods[it]
            meta_loss, inner_loss, ber = self._update_inner(model, mod)
            outer_loss += meta_loss
            avg_ber += ber
        # Gradient update on mean task meta-loss
        avg_ber /= ntasks
        outer_loss /= ntasks
        outer_loss.backward()
        self.optimizer.step()
        return to_numpy(outer_loss), avg_ber

    def _update_inner(self, model, mod):
        params = model.meta_parameters()
        l1, l2, cross_entropy_weight = [self.lambda_l1, self.lambda_l2,
                                        self.cross_entropy_weight]
        criterion = nn.CrossEntropyLoss()
        inner_loss = 0.
        for ib in range(self.inner_steps):
            data_si = np.random.randint(0, 2 ** self.bits_per_symbol, size=[self.inner_batch_size])
            data_sb = to_tensor(integers_to_symbols(data_si, self.bits_per_symbol)).float().to(self.device)
            data_c = mod.modulate_tensor(data_sb)
            data_c_noisy = add_cartesian_awgn_tensor(data_c, SNR_db=self.SNR_db, device=self.device)
            logits = model.forward(data_c_noisy, params=params)
            target = torch.from_numpy(data_si).to(self.device)
            loss = cross_entropy_weight * torch.mean(criterion(logits, target))
            if l1 > 0:
                loss += l1 * model.l1_loss(params.values())
            if l2 > 0:
                loss += l2 * model.l2_loss(params.values())
            # Update params: performs one SGD step
            model.zero_grad()
            params = update_parameters(model, loss, params=params,
                                       step_size=self.stepsize_inner,
                                       first_order=self.first_order)
            inner_loss = loss
        # Calculate BER after inner_steps updates
        with torch.no_grad():
            data_si = np.random.randint(0, 2 ** self.bits_per_symbol, size=[self.inner_batch_size])
            data_sb = integers_to_symbols(data_si, self.bits_per_symbol)
            data_c = mod.modulate_tensor(to_tensor(data_sb).float().to(self.device))
            data_c_noisy = add_cartesian_awgn_tensor(data_c, SNR_db=self.SNR_db, device=self.device)
            labels_si_g = torch.argmax(model.forward(data_c_noisy, params=params), dim=-1)
            labels_sb_g = integers_to_symbols(to_numpy(labels_si_g).astype('int'), self.bits_per_symbol)
            ber = np.mean(np.abs(labels_sb_g - data_sb))

        # Calculate meta-loss for this task
        meta_data_si = np.random.randint(0, 2 ** self.bits_per_symbol, size=[self.outer_batch_size])
        meta_data_sb = to_tensor(integers_to_symbols(meta_data_si, self.bits_per_symbol)).float().to(self.device)
        meta_data_c = mod.modulate_tensor(meta_data_sb)
        meta_data_c_noisy = add_cartesian_awgn_tensor(meta_data_c, SNR_db=self.SNR_db, device=self.device)
        meta_logits = model.forward(meta_data_c_noisy, params=params)
        meta_target = torch.from_numpy(meta_data_si).to(self.device)
        meta_loss = cross_entropy_weight * torch.mean(criterion(meta_logits, meta_target))
        if l1 > 0:
            meta_loss += l1 * model.l1_loss(params.values)
        if l2 > 0:
            meta_loss += l2 * model.l2_loss(params.values)
        return meta_loss, to_numpy(inner_loss), ber

    def update_test(self, mod, nshots, batch_size=32, step_size=1e-2):
        model = self.model
        params = model.meta_parameters()
        l1, l2, cross_entropy_weight = [self.lambda_l1, self.lambda_l2,
                                        self.cross_entropy_weight]
        criterion = nn.CrossEntropyLoss()
        bers = []
        # Calculate BER before any updates
        with torch.no_grad():
            data_si = np.random.randint(0, 2 ** self.bits_per_symbol, size=[batch_size])
            data_sb = integers_to_symbols(data_si, self.bits_per_symbol)
            data_c = mod.modulate_tensor(to_tensor(data_sb).float().to(self.device))
            data_c_noisy = add_cartesian_awgn_tensor(data_c, SNR_db=self.SNR_db, device=self.device)
            labels_si_g = torch.argmax(model.forward(data_c_noisy, params=params), dim=-1)
            labels_sb_g = integers_to_symbols(to_numpy(labels_si_g).astype('int'), self.bits_per_symbol)
            bers.append(np.mean(np.abs(labels_sb_g - data_sb)))
        # Do updates
        for i in range(nshots):
            data_si = np.random.randint(0, 2 ** self.bits_per_symbol, size=[batch_size])
            data_sb = to_tensor(integers_to_symbols(data_si, self.bits_per_symbol)).float().to(self.device)
            data_c = mod.modulate_tensor(data_sb)
            data_c_noisy = add_cartesian_awgn_tensor(data_c, SNR_db=self.SNR_db, device=self.device)
            logits = model.forward(data_c_noisy, params=params)
            target = torch.from_numpy(data_si).to(self.device)
            loss = cross_entropy_weight * torch.mean(criterion(logits, target))
            if l1 > 0:
                loss += l1 * model.l1_loss(params.values())
            if l2 > 0:
                loss += l2 * model.l2_loss(params.values())
            # Update params: performs one SGD step
            model.zero_grad()
            params = update_parameters(model, loss, params=params,
                                       step_size=step_size,
                                       first_order=True)

            # Calculate BER after each update
            with torch.no_grad():
                data_si = np.random.randint(0, 2 ** self.bits_per_symbol, size=[batch_size])
                data_sb = integers_to_symbols(data_si, self.bits_per_symbol)
                data_c = mod.modulate_tensor(to_tensor(data_sb).float().to(self.device))
                data_c_noisy = add_cartesian_awgn_tensor(data_c, SNR_db=self.SNR_db, device=self.device)
                labels_si_g = torch.argmax(model.forward(data_c_noisy, params=params), dim=-1)
                labels_sb_g = integers_to_symbols(to_numpy(labels_si_g).astype('int'), self.bits_per_symbol)
                bers.append(np.mean(np.abs(labels_sb_g - data_sb)))
        return bers

    def update_test_dev(self, mod, nshots, train_batch_size=32, test_batch_size=32,
                        step_size=1e-2, test_bers=[1e-5, 1e-4, 1e-3, 1e-2]):
        test_snrs = [lookup_table.get_optimal_SNR_for_BER(test_ber, self.bits_per_symbol) for test_ber in test_bers]
        model = self.model
        l1, l2, cross_entropy_weight = [self.lambda_l1, self.lambda_l2,
                                        self.cross_entropy_weight]
        criterion = nn.CrossEntropyLoss()
        bers = {}
        for test_SNR_db in test_snrs:
            bers[test_SNR_db] = []
        params = model.meta_parameters()
        # Calculate BER before any updates
        with torch.no_grad():
            for test_SNR_db in test_snrs:
                data_si = np.random.randint(0, 2**self.bits_per_symbol, size=[test_batch_size])
                data_sb = to_tensor(integers_to_symbols(data_si, self.bits_per_symbol)).float().to(self.device)
                data_c = mod.modulate_tensor(data_sb)
                data_c_noisy = add_cartesian_awgn_tensor(data_c, SNR_db=test_SNR_db, device=self.device)

                labels_sb_g = self.demodulate(to_numpy(data_c_noisy))
                bit_loss = np.sum(np.abs(labels_sb_g - to_numpy(data_sb)), 1)
                bers[test_SNR_db].append(np.mean(bit_loss) / self.bits_per_symbol)
        # Do updates
        for i in range(nshots):
            data_si = np.random.randint(0, 2 ** self.bits_per_symbol, size=[train_batch_size])
            data_sb = to_tensor(integers_to_symbols(data_si, self.bits_per_symbol)).float().to(self.device)
            data_c = mod.modulate_tensor(data_sb)
            data_c_noisy = add_cartesian_awgn_tensor(data_c, SNR_db=self.SNR_db, device=self.device)
            logits = model.forward(data_c_noisy, params=params)
            target = torch.from_numpy(data_si).to(self.device)
            loss = cross_entropy_weight * torch.mean(criterion(logits, target))
            if l1 > 0:
                loss += l1 * model.l1_loss(params.values())
            if l2 > 0:
                loss += l2 * model.l2_loss(params.values())
            # Update params: performs one SGD step
            model.zero_grad()
            params = update_parameters(model, loss, params=params,
                                       step_size=step_size,
                                       first_order=True)

            # Calculate BER after each update
            with torch.no_grad():
                for test_SNR_db in test_snrs:
                    data_si = np.random.randint(0, 2 ** self.bits_per_symbol, size=[test_batch_size])
                    data_sb = integers_to_symbols(data_si, self.bits_per_symbol)
                    data_c = mod.modulate_tensor(to_tensor(data_sb).float().to(self.device))
                    data_c_noisy = add_cartesian_awgn_tensor(data_c, SNR_db=test_SNR_db, device=self.device)
                    labels_si_g = torch.argmax(model.forward(data_c_noisy, params=params), dim=-1)
                    labels_sb_g = integers_to_symbols(to_numpy(labels_si_g).astype('int'), self.bits_per_symbol)
                    bit_loss = np.sum(np.abs(labels_sb_g - data_sb), 1)
                    bers[test_SNR_db].append(np.mean(bit_loss) / self.bits_per_symbol)
        return bers

