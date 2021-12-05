import os.path
import os.path
from bisect import bisect_right

import torch


class SequentialLR(torch.optim.lr_scheduler.SequentialLR):
    def step(self):
        self.last_epoch += 1
        idx = bisect_right(self._milestones, self.last_epoch)
        if idx > 0 and self._milestones[idx - 1] == self.last_epoch:
            self._schedulers[idx].step(0)
            self._last_lr = self._schedulers[idx].get_last_lr()
        else:
            self._schedulers[idx].step()
            self._last_lr = self._schedulers[idx].get_last_lr()


def get_scheduler(optimizer, args):
    scheduler_name = args.scheduler.lower()
    warmup_scheduler_iters = args.lr_warmup_epochs * args.epoch_length
    main_scheduler_iters = args.num_steps - warmup_scheduler_iters
    if scheduler_name == "poly":
        main_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda t: (1 - t / main_scheduler_iters) ** args.lr_poly_decay,
        )
    elif scheduler_name == "cos":
        main_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=main_scheduler_iters,
            eta_min=args.min_lr,
        )
    elif scheduler_name == "linear":
        main_lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=args.lr_linear_start_factor,
            end_factor=args.lr_linear_end_factor,
            total_iters=main_scheduler_iters,
        )
    else:
        raise NotImplementedError

    scheduler = main_lr_scheduler
    if args.lr_warmup_epochs > 0:
        if args.lr_warmup_method == "linear":
            warmup_lr_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=args.lr_warmup_linear_start_factor,
                total_iters=warmup_scheduler_iters,
            )
        elif args.lr_warmup_method == "constant":
            warmup_lr_scheduler = torch.optim.lr_scheduler.ConstantLR(
                optimizer,
                factor=args.lr_warmup_constant_factor,
                total_iters=args.lr_warmup_epochs(args.epoch_length),
            )
        else:
            raise NotImplementedError
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_lr_scheduler, scheduler],
            milestones=[warmup_scheduler_iters],
        )
    return scheduler
