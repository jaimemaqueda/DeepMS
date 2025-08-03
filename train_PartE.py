from collections import OrderedDict
from tqdm import tqdm
from dataset.datasetPartE import get_dataloader
from config import ConfigPartE
from utils import cycle
from trainer import TrainerPartE


def main():
    # create experiment cfg containing all hyperparameters
    cfg = ConfigPartE('train')

    # create network and training agent
    tr_agent = TrainerPartE(cfg)

    # load from checkpoint if provided
    if cfg.cont:
        tr_agent.load_ckpt(cfg.ckpt)

    # create dataloader
    train_loader = get_dataloader('train', cfg)
    val_loader = get_dataloader('validation', cfg)
    val_loader_all = get_dataloader('validation', cfg)
    val_loader = cycle(val_loader)

    # start training
    clock = tr_agent.clock

    # show number of parameters
    tr_agent.show_parameters()

    for e in range(clock.epoch, cfg.nr_epochs):
        # begin iteration
        pbar = tqdm(train_loader)
        for b, data in enumerate(pbar):
            # train step
            _, losses = tr_agent.train_func(data)

            pbar.set_description("EPOCH[{}][{}]".format(e, b))
            pbar.set_postfix(OrderedDict({k: v.item() for k, v in losses.items()}))

            # validation step
            if clock.step % cfg.val_frequency == 0:
                data = next(val_loader)
                _, losses = tr_agent.val_func(data)

            clock.tick()

            tr_agent.update_learning_rate()

        if clock.epoch % 1 == 0:
            tr_agent.evaluate(val_loader_all)

        clock.tock()

        if clock.epoch % cfg.save_frequency == 0:
            tr_agent.save_ckpt()

        tr_agent.save_ckpt('latest')


if __name__ == '__main__':
    main()

