from utils.setup import logger


def readable_time(time_difference):
    hours = time_difference // 3600
    minutes = (time_difference // 60) % 60
    seconds = time_difference % 60

    return hours, minutes, seconds


def log_training(
    epoch,
    aggr_metrics_train,
    tb_writer,
    start_epoch,
    total_epoch_time,
    epoch_start_time,
    epoch_end_time,
    num_batches,
    num_samples,
    aggr_imgs_train={},
):
    """
    Adapted from https://github.com/gzerveas/mvts_transformer
    """

    print()
    epoch_runtime = epoch_end_time - epoch_start_time
    print_str = "Epoch {} Training Summary: ".format(epoch)
    for k, v in aggr_metrics_train.items():
        tb_writer.add_scalar("{}/train".format(k), v, epoch)
        print_str += "{}: {:8f} | ".format(k, v)
    logger.info(print_str)
    logger.info(
        "Epoch runtime: {} hours, {} minutes, {} seconds\n".format(
            *readable_time(epoch_runtime)
        )
    )

    for k, v in aggr_imgs_train.items():
        tb_writer.add_figure("{}/train".format(k), v, epoch)

    total_epoch_time += epoch_runtime
    avg_epoch_time = total_epoch_time / (epoch - start_epoch)
    avg_batch_time = avg_epoch_time / num_batches
    avg_sample_time = avg_epoch_time / num_samples

    logger.info(
        "Avg epoch training time: {} hours, {} minutes, {} seconds".format(
            *readable_time(avg_epoch_time)
        )
    )
    logger.info("Avg batch training time: {} seconds".format(avg_batch_time))
    logger.info("Avg sample training time: {} seconds".format(avg_sample_time))


class AverageMeter(object):
    """
    Computes and stores the average and current value
    Adapted from https://github.com/facebookresearch/ijepa
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.max = float("-inf")
        self.min = float("inf")
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        try:
            self.max = max(val, self.max)
            self.min = min(val, self.min)
        except Exception:
            pass
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
