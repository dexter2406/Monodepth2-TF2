from new_trainer_v1 import Trainer
from utils import check_options
from options import get_options
from absl import app
from src.eval_depth import evaluate_depth
from src.eval_pose import evalualte_pose


def main(_):
    options = get_options()
    if options.run_mode == 'train':
        check_options(options)
        start_train(options)

    elif options.run_mode == 'eval_depth':
        start_eval_depth(options)

    elif options.run_mode == 'eval_pose':
        start_eval_pose(options)

    else:
        raise NotImplementedError('must choose from [\'trian\', \'eval_depth\', \'eval_pose\']')


def print_args(opts):
    print(opts.batch_size)


def start_train(options):
    trainer = Trainer(options)
    trainer.train()


def start_eval_depth(options):
    evaluate_depth(options)


def start_eval_pose(options):
    evalualte_pose(options)


if __name__ == '__main__':
    app.run(main)
