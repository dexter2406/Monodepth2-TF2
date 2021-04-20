from absl import app
from new_trainer_v1 import Trainer
from utils import check_options
from options import get_options


def start_train(FLAGS):
    trainer = Trainer(FLAGS)
    trainer.train()


def main(_):
    options = get_options()
    check_options(options, debug=True)
    start_train(options)


if __name__ == '__main__':
    app.run(main)
