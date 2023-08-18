from absl import app
from absl import flags

FLAGS = flags.FLAGS
flags.DEFINE_string('config', None, 'Path to the experiment configuration file.')


def main(argv):
    del argv
    print(FLAGS.config)


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--config", default=None)
    # parser.add_argument("--checkpoint", default=None)
    # args = parser.parse_args()
    # config_file = Path(args.config)

    # main(config_file, args.checkpoint)
    app.run(main)
