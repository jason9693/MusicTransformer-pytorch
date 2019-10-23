import argparse


def get_argument_parser(description=None):
    parser = argparse.ArgumentParser(description)
    parser.add_argument("-m", "--model_dir", type=str, required=True,
            help="The directory for a trained model is saved.")
    parser.add_argument("-c", "--conf", dest="configs", default=[], nargs="*",
            help="A list of configuration items. "
                 "An item is a file path or a 'key=value' formatted string. "
                 "The type of a value is determined by applying int(), float(), and str() "
                 "to it sequencially.")
    return parser