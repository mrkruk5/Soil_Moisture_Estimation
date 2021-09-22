import argparse
import json


def add_configs():
    #
    # Add configuration capabilities to a file.
    #
    parser = argparse.ArgumentParser(description='Configuration arguments provided at run time from the CLI')
    parser.add_argument(
        '-c',
        '--config_file',
        dest='config_file',
        type=str,
        default=None,
        help='Example: python3 config_example.py -c config_file.json'
    )
    args = parser.parse_args()
    if args.config_file is not None:
        file_name = args.config_file.split('/')[-1].split('.')[0]
        if '.json' in args.config_file:
            with open(args.config_file, 'r') as f:
                config = json.load(f)
                config = argparse.Namespace(**config)
                # TODO: Look into making this a named tuple instead of a Namespace.
    return config, file_name
