"""Reuse the reviewed isolated environment operations with this run's config."""
import argparse
from common import DATA, ENVIRONMENT, load

if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('action', choices=('stage-sources','bind-sources','attest'))
    args = parser.parse_args()
    helper = load(ENVIRONMENT, 'retained_isolated_environment')
    helper.CONFIG = DATA / 'environment-config.json'
    getattr(helper, args.action.replace('-', '_'))()
