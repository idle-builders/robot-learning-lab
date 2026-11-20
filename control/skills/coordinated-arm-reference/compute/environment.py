"""Reuse exact isolated wheel/source metadata operations and CUDA attestation."""
import argparse
from common import DATA, ENVIRONMENT, FORWARD, load

if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('stage-sources','bind-sources','attest','cuda-attest'))
    args=parser.parse_args()
    if args.action == 'cuda-attest':
        load(FORWARD+'/attest.py','coordinated_arm_retained_cuda_attestation')
    else:
        helper=load(ENVIRONMENT,'coordinated_arm_retained_environment')
        helper.CONFIG=DATA/'environment-config.json'
        getattr(helper,args.action.replace('-','_'))()
