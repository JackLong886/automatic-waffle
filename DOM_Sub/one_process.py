import os
import argparse
import subprocess

parser = argparse.ArgumentParser('cloud removal for remote sensing images')
parser.add_argument('--json_path_det', type=str, default=r'parameters_det.json')
parser.add_argument('--json_path_rem', type=str, default=r'parameters_rem.json')
parser.add_argument('--json_path_mos', type=str, default=r'parameters_mos.json')
opt = parser.parse_args()


os.system('"python cloud_detec.py --json_path {} --one_flag {}"'.format(opt.json_path_det, True))
os.system('"python cloud_removal0213_2.py --json_path {} --one_flag {}"'.format(opt.json_path_rem, True))
os.system('"python mosaic0219.py --json_path {} --one_flag {}"'.format(opt.json_path_mos, True))
