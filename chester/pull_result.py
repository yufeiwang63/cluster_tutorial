import sys
import os
import argparse
from chester.config import SEUSS_HOME_FOLDER, SEUSS_PROJECT_NAME

sys.path.append('.')
from chester import config

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('host', type=str)
    parser.add_argument('folder', type=str)
    parser.add_argument('--dry', action='store_true', default=False)
    parser.add_argument('--bare', action='store_true', default=False)
    parser.add_argument('--img', action='store_true', default=False)
    parser.add_argument('--pkl', action='store_true', default=False)
    parser.add_argument('--pth', action='store_true', default=False)
    parser.add_argument('--gif', action='store_true', default=False)
    parser.add_argument('--newdatadir', action='store_true', default=False)
    args = parser.parse_args()

    args.folder = args.folder.rstrip('/')
    if args.folder.rfind('/') !=-1:
        local_dir = os.path.join('./data', args.host, args.folder[:args.folder.rfind('/')])
    else:
        local_dir = os.path.join('./data', args.host)
    dir_path = '/data/{}/{}/'.format(SEUSS_HOME_FOLDER, SEUSS_PROJECT_NAME)
    remote_data_dir = os.path.join(dir_path, 'data', 'local', args.folder)
    command = """rsync -avzh --delete --progress {host}:{remote_data_dir} {local_dir} --include '*best_model.pth'  """.format(host=args.host,
                                                                                                remote_data_dir=remote_data_dir,
                                                                                                local_dir=local_dir)
    if args.bare:
        command += """  --exclude '*checkpoint*' --exclude '*ckpt*' --exclude '*.pth'  --exclude '*tfevents*' --exclude '*.pt' --include '*.csv' --include '*.json' --delete"""
    if not args.img:
        command += """ --exclude '*.png' """
    if not args.gif:
        command += """ --exclude '*.gif' """
    if not args.pkl:
        command += """ --exclude '*.pkl'  """
    if args.pth:
        command += """ --include '*best_model.pth'  """
        command += """ --include '*.pth'  """
    os.system(command)
