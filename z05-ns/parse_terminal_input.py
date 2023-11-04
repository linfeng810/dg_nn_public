import argparse

parser = argparse.ArgumentParser(
    prog='nssolver',
    description='solve ns eq',
    epilog='--end of help--'
)

parser.add_argument('-f', '--filename', required=False, dest='filename')
parser.add_argument('--timestep', required=False, type=float, dest='dt')

args = parser.parse_args()
