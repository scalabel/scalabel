"""
Check line lengh of files
"""
import glob
import re
from argparse import ArgumentParser
from os.path import join
from colorama import Fore


def find_files(base_directory, file_ext):
    """ traverses directories to find file_type files """
    paths = glob.glob(join(base_directory, '**/*{}'.format(file_ext)),
                      recursive=True)
    return paths


def count_chars(filepath, compiled_regex, tab_size, max_length, verbose):
    """ counts characters in given file, ignoring select lines based on
    regex, to check if the length exceeds the max length allowed """
    errors = 0
    with open(filepath, 'r') as f:
        for i, line in enumerate(f):
            if compiled_regex.match(line):
                continue
            line = line.replace('\t', " "*tab_size)
            if line[-1] == '\n':
                line = line[:-1]
            line_length = len(line)
            if line_length > max_length:
                errors += 1
                print("{}: {}: The line has {} characters".format(
                    filepath, i+1, line_length))
                if verbose:
                    print(Fore.RED + ">    {}".format(line) + Fore.WHITE)
    return errors


def lll(base_directory, file_ext,
        regex, tab_size, max_length, verbose):
    """ handles lll checking """
    compiled_regex = re.compile(regex)
    files = find_files(base_directory, file_ext)
    errors = 0
    for filepath in files:
        errors += count_chars(
            filepath, compiled_regex, tab_size, max_length, verbose)
    return errors


def main():
    """ main """
    parser = ArgumentParser()
    parser.add_argument("-d", "--directory",
                        dest="directory")
    parser.add_argument("-e", "--exceptions", dest="regex",
                        default='.*`json:".*" yaml:".*"`')
    parser.add_argument("-m", "--max_length", dest="max_length", default=80,
                        type=int)
    parser.add_argument("-t", "--tab_size", dest="tab_size", default=4,
                        type=int)
    parser.add_argument("-f", "--file_ext", dest="file_ext", default=".go")
    parser.add_argument("-v", "--verbose", dest="verbose", action='store_true')
    args = parser.parse_args()
    errors = lll(args.directory, args.file_ext, args.regex,
                 args.tab_size, args.max_length, args.verbose)
    if errors == 0:
        print('All good!')
    else:
        print('\nFound {} errors'.format(errors))
    exit(errors > 0)


if __name__ == "__main__":
    main()
