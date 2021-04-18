import argparse
import numpy as np

_mandatory_params = {'state_probabilities', 'state_weights', 'cis_dd_power', 'trans_dd'}
_allowed_params = _mandatory_params | {'cis_lengths'}

def save(filename: str, parameters: dict, metadata: dict={}) -> None:
    """
    Save model parameters and metadata
    """
    if parameters.keys() & _mandatory_params != _mandatory_params:
        raise ValueError(f"All of these parameters are mandatory: {_mandatory_params}")
    if parameters.keys() - _allowed_params:
        raise ValueError(f"Only these parameters are allowed: {_allowed_params}")
    _parameters = { 'cis_lengths': None,  **parameters }
    np.savez_compressed(filename, parameters=_parameters, metadata=metadata)

def load(filename: str) -> dict:
    """
    Load both parameters and metadata from a datafile
    """
    obj = np.load(filename, allow_pickle=True)
    p = obj['parameters'][()]
    try:
        m = obj['metadata'][()]
    except KeyError:
        m = {}
    assert p.keys() & _mandatory_params == _mandatory_params, f"Bad file, missing one of: {_mandatory_params}"
    assert p.keys() - _allowed_params == set(), f"Bad file, has extra parameters beyond: {_allowed_params}"

    p.setdefault('cis_lengths', None)

    return dict(parameters=p, metadata=m)

def load_params(filename: str) -> dict:
    """
    Load parameters from a datafile
    """
    return load(filename)['parameters']

def load_metadata(filename: str) -> dict:
    """
    Load metadata from a datafile
    """
    return load(filename)['metadata']

def parse_keyvalue(s):
    d = {}
    if s:
        args_list = s.split(',')
        for a in args_list:
            k, _, raw_v = a.partition('=')
            try:
                v = int(raw_v)
            except ValueError:
                try:
                    v = float(raw_v)
                except ValueError:
                    v = raw_v
            d[k] = v
    return d

def main():
    parser = argparse.ArgumentParser(description = 'Work with gc datafiles')
    parser.add_argument('-f', help='filename', dest='filename', type=str, required=True)
    parser.add_argument('-m', help='Comma separated key=value pairs of metadata to add', dest='metadata', type=str, required=False)
    parser.add_argument('--dry-run', help="Don't change the original file, just print the result", dest='dry_run', action='store_true', required=False)
    parser.add_argument('--no-print', help="Don't print the contens of the file", dest='verbose', action='store_false', required=False)
    parser.add_argument('--output', help="File to write changed object if -m is given. Defaults to input file", dest='output_filename', default=None, required=False)
    args = parser.parse_args()

    print("Reading old fit from file:", args.filename)
    fit = load(args.filename)
    if args.verbose:
        print("** Parameters info:")
        print("state probabilities shape:", fit['parameters']['state_probabilities'].shape)
        print("state weights:")
        print(fit['parameters']['state_weights'])
        print("cis dd power:", fit['parameters']['cis_dd_power'])
        print("trans dd:", fit['parameters']['trans_dd'])
        print("cis_lengths:", fit['parameters']['cis_lengths'])
        print("** Metadata:")
        for k,v in fit['metadata'].items():
            print(f"{k}: {v}")

    if not args.metadata:
        return

    new_metadata = parse_keyvalue(args.metadata)
    fit['metadata'].update(new_metadata)
    if args.verbose:
        print("** Added the following metadata attributes:")
        for k,v in new_metadata.items():
            print(f"{k}: {v}")
    if args.dry_run:
        return
    output_file = args.output_filename or args.filename
    print(f'Saving into {output_file}')
    save(output_file, **fit)
    print(f'Done.')

if __name__ == '__main__':
    main()

