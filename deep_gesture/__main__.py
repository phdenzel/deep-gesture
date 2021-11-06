"""
deep_gesture.__main__

@author: phdenzel
"""
import deep_gesture

__all__ = ['main']


def arg_parse():
    from argparse import ArgumentParser, RawTextHelpFormatter
    p = ArgumentParser(prog='deep_gesture', #description=__doc__,
                       formatter_class=RawTextHelpFormatter)

    # General flags
    p.add_argument("-d", "--device", dest="device_id", metavar="<device>", type=int,
                   default=deep_gesture.device_id,
                   help="Integer ID of the video capture device (commonly 0, 1, or 2 etc.)."
                   )
    p.add_argument("-f", "--file",
                   dest="video_file", metavar="<filename>", type=str,
                   help="Pathname of the pre-recorded gesture video file; triggers file mode."
                   )
    p.add_argument("-t", "--train", "--train-mode",
                   dest="train_mode", action="store_true", default=False,
                   help="Run deep_gesture in train mode."
                   )
    p.add_argument("--gestures",
                   dest="gestures", metavar="<gesture1>", nargs='+',
                   default=deep_gesture.gestures,
                   help="")
    p.add_argument("--sequences",
                   dest="N_training_sequences", metavar="<N_training_sequences>", type=int,
                   default=deep_gesture.N_training_sequences,
                   help="Set number of training sequences. Takes effect in train mode."
                   )
    p.add_argument("-l", "--length", "--sequence-length",
                   dest="sequence_length", metavar="<sequence_length>", type=int,
                   default=deep_gesture.sequence_length,
                   help="Set number of training sequences. Takes effect in train mode."
                   )
    p.add_argument("--test", "--test-mode", dest="test_mode", action="store_true",
                   help="Run program in testing mode.", default=False
                   )
    p.add_argument("-v", "--verbose", dest="verbose", metavar="<level>", type=int,
                   help="Define level of verbosity")

    args = p.parse_args()
    return p, args


def overwrite_dg_vars(args):
    """
    Args:
        args <Namespace>
    """
    deep_gesture.device_id = args.device_id
    deep_gesture.N_training_sequences = args.N_training_sequences
    deep_gesture.sequence_length = args.sequence_length
    deep_gesture.gestures = args.gestures


def main():

    parser, args = arg_parse()
    overwrite_dg_vars(args)

    # run test suite
    if args.test_mode:
        from test import main
    # run in train mode
    elif args.train_mode:
        from deep_gesture.record import train as main
    # run in file mode
    elif args.video_file:
        from deep_gesture.process import video as main
    # run in streaming mode
    else:
        from deep_gesture.process import stream as main

    main()
