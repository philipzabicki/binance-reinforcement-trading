import os
import pickle


def save_checkpoint(file_fullpath, iteration, unmatched_segments, all_results):
    checkpoint = {
        'iteration': iteration,
        'unmatched_segments': unmatched_segments,
        'all_results': all_results
    }
    with open(file_fullpath, 'wb') as f:
        pickle.dump(checkpoint, f)


def load_checkpoint(file_fullpath):
    if os.path.exists(file_fullpath):
        with open(file_fullpath, 'rb') as f:
            return pickle.load(f)
    else:
        return None
