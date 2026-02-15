from tqdm import tqdm

LEAVE_TQDM_BAR = True

def _progress_range(iterable, desc= None):
    return tqdm(
        iterable,
        leave=LEAVE_TQDM_BAR,
        desc=desc
    )
