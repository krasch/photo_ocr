from typing import List, Tuple, Callable
from functools import partial
from itertools import islice

from torch.utils.data import DataLoader

from photo_ocr.util.cuda import DEVICE


def run_in_batches(model, images, batch_size):
    # this data loader takes care of any batch-ification etc
    # maybe this is too much overhead? but it sure is convenient
    # check back here and maybe refactor to native python code if running into trouble with this
    batches = DataLoader(images, batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=False)

    for batch in batches:
        batch = batch.to(DEVICE)

        # run the model on the batch
        predictions = model(batch).cpu().data.numpy()

        # caller is not interested in a list of predictions per batch, but in a flat list of predictions per image
        # -> yield each prediction individually
        for prediction in predictions:
            yield prediction


def flatten(list_of_lists: List[List]) -> Tuple[List, Callable]:
    """
    Flatten a list of lists, while keeping the necessary information to reconstruct it later on.
    Returns the flat list +  a function that can be called to reconstruct a flat list (same or different one)
    with the same nested structure as the original list_of_lists
    :param list_of_lists:
    :return:
    """
    # take note of the number of entries for each of the inner lists
    # this will allow us to reconstruct the nested structure later on
    inner_list_lengths = [len(inner_list) for inner_list in list_of_lists]

    # flatten the list
    flat_list = [item for inner_list in list_of_lists for item in inner_list ]

    # can call this function to reconstruct the nested structure
    def reconstruct(another_flat_list):
        reconstructed_list_of_lists = []

        index = 0
        for length in inner_list_lengths:
            reconstructed_list_of_lists.append(another_flat_list[index: index+length])
            index += length

        return reconstructed_list_of_lists

    return flat_list, reconstruct


