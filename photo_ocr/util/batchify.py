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



