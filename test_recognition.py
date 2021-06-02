import argparse



import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
from PIL import Image

from photo_ocr.recognition.dataset import TextRecognitionDataset
from photo_ocr.recognition.model.model import None_VGG_None_CTC, None_ResNet_None_CTC, TPS_ResNet_BiLSTM_Attn, TPS_ResNet_BiLSTM_Attn_case_sensitive, TPS_ResNet_BiLSTM_CTC
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from refactoraid import refactoraid
#refactoraid.set_collect_references()

from pathlib import Path


def load_model(opt):

    # RGB not supported
    input_shape = opt["imgH"], opt["imgW"], 1

    model = opt["model"](pretrained=True, progress=True, image_shape=input_shape)
    model.eval()

    return model


def load_data(opt):
    root = Path(opt["image_folder"])
    files = list(root.glob("*.jpg")) + list(root.glob("*.jpeg")) + list(root.glob("*.png"))
    images = [Image.open(f).convert('L') for f in files]

    demo_data = TextRecognitionDataset(images, keep_ratio=opt["PAD"])

    if opt["PAD"]:
        demo_loader = torch.utils.data.DataLoader(
            demo_data, batch_size=2,
            shuffle=False,
            num_workers=1, pin_memory=True)
    else:
        demo_loader = torch.utils.data.DataLoader(
            demo_data, batch_size=2,
            shuffle=False,
            num_workers=1,
            pin_memory=True)

    return files, demo_loader


def predict_batch(batch, model, model_name, pad):
    image_tensors, image_path_list = batch

    image = image_tensors.to(device)

    preds = model(image)

    for img_name, pred in zip(image_path_list, preds.cpu().data.numpy()):
        refactoraid.set_key(model_name + "_" + Path(img_name).name+"_"+str(pad))

        pred_str, confidence_score = model.decode(pred)

        refactoraid.add("pred", pred)
        refactoraid.add("pred_str", pred_str)
        refactoraid.add("conf", confidence_score)


def predict_all(model, demo_loader):
    all_preds = []
    with torch.no_grad():
        for batch in demo_loader:
            batch = batch.to(device)
            preds = model(batch)
            all_preds.append(preds)
    return torch.cat(all_preds, dim=0)



def demo(opt):
    model_name = (opt["model"].__name__ +".pth").replace("_", "-")

    model = load_model(opt)
    files, demo_loader = load_data(opt)
    preds = predict_all(model, demo_loader)

    for img_name, pred in zip(files, preds.cpu().data.numpy()):
        refactoraid.set_key(model_name + "_" + Path(img_name).name + "_" + str(opt["PAD"]))

        pred_str, confidence_score = model.decode(pred)

        refactoraid.add("pred", pred)
        refactoraid.add("pred_str", pred_str)
        refactoraid.add("conf", confidence_score)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_folder', help='path to image_folder which contains text images', default="/home/kat/Projekte/Research/photo_ocr/data/recognition")
    """ Data processing """
    parser.add_argument('--imgH', type=int, default=32, help='the height of the input image')
    parser.add_argument('--imgW', type=int, default=100, help='the width of the input image')
    parser.add_argument('--PAD', action='store_true', help='whether to keep ratio then pad for image resize')

    opt = dict(parser.parse_args().__dict__)

    models = [None_ResNet_None_CTC,
              None_VGG_None_CTC,
              TPS_ResNet_BiLSTM_Attn,
              TPS_ResNet_BiLSTM_Attn_case_sensitive,
              TPS_ResNet_BiLSTM_CTC]

    cudnn.benchmark = True
    cudnn.deterministic = True

    for model in models:
        opt["model"] = model
        demo(opt)

    opt["PAD"] = True
    for model in models:
        opt["model"] = model
        demo(opt)

    refactoraid.check()