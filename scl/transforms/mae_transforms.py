from torchvision import transforms
from .randaug import RandAugment
from PIL import Image
import PIL


def clip_transform(size):
    trs = transforms.Compose([
            transforms.RandomResizedCrop(size, scale=(0.8, 1.0), interpolation=3),  # 3 is bicubic
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])])
    return trs

def clip_transform_randaug(size):
    trs = transforms.Compose([
            transforms.RandomResizedCrop(size, scale=(0.9, 1.0), interpolation=3),  # 3 is bicubic
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])])

    trs.transforms.insert(0, RandAugment(2, 9))
    return trs

def clip_transform_test(size):
    t = []
    t.append(
        transforms.Resize((size, size), interpolation=PIL.Image.BICUBIC),  # to maintain same ratio w.r.t. 224 images
    )
    t.append(transforms.CenterCrop(size))
    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]))
    return transforms.Compose(t)

