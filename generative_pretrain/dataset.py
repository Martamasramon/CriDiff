from PIL import Image
from torchvision        import transforms as T, utils
from torch.utils.data   import Dataset
import pandas as pd
import numpy  as np

class MyDataset(Dataset):
    def __init__(
        self,
        img_path,
        image_size,
        exts                    = ['jpg', 'jpeg', 'png', 'tiff'],
        augment_horizontal_flip = False,
        use_histo               = False, 
        use_t2w                 = False, 
        is_pretrain             = True, 
        is_train                = True
    ):
        super().__init__()
        root   = 'pretrain' if is_pretrain else 'finetune'
        suffix = 'train'    if is_train    else 'test'
        self.img_path   = img_path
        self.img_dict   = pd.read_csv(f'../../Dataset_preparation/{root}_{suffix}.csv')
        self.image_size = image_size
        self.use_histo  = use_histo
        self.use_t2w    = use_t2w
        
        self.transform = T.Compose([
            T.CenterCrop(image_size),
            T.Resize(image_size//2),
            T.Resize(image_size),
            T.RandomHorizontalFlip() if augment_horizontal_flip else nn.Identity(),
            T.ToTensor(),
        ])
        
        self.label_transform = T.Compose([
            T.CenterCrop(image_size),
            T.RandomHorizontalFlip() if augment_horizontal_flip else nn.Identity(),
            T.ToTensor(), 
        ])

    def __len__(self):
        return len(self.img_dict)

    def __getitem__(self, idx):
        item   = self.img_dict.iloc[idx]
        img    = Image.open(self.img_path + item["SID"]).convert('L')
        #### Add normalisation of image by a single value, don't allow ToTensor to do min-max!
        sample = {}

        if self.use_histo:
            embed_histo = torch.load(item["Histo"],weights_only=True) # torch.load(map_location=torch.device("cpu")
            sample['Histo'] = embed_histo
        
        if self.use_t2w:
            embed_t2w = torch.load(item["T2W"],weights_only=True) # torch.load(map_location=torch.device("cpu")
            sample['T2W'] = embed_t2w

        sample['Image'] = self.transform(img)
        sample['Label'] = self.label_transform(img)
        
        return sample
    