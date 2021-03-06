# Modified from https://github.com/KaiyangZhou/deep-person-reid/blob/master/torchreid/data/datasets/dataset.py  # noqa
# to support unsupervised features

import copy
import os.path as osp
import tarfile
import zipfile
import numpy as np
import torchvision.transforms as T
import cv2

from torchvision.utils import save_image as save_image
from ...utils import bcolors
from ...utils.dist_utils import get_dist_info, synchronize
from ...utils.file_utils import download_url, download_url_from_gd, mkdir_if_missing
from ..utils.data_utils import read_image
from PIL import Image
from os import listdir as osl
class Dataset(object):
    """An abstract class representing a Dataset.

    This is the base class for ``ImageDataset``.

    Args:
        data (list): contains tuples of (img_path(s), pid, camid).
        mode (str): 'train', 'val', 'trainval', 'query' or 'gallery'.
        transform: transform function.
        verbose (bool): show information.
    """

    def __init__(
        self, data, mode, transform=None, verbose=True, sort=True, **kwargs,
    ):
        self.data = data
        self.transform = transform
        self.mode = mode
        self.verbose = verbose

        self.num_pids, self.num_cams = self.parse_data(self.data)

        if sort:
            self.data = sorted(self.data)

        if self.verbose:
            self.show_summary()

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        return len(self.data)

    def __add__(self, other):
        raise NotImplementedError

    def parse_data(self, data):
        """Parses data list and returns the number of person IDs
        and the number of camera views.
        Args:
            data (list): contains tuples of (img_path(s), pid, camid)
        """
        pids = set()
        cams = set()
        for _, pid, camid in data:
            pids.add(pid)
            cams.add(camid)
        return len(pids), len(cams)

    def show_summary(self):
        """Shows dataset statistics."""
        pass

    def download_dataset(self, dataset_dir, dataset_url, dataset_url_gid=None):
        """Downloads and extracts dataset.
        Args:
            dataset_dir (str): dataset directory.
            dataset_url (str): url to download dataset.
        """
        if osp.exists(dataset_dir):
            return

        if dataset_url is None:
            raise RuntimeError(
                "{} dataset needs to be manually "
                "prepared, please download this dataset "
                "under the folder of {}".format(self.__class__.__name__, dataset_dir)
            )

        rank, _, _ = get_dist_info()

        if rank == 0:

            print('Creating directory "{}"'.format(dataset_dir))
            mkdir_if_missing(dataset_dir)
            fpath = osp.join(dataset_dir, osp.basename(dataset_url))

            print(
                'Downloading {} dataset to "{}"'.format(
                    self.__class__.__name__, dataset_dir
                )
            )

            if dataset_url_gid is not None:
                download_url_from_gd(dataset_url_gid, fpath)
            else:
                download_url(dataset_url, fpath)

            print('Extracting "{}"'.format(fpath))
            try:
                tar = tarfile.open(fpath)
                tar.extractall(path=dataset_dir)
                tar.close()
            except Exception:
                zip_ref = zipfile.ZipFile(fpath, "r")
                zip_ref.extractall(dataset_dir)
                zip_ref.close()

            print("{} dataset is ready".format(self.__class__.__name__))

        synchronize()

    def check_before_run(self, required_files):
        """Checks if required files exist before going deeper.
        Args:
            required_files (str or list): string file name(s).
        """
        if isinstance(required_files, str):
            required_files = [required_files]

        for fpath in required_files:
            if not osp.exists(fpath):
                raise RuntimeError('"{}" is not found'.format(fpath))

    def __repr__(self):
        msg = (
            "  -----------------------------------------------------\n"
            "  dataset                 | # ids | # items | # cameras\n"
            "  -----------------------------------------------------\n"
            "  {:20s}    | {:5d} | {:7d} | {:9d}\n"
            "  -----------------------------------------------------\n".format(
                self.__class__.__name__ + "-" + self.mode,
                self.num_pids,
                len(self.data),
                self.num_cams,
            )
        )

        return msg


class ImageDataset(Dataset):
    """A base class representing ImageDataset.

        All other image datasets should subclass it.
        ``_get_single_item`` returns an image given index.
        It will return (``img``, ``img_path``, ``pid``, ``camid``, ``index``)
        where ``img`` has shape (channel, height, width). As a result,
        data in each batch has shape (batch_size, channel, height, width).
    """

    def __init__(self, data, mode, cfg=None ,data_idx=0,pseudo_labels=None, **kwargs):
        if "verbose" not in kwargs.keys():
            kwargs["verbose"] = False if (pseudo_labels is not None) else True
        super(ImageDataset, self).__init__(data, mode, **kwargs)
        # "all_data" stores the original data list
        # "data" stores the pseudo-labeled data list
        self.data_idx = data_idx
        self.cfg = cfg
        self.all_data = copy.deepcopy(self.data)
        self._set_mask_params(cfg,mode)
        if pseudo_labels is not None:
            self.renew_labels(pseudo_labels)
        self._mask_method = {
            0:self._get_fg_img,
            1:self._get_rand_bg,
        }

    def __getitem__(self, indices):
        if isinstance(indices, (tuple, list)):
            return [self._get_single_item(index) for index in indices]
        return self._get_single_item(indices)

    def _get_single_item(self, index):
        img_path, pid, camid = self.data[index]
        if self.mode == 'trainval':
            img_path = img_path.replace(str(self.cfg.DATA_ROOT),str(self.cfg.DATA_ROOT_REPLACE))
        img = read_image(img_path)
        if self.transform is not None:
            img = self.transform(img)  
        data_dict = {
            "img": img,
            "path": img_path,
            "id": pid,
            "cid": camid,
            "ind": index,
        }      

        # modified
        # fg,bg can be used to calculate loss

        if self.mask:
            mask_path = img_path.replace(self.original_dir,self.mask_dir)
            img,fg,bg = self._get_masked_img(mask_path,img,self.mask_class,self.is_hard,self.is_lip)
            if np.random.uniform(0,1)>self.repalce_proportion:
                data_dict['img']=img
            data_dict['fg']=fg
            data_dict['bg']=bg
        return data_dict
    #modified
    def _get_masked_img(self,mask_path,img,class_name,is_hard,is_lip):
        '''
        img: original image
        there are 3 different classes, [
            only_foreground,
            foreground+random_backgroud,
            foreground+target_background
            ]
        is_hard : soft mask or hard mask
            soft: 0 - 1
            hard: 0 or 1
        '''
        h,w = img.shape[1],img.shape[2]
        #resize mask and totensor
        mask = Image.fromarray(np.load(mask_path.replace('jpg','npy')))
        transform = T.Compose([
            T.Resize((h,w)),T.ToTensor()
        ])
        mask = transform(mask)
        if is_hard:
        # lip mask should include 0 or 1
            if is_lip:
                mask[mask>0]=1
            else:
                mask[mask<0.5]=0
                mask[mask>=0.5]=1
        if class_name!=2:
            return self._mask_method[class_name](mask,img)
        else:
            return img,mask,1-mask

    def _get_fg_img(self,mask,img):
        img = img*mask
        bg = 1-mask
        return img,mask,bg

    def _get_rand_bg(self,mask,img):
        # chosen_bg = np.random.randint(0,len(bg_paths))
        bg_candidates = []
        rand_rates,chosen_indexes = self._gen_rand_rate_and_path(self.mix_num)
        h,w = img.shape[1],img.shape[2]

        #read bgs
        for chosen_index in chosen_indexes:
            img_bg = cv2.imread(osp.join(self.rand_bg_dir,self.bg_paths[chosen_index]))
            img_bg = cv2.resize(img_bg,(w,h))
            bg_candidates.append(img_bg)
        # init bg_img
        # bg_candidates[0]+bg_candidates[1]
        bg_img = np.zeros_like(bg_candidates[0],dtype=np.float64)

        # stack imgs
        for index,rate in enumerate(rand_rates):
            temp = rate*bg_candidates[index]
            bg_img += temp
        bg_img = np.uint8(bg_img.clip(min=0, max=255))
        bg_img = Image.fromarray(cv2.cvtColor(bg_img,cv2.COLOR_BGR2RGB))
        transform = T.Compose([
            T.Resize((h,w)),T.ToTensor()
        ])
        bg_img = self.transform(bg_img)
        # bg_img = transform(bg_img)
        bg = 1-mask
        fg = img*mask
        bg_img = bg_img*bg
        img = bg_img + fg
        return img,mask,bg
    def _set_mask_params(self,cfg,mode):
        if mode=='query' or mode=='gallery' or mode=='query+gallery':
            print(mode)
            self.mask = cfg.DATA.TEST.mask[self.data_idx]
            print('is_mask:',self.mask)
            if self.mask:
                # self.replace_proportion to control the proportion of using fg image or rand_bg image,
                # if you set it as 1 ,it means use original image ,set it as 0 means use all fg image or rand_bg image
                self.mix_num = cfg.DATA.TEST.mix_num
                self.repalce_proportion=cfg.DATA.TEST.repalce_proportion
                self.mask_class = cfg.DATA.TEST.mask_class[self.data_idx]
                self.is_hard = cfg.DATA.TEST.is_hard[self.data_idx]
                self.is_lip = cfg.DATA.TEST.is_lip[self.data_idx]
                self.mask_dir = cfg.DATA.TEST.mask_dir[self.data_idx]
                self.original_dir = cfg.DATA.TEST.original_dir[self.data_idx]
                print('is_hard:',self.is_hard,'is_lip:',self.is_lip)
                self.rand_bg_dir = self.cfg.DATA.TEST.rand_bg_dir[self.data_idx]
                self.bg_paths = osl(self.rand_bg_dir)
        else:
            self.mask = cfg.DATA.TRAIN.mask[self.data_idx]
            print('is_mask:',self.mask)

            if self.mask:
                self.mix_num = cfg.DATA.TRAIN.mix_num
                # self.replace_proportion to control the proportion of using fg image or rand_bg image,
                # if you set it as 1 ,it means use original image ,set it as 0 means use all fg image or rand_bg image
                self.repalce_proportion=cfg.DATA.TRAIN.repalce_proportion
                self.mask_class = cfg.DATA.TRAIN.mask_class[self.data_idx]
                self.is_hard = cfg.DATA.TRAIN.is_hard[self.data_idx]
                self.is_lip = cfg.DATA.TRAIN.is_lip[self.data_idx]
                self.mask_dir = cfg.DATA.TRAIN.mask_dir[self.data_idx]
                self.original_dir = cfg.DATA.TRAIN.original_dir[self.data_idx]
                print('is_hard:',self.is_hard,'is_lip:',self.is_lip)
                self.rand_bg_dir = self.cfg.DATA.TRAIN.rand_bg_dir[self.data_idx]
                self.bg_paths = osl(self.rand_bg_dir)
    def _gen_rand_rate_and_path(self,mix_num=3):
        '''mix_num:
           number of mix up bgs,defalut is 3,if you only want one bg, set it as 1
           if we want  3 bgs, we hope these 3 bgs have similar weights, so the center of 
           rand number is 1/mix_num,and the range of rand we set as 1/mix_num +- 0.2
        '''
        center = 1/mix_num
        rand_rates = []
        chosen_indexes = []
        if mix_num==1:
            rand_rates = [1,]
        elif mix_num==2:
            a = np.random.uniform(center-0.2,center+0.2)
            rand_rates = [a,1-a]
        elif mix_num==3:
            a = np.random.uniform(center-0.2,center+0.2)
            b = np.random.uniform(center-0.2,center+0.2)    
            rand_rates = [1-max(a,b),min(a,b),abs(a-b)]
        length_bg_paths = len(self.bg_paths)-1
        for i in range(mix_num):
            chosen_indexes.append(np.random.randint(0,length_bg_paths))
        return rand_rates,chosen_indexes
        # outline read background image
    # def _get_rand_tg(self,mask,img):
    #     '''
    #     online read background image ,get a image form target domain,use mask and 1-mask to 
    #     a new image, you can look how to implement in a cross domain code about where is the image
    #     come from
    #     '''
    #     raise NotImplementedError
    def __add__(self, other):
        """
        work for combining query and gallery into the test data loader
        """
        return ImageDataset(
            self.data + other.data,
            self.mode + "+" + other.mode,
            cfg=self.cfg,
            pseudo_labels=None,
            transform=self.transform,
            verbose=False,
            sort=False,
        )

    def renew_labels(self, pseudo_labels):
        assert isinstance(pseudo_labels, list)
        assert len(pseudo_labels) == len(
            self.all_data
        ), "the number of pseudo labels should be the same as that of data"

        data = []
        for label, (img_path, _, camid) in zip(pseudo_labels, self.all_data):
            if label != -1:
                data.append((img_path, label, camid))
        self.data = data

        self.num_pids, self.num_cams = self.parse_data(self.data)

        if self.verbose:
            self.show_summary()

    def show_summary(self):
        print(
            bcolors.BOLD
            + "=> Loaded {} from {}".format(self.mode, self.__class__.__name__)
            + bcolors.ENDC
        )
        print("  ----------------------------")
        print("  # ids | # images | # cameras")
        print("  ----------------------------")
        print(
            "  {:5d} | {:8d} | {:9d}".format(
                self.num_pids, len(self.data), self.num_cams
            )
        )
        print("  ----------------------------")
