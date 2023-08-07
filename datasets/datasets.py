
from typing import *
from pathlib import Path
import pickle

from torch.utils.data import Dataset
import torch
import rasterio as rio
import numpy as np

from .utils import *

class TimeSerieDS(Dataset):
    CLOUD_TRESHOLD = 70 # above this value a pixel is considered cloudy in s2cloudless
    CLOUDFREE_TRESHOLD = 0.1 # if the avg number of cloudy pixel is below this value, the image is cloudfree

    def __init__(self,
                 path_dicts: List[Dict[str, Path]],
                 ts_len: int=3,
                 s1_nch: int=2,
                 s2_nch: int=13,
                 s1_range: Tuple[float, float]=(-25.,0.),
                 s2_range: Tuple[float, float]=(0.,10000.),
                 s1_ch: List[int]|None=None,
                 s2_ch: List[int]|None=None,
                 targets: List[Tuple[int, int]]|None=None,
                 sel_mode: Literal['one', 'month', 'all']='one',
                 seed: int|None=None,
                 s1_type: Literal['float', 'uint']='uint'
                 ):
        self.path_dicts = path_dicts
        self.s1_nch = s1_nch
        self.s2_nch = s2_nch
        self.s1_range = s1_range
        self.s2_range = s2_range
        self.ts_len = ts_len
        self.s1_ch = np.arange(s1_nch) if s1_ch is None else np.array(s1_ch)
        self.s2_ch = np.arange(s2_nch) if s2_ch is None else np.array(s2_ch)
        self.sel_mode = sel_mode
        self.s1_type = s1_type

        self._rng = np.random.default_rng(seed=seed)

        if targets is not None:
            self.targets = targets
        
        else:
            self.targets = self.select_targets()
        
    def select_targets(self):

        targets = []
        cloudfrees = []
        self.total_cloudfree_distrib = np.zeros(12) # keep track of this just for stats
        self.target_cloudfree_distrib = np.zeros(12)

        for path_dict in self.path_dicts:
            with rio.open(path_dict['s2cloudless'], 'r') as src:
                cloudfree = (src.read() > self.CLOUD_TRESHOLD).mean(axis=(1, 2)) < self.CLOUDFREE_TRESHOLD
                cloudfrees.append(cloudfree)

            months = np.array(read_months(path_dict['s2_properties']))
            for m in range(12):
                self.total_cloudfree_distrib[m] += np.count_nonzero(cloudfree[months == m] == 1)
        
        for ts_idx, (path_dict, cloudfree) in enumerate(zip(self.path_dicts, cloudfrees)):

            candidate_targets = np.where(cloudfree == 1)[0]
            candidate_targets = candidate_targets[candidate_targets >= self.ts_len-1]
            self._rng.shuffle(candidate_targets)

            months = np.array(read_months(path_dict['s2_properties']))

            if self.sel_mode != 'all':
                _, idx = np.unique(months[candidate_targets], return_index=True)
                candidate_targets = candidate_targets[idx]
            
            if self.sel_mode == 'one':
                # process to favor poorly represented months
                distrib = self.total_cloudfree_distrib[months[candidate_targets]-1]
                proba = np.where(distrib != 0, 1 / (distrib + 1e-5), 0)
                proba = proba / proba.sum()

                candidate_targets = self._rng.choice(candidate_targets, 1, p=proba)
            
            for m in range(12):
                self.target_cloudfree_distrib[m] += np.count_nonzero(months[candidate_targets] == m)

            for idx in candidate_targets:
                targets.append((ts_idx, idx))
        
        return targets
    
    def save(self, path: Path):
        with open(path, 'wb') as src:
            pickle.dump((self.path_dicts, self.targets), src)
    
    @classmethod
    def from_save(cls, path: Path, **kwargs):
        with open(path, 'rb') as src:
            path_dicts, targets = pickle.load(src)
        
        kwargs.update({'targets': targets})
        return TimeSerieDS(path_dicts, **kwargs)
    
    def get_indices(self, idx: int, sen_type: Literal['s1', 's2']):
        if sen_type == 's1':
            nch = self.s1_nch
            ch = self.s1_ch

        elif sen_type == 's2':
            nch = self.s2_nch
            ch = self.s2_ch

        return np.concatenate([1+nch*i+ch for i in range(idx-self.ts_len+1, idx+1)]).tolist()
    
    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self, index) -> Dict[str, torch.Tensor]:
        ts_idx, idx = self.targets[index]
        path_dict = self.path_dicts[ts_idx]

        with rio.open(path_dict['s1'], 'r') as src:
            indices = self.get_indices(idx, 's1')
            s1 = src.read(indices)

            # replaces nans with mean value of the channel
            s1_nan = np.isnan(s1)
            mean = s1.mean(axis=(1, 2), where=~s1_nan, keepdims=True)
            s1 = np.where(s1_nan, mean, s1)

        with rio.open(path_dict['s2'], 'r') as src:
            indices = self.get_indices(idx, 's2')
            s2 = src.read(indices)

        indices = [idx-i for i in range(self.ts_len-1, -1, -1)]
        with rio.open(path_dict['s2cloudless'], 'r') as src:
            s2cloudless = src.read([1+i for i in indices])
        
        s1 = s1[:,:256,:256] # still have to do this :/
        s2 = s2[:,:256,:256] # maybe try and fix this in the data dl pipeline
        s2cloudless = s2cloudless[:,:256,:256] # hard code image size for now

        # TODO: deal with s1 with few nans value?
        # TODO: still need to reimplement dl s1 as uint16

        s1_months = np.array(read_months(path_dict['s1_properties']))[indices]
        s2_months = np.array(read_months(path_dict['s2_properties']))[indices]
        lat, lon = read_lat_lon(path_dict['tile'])

        if self.s1_type == 'uint':
            s1 = - s1 / 1000
        s1 = clip_and_rescale(s1, *self.s1_range, -1, 1)
        s2 = clip_and_rescale(s2, *self.s2_range, -1, 1)

        s1 = torch.from_numpy(s1).to(dtype=torch.float32)
        s2 = torch.from_numpy(s2).to(dtype=torch.float32)
        s2cloudless = torch.from_numpy(s2cloudless).to(dtype=torch.int8)
        s1_months = torch.tensor(s1_months).to(dtype=torch.int8)
        s2_months = torch.tensor(s2_months).to(dtype=torch.int8)
        latlon = torch.tensor([lat, lon]).to(dtype=torch.float32)

        return {
            's1': s1,
            's2': s2,
            's2cloudless': s2cloudless,
            's1_months': s1_months,
            's2_months': s2_months,
            'latlon': latlon
        }




    




    
