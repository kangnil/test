from abc import abstractmethod
from typing import List, TypedDict, Union, Dict
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from .constants import HF_CACHE_DIR

class ImageTextDict(TypedDict):
    images: List[str]
    texts: List[str]

class Score(nn.Module):

    def __init__(self,
                 model: str,
                 device: str='cuda',
                 cache_dir: str=HF_CACHE_DIR,
                 **kwargs):
        """Initialize the ScoreModel
        """
        super().__init__()
        assert model in self.list_all_models()
        self.device = device
        self.model = self.prepare_scoremodel(model, device, cache_dir, **kwargs)
    
    @abstractmethod
    def prepare_scoremodel(self,
                           model: str,
                           device: str,
                           cache_dir: str,
                           **kwargs):
        """Prepare the ScoreModel
        """
        pass
    
    @abstractmethod
    def list_all_models(self) -> List[str]:
        """List all available models
        """
        pass

    def forward(self,
                images: Union[str, List[str]],
                texts: Union[str, List[str]],
                **kwargs) -> torch.Tensor:
        """Return the similarity score(s) between the image(s) and the text(s)
        If there are m images and n texts, return a m x n tensor
        """
        if type(images) == str:
            images = [images]
        if type(texts) == str:
            texts = [texts]
        
        scores = torch.zeros(len(images), len(texts)).to(self.device)
        for i, image in enumerate(images):
            scores[i] = self.model.forward([image] * len(texts), texts, **kwargs)
        return scores
    
    def batch_forward(self,
                      dataset_image: torch.Tensor,
                      dataset_text: str,
                      batch_size: int=16,
                      **kwargs) -> torch.Tensor:
        """Return the similarity score(s) between the image(s) and the text(s)
        If there are m images and n texts, return a m x n tensor
        """
        # num_samples = len(dataset)
        num_images =dataset_image.shape[0]
        # num_texts = len(dataset[0]['texts'])
        scores = torch.zeros(num_images, 1).to(self.device)
 
        dataloader = DataLoader(dataset_image, batch_size=batch_size, shuffle=False)
        counter = 0

         
        for batch_idx, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
            cur_batch_size = batch.shape[0]
            all_images = list(batch.unbind(0))
            all_texts = [dataset_text] * len(all_images)  
            combined_scores = self.model.forward(all_images, all_texts, **kwargs)  # Output: (cur_batch_size * num_images,)
            scores[counter:counter + cur_batch_size, :] = combined_scores.unsqueeze(1)
            counter += cur_batch_size
        return scores