from typing import Dict, List, NamedTuple, Optional, Tuple
from enum import Enum, IntEnum
from copy import deepcopy
from torch import Tensor
from contextlib import contextmanager
import math

import numpy as np
import torch
from torch import Tensor
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm


class ComplexityType(Enum):
  INVERSE_MARGIN = 22
  LOG_SUM_OF_SPEC_OVER_MARGIN = 34
  LOG_SUM_OF_SPEC = 35
  LOG_SUM_OF_FRO = 39
  FRO_DIST = 40
  PACBAYES_INIT = 48
  PACBAYES_ORIG = 49
  PACBAYES_FLATNESS = 53


class EvaluationMetrics(NamedTuple):
  acc: float
  avg_loss: float
  num_correct: int
  num_to_evaluate_on: int
  all_complexities: Dict[ComplexityType, float]

# instance 创建实例
CT = ComplexityType

# https://github.com/bneyshabur/generalization-bounds/blob/master/measures.py
#
@torch.no_grad()
def _reparam(model):
  def in_place_reparam(model, prev_layer=None):
    for child in model.children():
      prev_layer = in_place_reparam(child, prev_layer)
      if child._get_name() == 'Conv2d':
        prev_layer = child
      elif child._get_name() == 'BatchNorm2d':
        scale = child.weight / ((child.running_var + child.eps).sqrt())
        prev_layer.bias.copy_( child.bias  + ( scale * (prev_layer.bias - child.running_mean) ) )
        perm = list(reversed(range(prev_layer.weight.dim())))
        prev_layer.weight.copy_((prev_layer.weight.permute(perm) * scale ).permute(perm))
        child.bias.fill_(0)
        child.weight.fill_(1)
        child.running_mean.fill_(0)
        child.running_var.fill_(1)
    return prev_layer
  model = deepcopy(model)
  in_place_reparam(model)
  return model

@contextmanager
def _perturbed_model(
        model,
        sigma: float,
        rng,
        magnitude_eps: Optional[float] = None
):
  device = next(model.parameters()).device
  if magnitude_eps is not None:
    noise = [torch.normal(0,sigma**2 * torch.abs(p) ** 2 + magnitude_eps ** 2, generator=rng) for p in model.parameters()]
  else:
    noise = [torch.normal(0,sigma**2,p.shape, generator=rng).to(device) for p in model.parameters()]
  model = deepcopy(model)
  try:
    [p.add_(n) for p,n in zip(model.parameters(), noise)]
    yield model
  finally:
    [p.sub_(n) for p,n in zip(model.parameters(), noise)]
    del model

# https://drive.google.com/file/d/1_6oUG94d0C3x7x2Vd935a2QqY-OaAWAM/view
@torch.no_grad()
def _pacbayes_sigma(
        model,
        dataloader,
        accuracy: float,
        seed: int,
        magnitude_eps: Optional[float] = None,
        search_depth: int = 15,
        montecarlo_samples: int = 10,
        accuracy_displacement: float = 0.1,
        displacement_tolerance: float = 1e-2,
) -> float:
  lower, upper = 0, 2
  sigma = 1

  BIG_NUMBER = 10348628753
  device = next(model.parameters()).device
  rng = torch.Generator(device=device) if magnitude_eps is not None else torch.Generator()
  rng.manual_seed(BIG_NUMBER + seed)

  for _ in tqdm(range(search_depth), desc="Searching sigma process: "):
    sigma = (lower + upper) / 2
    accuracy_samples = []
    for _ in range(montecarlo_samples):
      with _perturbed_model(model, sigma, rng, magnitude_eps) as p_model:
        loss_estimate = 0
        for data, target in dataloader:
          # data = data.to(device, dtype=torch.float)
          # modified line to the line below to move target to device
          data, target = data.to(device, dtype=torch.float), target.to(device)
          logits = p_model(data)
          pred = logits.data.max(1, keepdim=True)[1]  # get the index of the max logits
          batch_correct = pred.eq(target.data.view_as(pred)).type(torch.FloatTensor).cpu() #
          loss_estimate += batch_correct.sum()
        loss_estimate /= len(dataloader.dataset)
        accuracy_samples.append(loss_estimate)
    displacement = abs(np.mean(accuracy_samples) - accuracy) # 添加扰动后模型在训练集上的平均准确率减原来准确率的绝对值
    if abs(displacement - accuracy_displacement) < displacement_tolerance:
      break
    elif displacement > accuracy_displacement:
      # Too much perturbation
      upper = sigma
    else:
      # Not perturbed enough to reach target displacement
      lower = sigma
  return sigma

@torch.no_grad()
def get_all_measures(model, init_model, dataloader,
                     acc: float,
                     seed: int,
                     ) -> Dict[CT , float]:
  measures = {}

  model = _reparam(model)
  init_model = _reparam(init_model)

  device = next(model.parameters()).device
  m = len(dataloader.dataset)

  def get_weights_only(model) -> List[Tensor]:
    blacklist = {'bias', 'bn'}
    return [p for name, p in model.named_parameters() if all(x not in name for x in blacklist)]

  weights = get_weights_only(model) #获得参数
  dist_init_weights = [p-q for p,q in zip(weights, get_weights_only(init_model))] #参数距离
  d = len(weights)

  def get_vec_params(weights: List[Tensor]) -> Tensor: #获得参数的vec
    return torch.cat([p.view(-1) for p in weights], dim=0)

  w_vec = get_vec_params(weights)
  dist_w_vec = get_vec_params(dist_init_weights) #
  num_params = len(w_vec) #参数数量

  def get_reshaped_weights(weights: List[Tensor]) -> List[Tensor]:
    # If the weight is a tensor (e.g. a 4D Conv2d weight), it will be reshaped to a 2D matrix
    return [p.view(p.shape[0],-1) for p in weights] #变化参数形状

  reshaped_weights = get_reshaped_weights(weights)
  dist_reshaped_weights = get_reshaped_weights(dist_init_weights)

  # Measures on the output of the network
  def _margin(
          model,
          dataloader: DataLoader
  ) -> Tensor:
    margins = []
    for data, target in dataloader:
      data = data.to(device, dtype=torch.float)
      logits = model(data)
      correct_logit = logits[torch.arange(logits.shape[0]), target].clone()
      logits[torch.arange(logits.shape[0]), target] = float('-inf')
      max_other_logit = logits.data.max(1).values  # get the index of the max logits
      margin = correct_logit - max_other_logit
      margins.append(margin)
    return torch.cat(margins).kthvalue(m // 10)[0]

  margin = _margin(model, dataloader).abs()
  measures['1/margin'] = torch.tensor(1, device=device) / margin ** 2 # 22  -----保留----

  # Norm & Margin-Based Measures
  fro_norms = torch.cat([p.norm('fro').unsqueeze(0) ** 2 for p in reshaped_weights])
  spec_norms = torch.cat([p.svd().S.max().unsqueeze(0) ** 2 for p in reshaped_weights])
  dist_fro_norms = torch.cat([p.norm('fro').unsqueeze(0) ** 2 for p in dist_reshaped_weights])
  dist_spec_norms = torch.cat([p.svd().S.max().unsqueeze(0) ** 2 for p in dist_reshaped_weights])
  LOG_PROD_OF_SPEC = spec_norms.log().sum()
  LOG_PROD_OF_FRO = fro_norms.log().sum()

  measures['sum_spectral/margin'] = math.log(d) + (1/d) * (LOG_PROD_OF_SPEC -  2 * margin.log()) # ----保留----
  measures['sum_spectral'] = math.log(d) + (1/d) * LOG_PROD_OF_SPEC # 35  ----保留----
  measures['sum_frobenius'] = math.log(d) + (1/d) * LOG_PROD_OF_FRO # 39  ----保留----
  measures['frobenius_initial'] = dist_fro_norms.sum() # 40 ----保留----



  # # Flatness-based measures
  # sigma = _pacbayes_sigma(model, dataloader, acc, seed)
  # def _pacbayes_bound(reference_vec: Tensor) -> Tensor:
  #   return (reference_vec.norm(p=2) ** 2) / (4 * sigma ** 2) + math.log(m / sigma) + 10
  # measures['pacbayes_initial'] = _pacbayes_bound(dist_w_vec)
  # measures['pacbayes_origin'] = _pacbayes_bound(w_vec)
  # measures['sharpness'] = torch.tensor(1 / sigma ** 2)


  # Adjust for dataset size
  def adjust_measure(measure: str, value: float) -> float:
    if measure == 'sum_spectral/margin' or measure == 'sum_spectral' or measure == 'sum_frobenius':
      return 0.5 * (value - np.log(m))
    else:
      return np.sqrt(value / m)

  return {k: adjust_measure(k, v.item()) for k, v in measures.items()}   

