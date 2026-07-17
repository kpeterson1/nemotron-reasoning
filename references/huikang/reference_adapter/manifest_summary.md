# Adapter Manifest Summary

Total tensors: 12008

## Prefix kind counts

- backbone: 12008

## Expert kind counts

- <blank>: 232
- per_expert: 11776

## Mamba/projection conversion counts

- <blank>: 11916
- in_proj: 46
- out_proj: 46

## Rank counts

- 32: 12008

## Projection counts

- down_proj: 5934
- in_proj: 46
- k_proj: 12
- o_proj: 12
- out_proj: 46
- q_proj: 12
- up_proj: 5934
- v_proj: 12

## Quick red flags

- Tinker-style prefix `base_model.model.model`: 0
- Kaggle-style prefix `base_model.model.backbone`: 12008
- Fused expert `experts.w1`: 0
- Fused expert `experts.w2`: 0
- Fused expert `experts.w3`: 0
- Per-expert tensors: 11776
- `gate_proj` tensors: 0
- `x_proj` tensors: 0
- `in_proj` tensors: 46
