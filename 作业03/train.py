#!/usr/bin/env python
# coding: utf-8

# 数据集图片和标注路径
data_root = 'data/Glomeruli-dataset'
img_dir = 'images'
ann_dir = 'masks'

# 类别和对应的颜色
classes = ('background', 'glomeruili')
palette = [[128, 128, 128], [151, 189, 8]]

from mmseg.registry import DATASETS
from mmseg.datasets import BaseSegDataset

@DATASETS.register_module()
class StanfordBackgroundDataset(BaseSegDataset):
  METAINFO = dict(classes = classes, palette = palette)
  def __init__(self, **kwargs):
    super().__init__(img_suffix='.png', seg_map_suffix='.png', **kwargs)

from mmengine import Config
cfg: Config = Config.fromfile('pspnet_r50_test.py')

# 载入预训练模型权重
cfg.load_from = 'pspnet_r50-d8_512x1024_40k_cityscapes_20200605_003338-2966598c.pth'

print(cfg.pretty_text)

from mmengine.runner import Runner
from mmseg.utils import register_all_modules

# register all modules in mmseg into the registries
# do not init the default scope here because it will be init in the runner
register_all_modules(init_default_scope=False)
runner = Runner.from_cfg(cfg)


# ## 开始训练
#
# 如果遇到报错`CUDA out of memeory`，重启实例或使用显存更高的实例即可。

# In[9]:


runner.train()


# In[ ]:




