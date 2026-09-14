import argparse
import re
from frame_v2.config import AppCfg, ModelCfg, RunCfg
from frame_v2.framework.runner import run, register
from frame_v2.tasks.cifar import CifarTask
from frame_v2.tasks.clevr import ClevrTask
from frame_v2.tasks.cars import CarsTask
from frame_v2.tasks.country import CountryTask
from frame_v2.viz.plotting import visualize
from frame_v2.tasks.dmlab import DMLabTask
from frame_v2.tasks.diabetic_retinopathy import DiabeticRetinopathyTask
from frame_v2.tasks.dtd import DTDTask
from frame_v2.tasks.eurosat import EuroSATTask
from frame_v2.tasks.fer2013 import FER2013Task
from frame_v2.tasks.gtsrb import GTSRBTask
from frame_v2.tasks.kitti import KITTITask
from frame_v2.tasks.mnist import MNISTTask
from frame_v2.tasks.oxford_flowers import OxfordFlowersTask
from frame_v2.tasks.oxford_pet import OxfordPetTask
from frame_v2.tasks.patch_camelyon import PatchCamelyonTask
from frame_v2.tasks.sun397 import SUN397Task
from frame_v2.tasks.svhn import SVHNTask
from frame_v2.tasks.voc_pose import VOCPoseTask
from frame_v2.tasks.resisc45 import RESISC45Task
from frame_v2.tasks.stl10 import STL10Task
from frame_v2.tasks.renderedsst2 import RenderedSST2Task

# register tasks
register(CifarTask); register(ClevrTask); register(CarsTask); register(CountryTask);register(DMLabTask); 
register(DiabeticRetinopathyTask); register(DTDTask); register(EuroSATTask); register(FER2013Task); 
register(GTSRBTask); register(KITTITask); register(MNISTTask); register(OxfordFlowersTask); 
register(OxfordPetTask); register(PatchCamelyonTask); register(RESISC45Task); 
register(STL10Task); register(RenderedSST2Task);register(SUN397Task); register(SVHNTask); register(VOCPoseTask); 


TASK_KEYS = [
    "car",
    "cifar",
    "clevr",
    "country",
    "dmlab",
    "dr",
    "dtd",
    "eurosat",
    "fer2013",
    "gtsrb",
    "kitti",
    "mnist",
    "oxford_flowers",
    "oxford_pet",
    "patch_camelyon",
    "renderedsst2",
    "resisc45",
    "stl10",
    "sun397",
    "svhn",
    "voc_pose",
]






def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="RN50")
    p.add_argument("--pretrained", default="openai")
    p.add_argument("--gpu", default="0")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n", type=int, default=100, help="N images per task")
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--tasks", nargs="*", default=TASK_KEYS, help="task keys from registry")
    p.add_argument("--milestones", nargs="*", type=int, default=[1,5,10,15,20,25,50,75,100], help="milestones for incremental saving")
    return p.parse_args()


if __name__ == "__main__":
    a = parse_args()
    cfg = AppCfg(model=ModelCfg(a.model, a.pretrained), run=RunCfg(seed=a.seed, gpu_id=a.gpu, n_per_task=a.n, batch_size=a.batch_size, milestones=a.milestones), tasks=a.tasks)
    results, layer_names = run(cfg)
    visualize(results, layer_names, cfg.run.output_dir, cfg.model.name, cfg.model.pretrained)