# demo_free_inputs.py
import os, torch
import torchvision.transforms as T
from torchvision.utils import save_image
from lin_ign import IGN  # adjust import
from utils import find_latest_checkpoint, make_ign_inputs


def main(conf):
    device = torch.device(conf.device)
    model = IGN(conf).to(device)
    model.load_checkpoint(conf.ckpt)  # implement load

    inputs = make_ign_inputs(size=conf.img_size)

    os.makedirs(conf.out_dir, exist_ok=True)
    for name, x in inputs.items():
        x = x.to(device)
        y = model(x)   # project/generate
        save_image(x, os.path.join(conf.out_dir, f"in_{name}.png"))
        save_image(y, os.path.join(conf.out_dir, f"out_{name}.png"))
        print(f"Saved {name}")

if __name__ == "__main__":
    class C: pass
    conf = C()
    conf.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    conf.img_size = 28
    conf.ckpt = find_latest_checkpoint("./results/ign_mnist/ckpts")
    conf.out_dir = "./results/ign_mnist/free_inputs"
    main(conf)