import time
import torch
import torchvision
import numpy as np

if torch.cuda.is_available():
    print('Cuda avilable',torch.cuda.get_device_name(0))
else:
    print('cuda is not avilable')


def tput(model, name):
    with torch.no_grad():
        input = torch.rand(1, 3, 224, 224)
        if torch.cuda.is_available():
            input = input.to('cuda')
            model = model.to('cuda')
        model.eval()
        model(input)
        times = list()
        for _ in range(50):
            t1 = time.time()
            model(input)
            t2 = time.time()
            times.append(t2-t1)
    print(f"Model {name}: {np.mean(times[10:]) * 1000}")


if __name__ == '__main__':
    print('Torchvision classification models test')
    all_model_names = torchvision.models.list_models(module=torchvision.models)
    skip_list = ["regnet_y_128gf", "vit_h_14"]
    for model_name in all_model_names:
        if model_name not in skip_list:
            tput(torchvision.models.get_model(model_name, weights=None), model_name)
