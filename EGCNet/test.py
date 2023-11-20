import torch
from tqdm import tqdm
from opt import opt
from utils.metrics import evaluate
import datasets
from torch.utils.data import DataLoader
from utils.comm import generate_model
from utils.metrics import Metrics
from  torchvision.utils import   save_image
from utils.sobel import SobelComputer
save_path = "cli"
import cv2
import numpy as np
import torch.nn.functional as F
import  matplotlib.pyplot as plt
from PIL import Image
from utils.sobel import SobelComputer



def test():
    print('loading data......')
    test_data = getattr(datasets, opt.dataset)(opt.root, opt.test_data_dir, mode='test')
    test_dataloader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=opt.num_workers)
    total_batch = int(len(test_data) / 1)
    model = generate_model(opt)

    model.eval()

    # metrics_logger initialization
    metrics = Metrics(['recall', 'specificity', 'precision', 'F1', 'F2',
                       'ACC_overall', 'IoU_poly', 'IoU_bg', 'IoU_mean'])

    with torch.no_grad():
        bar = tqdm(enumerate(test_dataloader), total=total_batch)
        for i, (data , filename) in bar:
            img, gt = data['image'], data['label']

            if opt.use_gpu:
                img = img.cuda()
                gt = gt.cuda()

            output,pr1,pr2,pr3,pr4,edge1,edge0,gc,merge= model(img)
            _recall, _specificity, _precision, _F1, _F2, \
            _ACC_overall, _IoU_poly, _IoU_bg, _IoU_mean = evaluate(output, gt)


            

            edge= F.upsample(edge0, size=(320,320), mode='bilinear', align_corners=False)
            save_image(edge,"./kva_edge/{}.png".format(filename[0]))

            # gc = gc.detach().cpu().numpy()
            # gc = gc[0][0]



            # image_heatmap = cv2.normalize(gc, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            # image_heatmap = cv2.applyColorMap(image_heatmap, 2)
            # cv2.imwrite('./gc/gc.jpg', image_heatmap)

            metrics.update(recall= _recall, specificity= _specificity, precision= _precision,
                            F1= _F1, F2= _F2, ACC_overall= _ACC_overall, IoU_poly= _IoU_poly,
                            IoU_bg= _IoU_bg, IoU_mean= _IoU_mean
                        )

            #
            # image_heatmap = cv2.normalize(gc, None, alpha=0, beta=500, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            # image_heatmap = cv2.applyColorMap(image_heatmap, 6)
            # cv2.imwrite('./gc/gc1.jpg', image_heatmap)

    metrics_result = metrics.mean(total_batch)

    print("Test Result:")
    print('recall: %.4f, specificity: %.4f, precision: %.4f, F1: %.4f, F2: %.4f, '
          'ACC_overall: %.4f, IoU_poly: %.4f, IoU_bg: %.4f, IoU_mean: %.4f'
          % (metrics_result['recall'], metrics_result['specificity'], metrics_result['precision'],
             metrics_result['F1'], metrics_result['F2'], metrics_result['ACC_overall'],
             metrics_result['IoU_poly'], metrics_result['IoU_bg'], metrics_result['IoU_mean']))


if __name__ == '__main__':


    print('--- PolypSeg Test---')
    test()

    print('Done')
