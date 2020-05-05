import matplotlib.pyplot as plt
import numpy as np
from utils import flow_vis
import sys
import cvbase as cvb
from PIL import Image

# flow_path_x = '/home/m3kowal/Research/vfhlt/PyTorchConv3D/data/KINETICS/kinetics100/valid/snatch_weight_lifting/7s0_uEmFEmU_000009_000019/flow_x_134.png'
# flow_path_y = '/home/m3kowal/Research/vfhlt/PyTorchConv3D/data/KINETICS/kinetics100/valid/snatch_weight_lifting/7s0_uEmFEmU_000009_000019/flow_y_134.png'

flow_path_x = '/media/ssd1/m3kowal/videos/SyntheticCarHit/b1018-GreenCity-Dawn-Cloudy-141_07-Civilian_Father-Kite-p-8a69b3906c489307fc795e1c41885001/flow_x_55.png'
flow_path_y = '/media/ssd1/m3kowal/videos/SyntheticCarHit/b1018-GreenCity-Dawn-Cloudy-141_07-Civilian_Father-Kite-p-8a69b3906c489307fc795e1c41885001/flow_y_55.png'

flow_x = Image.open(flow_path_x)
flow_y = Image.open(flow_path_y)

flow_x = np.asarray(flow_x)
flow_y = np.asarray(flow_y)
flow = np.stack([flow_x, flow_y])
print(np.unique(flow_x))
print(np.unique(flow_y))

flow = flow - 0.5


# flow.shape = [b, 2, 256, 256]
flow_color = flow_vis.flow_to_color(flow.transpose(1,2,0), convert_to_bgr=False)
plt.imshow(flow_color)
plt.show()


# rgb_img = img[0,:,0,:,:].detach().cpu().numpy().transpose(1, 2, 0)
# rgb_img = rgb_img.astype(np.int)
# # Display the image
# fig, ax = plt.subplots(1, 2)
# ax[0].imshow(rgb_img)
# ax[1].imshow(flow_color.transpose(1,0,2))
# plt.show()
# plt.imshow(img)
# plt.show()