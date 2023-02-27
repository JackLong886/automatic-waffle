from osgeo import gdal
import numpy as np
import gc
from PIL import Image
import matplotlib.pyplot as plt
from collections import Counter
import time
import os
from DOM_Sub.utils import IMAGE2


if __name__ == '__main__':
    in_file = r'C:\LCJ\image_data\mosaic_img\SV-2_20220723_L2A0000176260_1012201670020003\pansharpen.tif'
    out_file = r'C:\Users\DELL\Desktop\tmp\SV2_0723-2_test.tif'
    img = IMAGE2()
    img.read_img(filename=in_file)
    stat = img.compute_statistics()
    print(stat)
    os.system('gdal_translate -of GTiff -ot Byte -scale 0 1250 0 255 {} {}'.format(in_file, out_file))



