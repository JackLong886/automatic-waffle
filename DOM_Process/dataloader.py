from osgeo import gdal
gdal.PushErrorHandler('CPLQuietErrorHandler')
import numpy as np
import cv2


def info(ipt_path):
    img = gdal.Open(ipt_path, gdal.GA_ReadOnly)  # 读取辐射率数据，浮点型，可见光波段
    width = img.RasterXSize  # 影像长
    height = img.RasterYSize  # 影像宽
    # geotrans = img.GetGeoTransform()  # 获取仿射矩阵信息
    # proj = img.GetProjection()  # 获取投影信息
    return width, height, img


def io_info(ipt_path, opt_path, opt_bands=None, opt_driver=None, opt_dtype=None):
    # ---原始数据信息---
    img = gdal.Open(ipt_path, gdal.GA_ReadOnly)  # 读取辐射率数据，浮点型，可见光波段
    width = img.RasterXSize  # 影像长
    height = img.RasterYSize  # 影像宽
    geotrans = img.GetGeoTransform()  # 获取仿射矩阵信息
    proj = img.GetProjection()  # 获取投影信息
    # ---输出数据信息---
    if opt_driver is None:
        opt_driver = img.GetDriver().ShortName
    if opt_dtype is None:
        opt_dtype = img.GetRasterBand(1).DataType
    if opt_bands is None:
        opt_bands = img.RasterCount
    driver = gdal.GetDriverByName(opt_driver)
    datatype = opt_dtype  # 定义数据格式
    out = driver.Create(opt_path, int(width), int(height), opt_bands, datatype)  # 写入空数据
    out.SetGeoTransform(geotrans)  # 写入仿射变换参数
    out.SetProjection(proj)  # 写入投影
    # ---返回数据---
    return width, height, img, out


def block_img(width, height, block_size):
    frame = []
    x = 0
    y = 0
    while y < height:  # 高度方向滑窗
        if y + block_size >= height:
            yoff = y
            ysize = height - y
            y_end = True
        else:
            yoff = y
            ysize = block_size
            y_end = False

        while x < width:  # 宽度方向滑窗
            if x + block_size >= width:
                xoff = x
                xsize = width - x
                x_end = True
            else:
                xoff = x
                xsize = block_size
                x_end = False
            frame.append([xoff, yoff, xsize, ysize])
            x += block_size
            if x_end:
                break
        y += block_size
        x = 0
        if y_end:
            break
    return frame


def block_tif(width, height, block_size):
    y_end = False
    extents = []
    y = 0
    while y < height:
        if y + block_size >= height:
            yoff = y
            ysize = height - y
            y_end = True
        else:
            yoff = y
            ysize = block_size
        extents.append([0, yoff, width, ysize])
        y += block_size
        if y_end: break
    return extents


def block_main(width, height, winx, winy, xstep=0, ystep=0, end_back=False):

    if xstep == 0:
        xstep = winx
    if ystep == 0:
        ystep = winy

    x_half_overlap = (winx - xstep + 1) // 2
    y_half_overlap = (winy - ystep + 1) // 2

    extents = []
    y = 0
    x = 0

    while y < height:
        if y + winy >= height:
            if end_back:
                yoff = y
                ysize = height - y
            else:
                yoff = height - winy
                ysize = winy
            y_end = True
        else:
            yoff = y
            ysize = winy
            y_end = False

        if yoff == 0:
            y_crop_off = 0
            y_crop_size = winy
        else:
            y_crop_off = y_half_overlap
            y_crop_size = winy - y_half_overlap

        if not y_end:
            y_crop_size -= y_half_overlap

        while x < width:
            if x + winx >= width:
                if end_back:
                    xoff = x
                    xsize = width - x
                else:
                    xoff = width - winx
                    xsize = winx
                x_end = True
            else:
                xoff = x
                xsize = winx
                x_end = False

            if xoff == 0:
                x_crop_off = 0
                x_crop_size = winy
            else:
                x_crop_off = x_half_overlap
                x_crop_size = winx - x_half_overlap

            if not x_end:
                x_crop_size -= x_half_overlap

            extents.append([xoff, yoff, xsize, ysize, x_crop_off, y_crop_off, x_crop_size, y_crop_size])
            x += xstep
            if x_end: break
        y += ystep
        x = 0
        if y_end: break

    return extents


def block_writer(outds, blk, ipt_data, pad=False):
    if pad:
        crop_size = int(0.25 * ipt_data.shape[0])
        for i in range(ipt_data.shape[2]):
            outds.GetRasterBand(i + 1).WriteArray(
                ipt_data[crop_size:-crop_size, crop_size:-crop_size, i],
                blk[0],
                blk[1])
    else:
        if len(blk)>4:
            for i in range(ipt_data.shape[2]):
                outds.GetRasterBand(i + 1).WriteArray(
                    ipt_data[blk[5]:blk[5] + blk[7], blk[4]:blk[4] + blk[6], i], blk[0] + blk[4], blk[1] + blk[5]
                )
        else:
            for i in range(ipt_data.shape[2]):
                outds.GetRasterBand(i + 1).WriteArray(ipt_data[:, :, i], blk[0], blk[1])


def block_reader(iptds, blk, pad=False):
    if pad:
        blk = blk[0:4]
        width = iptds.RasterXSize
        height = iptds.RasterYSize
        block_size = blk[2]
        # 顶边和两个顶点
        if blk[1] == 0:

            # 左上
            if blk[0] == 0:
                blk[2] += 0.5 * block_size
                blk[3] += 0.5 * block_size
                blk = list(map(int, blk))

                winds = iptds.ReadAsArray(*blk)
                winds = np.transpose(winds, (1, 2, 0))
                winds = cv2.copyMakeBorder(winds, top=int(0.5 * block_size), bottom=0, left=int(0.5 * block_size),
                                           right=0,
                                           borderType=cv2.BORDER_REFLECT)
                return winds

            # 右上
            elif blk[0] == width - block_size:
                blk[0] -= 0.5 * block_size
                blk[2] += 0.5 * block_size
                blk[3] += 0.5 * block_size
                blk = list(map(int, blk))

                winds = iptds.ReadAsArray(*blk)
                winds = np.transpose(winds, (1, 2, 0))
                winds = cv2.copyMakeBorder(winds, top=int(0.5 * block_size), bottom=0, left=0,
                                           right=int(0.5 * block_size),
                                           borderType=cv2.BORDER_REFLECT)
                return winds

            # 顶边
            else:
                blk[0] -= 0.5 * block_size
                blk[2] += block_size
                blk[3] += 0.5 * block_size
                blk = list(map(int, blk))
                if (blk[2] > width - blk[0]) | (blk[3] > height - blk[1]):
                    addr = blk[2] + blk[0] - width
                    addb = blk[3] + blk[1] - height
                    blk[2] = min(blk[2], width - blk[0])
                    blk[3] = min(blk[3], height - blk[1])
                    winds = iptds.ReadAsArray(*blk)
                    winds = np.transpose(winds, (1, 2, 0))
                    winds = cv2.copyMakeBorder(winds,
                                               top=int(0.5 * block_size),
                                               bottom=max(0, addb),
                                               left=0,
                                               right=max(0, addr),
                                               borderType=cv2.BORDER_REFLECT)
                    return winds
                winds = iptds.ReadAsArray(*blk)
                winds = np.transpose(winds, (1, 2, 0))
                winds = cv2.copyMakeBorder(winds, top=int(0.5 * block_size), bottom=0, left=0, right=0,
                                           borderType=cv2.BORDER_REFLECT)
                return winds

        # 底边和两个顶点
        elif blk[1] == height - block_size:
            # 左下
            if blk[0] == 0:
                blk[1] -= 0.5 * block_size
                blk[2] += 0.5 * block_size
                blk[3] += 0.5 * block_size
                blk = list(map(int, blk))

                winds = iptds.ReadAsArray(*blk)
                winds = np.transpose(winds, (1, 2, 0))
                winds = cv2.copyMakeBorder(winds, top=0, bottom=int(0.5 * block_size), left=int(0.5 * block_size),
                                           right=0,
                                           borderType=cv2.BORDER_REFLECT)
                return winds

            # 右下
            elif blk[0] == width - block_size:
                blk[0] -= 0.5 * block_size
                blk[1] -= 0.5 * block_size
                blk[2] += 0.5 * block_size
                blk[3] += 0.5 * block_size
                blk = list(map(int, blk))

                winds = iptds.ReadAsArray(*blk)
                winds = np.transpose(winds, (1, 2, 0))
                winds = cv2.copyMakeBorder(winds, top=0, bottom=int(0.5 * block_size), left=0,
                                           right=int(0.5 * block_size),
                                           borderType=cv2.BORDER_REFLECT)
                return winds

            # 底边
            else:
                blk[0] -= 0.5 * block_size
                blk[1] -= 0.5 * block_size
                blk[2] += block_size
                blk[3] += 0.5 * block_size
                blk = list(map(int, blk))
                if (blk[2] > width - blk[0]) | (blk[3] > height - blk[1]):
                    addr = blk[2] + blk[0] - width
                    addb = blk[3] + blk[1] - height
                    blk[2] = min(blk[2], width - blk[0])
                    blk[3] = min(blk[3], height - blk[1])
                    winds = iptds.ReadAsArray(*blk)
                    winds = np.transpose(winds, (1, 2, 0))
                    winds = cv2.copyMakeBorder(winds,
                                               top=0,
                                               bottom=int(0.5 * block_size) + addb,
                                               left=0,
                                               right=max(0, addr),
                                               borderType=cv2.BORDER_REFLECT)
                    return winds
                winds = iptds.ReadAsArray(*blk)
                winds = np.transpose(winds, (1, 2, 0))
                winds = cv2.copyMakeBorder(winds, top=0, bottom=int(0.5 * block_size), left=0, right=0,
                                           borderType=cv2.BORDER_REFLECT)
                return winds

        # 左边不含顶点
        elif blk[0] == 0:
            if (blk[1] != 0) & (blk[1] != height - block_size):
                # 左边
                blk[1] -= 0.5 * block_size
                blk[2] += 0.5 * block_size
                blk[3] += block_size
                blk = list(map(int, blk))
                if (blk[2] > width - blk[0]) | (blk[3] > height - blk[1]):
                    addr = blk[2] + blk[0] - width
                    addb = blk[3] + blk[1] - height
                    blk[2] = min(blk[2], width - blk[0])
                    blk[3] = min(blk[3], height - blk[1])
                    winds = iptds.ReadAsArray(*blk)
                    winds = np.transpose(winds, (1, 2, 0))
                    winds = cv2.copyMakeBorder(winds, top=0, bottom=max(0, addb),
                                               left=int(0.5 * block_size), right=max(0, addr),
                                               borderType=cv2.BORDER_REFLECT)
                    return winds
                winds = iptds.ReadAsArray(*blk)
                winds = np.transpose(winds, (1, 2, 0))
                winds = cv2.copyMakeBorder(winds, top=0, bottom=0, left=int(0.5 * block_size), right=0,
                                           borderType=cv2.BORDER_REFLECT)
                return winds

        # 右边不含顶点
        elif blk[0] == width - block_size:
            if (blk[1] != 0) & (blk[1] != height):
                # 右边
                blk[0] -= 0.5 * block_size
                blk[1] -= 0.5 * block_size
                blk[2] += 0.5 * block_size
                blk[3] += block_size
                blk = list(map(int, blk))
                if (blk[2] > width - blk[0]) | (blk[3] > height - blk[1]):
                    addb = blk[3] + blk[1] - height
                    blk[3] = min(blk[3], height - blk[1])
                    winds = iptds.ReadAsArray(*blk)
                    winds = np.transpose(winds, (1, 2, 0))
                    winds = cv2.copyMakeBorder(winds, top=0, bottom=max(0, addb),
                                               left=0, right=int(0.5 * block_size),
                                               borderType=cv2.BORDER_REFLECT)
                    return winds
                winds = iptds.ReadAsArray(*blk)
                winds = np.transpose(winds, (1, 2, 0))
                winds = cv2.copyMakeBorder(winds, top=0, bottom=0, left=0, right=int(0.5 * block_size),
                                           borderType=cv2.BORDER_REFLECT)
                return winds

        # 中间
        else:
            # 中间
            blk[0] -= 0.5 * block_size
            blk[1] -= 0.5 * block_size
            blk[2] += block_size
            blk[3] += block_size
            blk = list(map(int, blk))
            if (blk[2] > width - blk[0]) | (blk[3] > height - blk[1]):
                addr = blk[2] + blk[0] - width
                addb = blk[3] + blk[1] - height
                blk[2] = min(blk[2], width - blk[0])
                blk[3] = min(blk[3], height - blk[1])
                winds = iptds.ReadAsArray(*blk)
                winds = np.transpose(winds, (1, 2, 0))
                winds = cv2.copyMakeBorder(winds, top=0, bottom=max(0, addb), left=0, right=max(0, addr),
                                           borderType=cv2.BORDER_REFLECT)
                return winds
            winds = iptds.ReadAsArray(*blk)
            winds = np.transpose(winds, (1, 2, 0))
            return winds
    else:
        if len(blk) > 4:
            return np.einsum('ijk->jki', iptds.ReadAsArray(*blk[0:4]))
        else:
            return np.einsum('ijk->jki', iptds.ReadAsArray(*blk))


if __name__ == '__main__':
    col, row, ipt, opt = io_info(r'C:\Users\zou\Desktop\l1.tif', r'C:\Users\zou\Desktop\temp.tif')
    block = block_main(col, row, 512, 512, 400, 400)
    for blocks in block:
        temp = block_reader(ipt, blocks)
        block_writer(opt, blocks, temp)
