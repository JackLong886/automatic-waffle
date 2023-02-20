import os
import shutil
import numpy as np
import numexpr as ne
import dataloader as dl
from osgeo import gdalconst, gdal
from osgeo_utils.gdal_pansharpen import gdal_pansharpen
import sys
import argparse
import ChinaSensor as sensors
import json
import time
import pywt
from ChinaSensor import weights
global global_message


def arcgisopt(msds):
    highlist = []
    lowlist = []
    for b in range(msds.RasterCount):
        msband = msds.GetRasterBand(b + 1).ReadAsArray()
        low_value = np.percentile(msband[msband != 0], 0.5)
        high_value = np.percentile(msband[msband != 0], 99.5)
        highlist.append(high_value)
        lowlist.append(low_value)
    return highlist, lowlist


def gamma_trans(img, gamma):  # gamma函数处理
    gamma_table = [np.power(x / 255.0, gamma) * 255.0 for x in range(256)]  # 建立映射表
    gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)  # 颜色值为整数
    return gamma_table[img] # 图片颜色查表。另外可以根据光强（颜色）均匀化原则设计自适应算法。


def stretch(high, low, msband):
    msband = msband.astype(np.float32)
    for i in range(msband.shape[2]):
        msband[:, :, i] = (msband[:, :, i] - low[i]) / (high[i] - low[i])
    return np.clip(msband*400, 1, 255).astype(np.uint8)


def GenExtents(width, height, win_size, win_std=0):
    if win_std == 0:
        win_std = win_size
    frame = []
    x = 0
    y = 0
    while y < height:  # 高度方向滑窗
        if y + win_size >= height:
            y_left = height - win_size
            y_right = win_size
            y_end = True
        else:
            y_left = y
            y_right = win_size
            y_end = False

        while x < width:  # 宽度方向滑窗
            if x + win_size >= width:
                x_left = width - win_size
                x_right = win_size
                x_end = True
            else:
                x_left = x
                x_right = win_size
                x_end = False
            frame.append((x_left, y_left, x_right, y_right))
            x += win_std
            if x_end:
                break
        y += win_std
        x = 0
        if y_end:
            break
    return frame


class IMAGE2:
    # 读图像文件
    def read_img(self, filename):
        self.in_file = filename
        self.dataset = gdal.Open(self.in_file)  # 打开文件
        self.im_width = self.dataset.RasterXSize  # 栅格矩阵的列数
        self.im_height = self.dataset.RasterYSize  # 栅格矩阵的行数
        self.im_bands = self.dataset.RasterCount  # 波段数
        self.im_geotrans = self.dataset.GetGeoTransform()  # 仿射矩阵，左上角像素的大地坐标和像素分辨率
        self.im_proj = self.dataset.GetProjection()  # 地图投影信息，字符串表示

    def get_extent(self, extent):
        x, y, s_size, y_size = extent
        dataset = gdal.Open(self.in_file)
        extent_img = dataset.ReadAsArray(x, y, s_size, y_size)
        return extent_img

    def create_img(self, filename, out_bands, datatype=gdal.GDT_UInt16):
        self.out_bands = out_bands
        # 创建文件
        driver = gdal.GetDriverByName("GTiff")
        self.output_dataset = driver.Create(filename, self.im_width, self.im_height, out_bands, datatype)
        self.output_dataset.SetGeoTransform(self.im_geotrans)  # 写入仿射变换参数
        self.output_dataset.SetProjection(self.im_proj)  # 写入投影

    def write_extent(self, extent, im_data):
        x, y, s_size, y_size = extent
        if self.out_bands == 1:
            self.output_dataset.GetRasterBand(1).WriteArray(im_data, xoff=x, yoff=y)  # 写入数组数据
        else:
            for i in range(self.out_bands):
                self.output_dataset.GetRasterBand(i + 1).WriteArray(im_data[i], xoff=x, yoff=y)

    def compute_statistics(self):
        # min max mean std
        statis = []
        for i in range(self.im_bands):
            s = self.dataset.GetRasterBand(i + 1).ComputeStatistics(True)
            statis.append(s)
        return statis


def Brovey(pan, hs, pan_stat, hs_stat, weights=(1, 1, 1, 1)):
    m, n, C = hs.shape
    u_hs = hs
    hs_stat = np.mean(hs_stat, axis=0)
    pmin, pmax, pmean, pstd = pan_stat[0]
    hmin, hmax, hmean, hstd = hs_stat

    # I = np.mean(u_hs, axis=-1)
    I = np.average(u_hs, axis=-1, weights=weights)

    image_hr = ne.evaluate('(pan - pmean) * (hstd / pstd) + hmean')
    image_hr = np.squeeze(image_hr)
    I_Brovey = []
    for i in range(C):
        u_hs_tmp = u_hs[:, :, i]
        temp = ne.evaluate('image_hr * u_hs_tmp / (I + 1e-8)')
        temp = np.expand_dims(temp, axis=-1)
        I_Brovey.append(temp)
    I_Brovey = np.concatenate(I_Brovey, axis=-1)
    return np.float32(I_Brovey)


def rpc_process(input_image, temp_path, rpc_height=0.0, rpc_dem=None):
    if not os.path.exists(input_image):
        raise KeyError('找不到文件' + str(input_image))
    if not os.path.exists(temp_path):
        os.makedirs(temp_path)
    file_name = os.path.basename(input_image)
    prefix, suffix = os.path.splitext(file_name)[0], os.path.splitext(file_name)[-1]
    out_name = prefix + '_rpc.vrt'
    rpc_outfile = os.path.normpath(os.path.join(temp_path, out_name))

    if rpc_dem is None or rpc_dem == '':
        trans_opt = ['RPC_HEIGHT={0}'.format(str(rpc_height))]
    else:
        trans_opt = ['RPC_HEIGHT={0}'.format(str(rpc_height)), 'RPC_DEM={0}'.format(str(rpc_dem))]
    input_datasets = gdal.Open(input_image, gdal.GA_Update)
    options = gdal.WarpOptions(format='VRT',
                               dstSRS='EPSG:32649',
                               dstNodata=0, srcNodata=0,
                               resampleAlg='bilinear',
                               outputType=gdal.GDT_UInt16,
                               rpc=True,
                               transformerOptions=trans_opt,
                               )
    rpc_datasets = gdal.Warp(destNameOrDestDS=rpc_outfile, srcDSOrSrcDSTab=input_datasets, options=options)
    if rpc_datasets is None:
        raise KeyError('缺少RPC相关文件，请放置于影像相同目录!')
    return rpc_outfile, rpc_datasets


def get_bbox(path):
    """
    Return the bounding box of a raster plus its resolution.
    path: Raster path
    return [minx, miny, maxx, maxy, resx, resy]
    """
    src_ds = gdal.Open(str(path), gdal.GA_ReadOnly)
    geoTrans = src_ds.GetGeoTransform()
    resX = geoTrans[1]
    resY = geoTrans[5]
    ulx = geoTrans[0]
    uly = geoTrans[3]
    lrx = ulx + (src_ds.RasterXSize * resX)
    lry = uly + (src_ds.RasterYSize * resY)

    bbox = [ulx, lry, lrx, uly, resX, resY]
    return bbox


def callback2(v1, v2, v3):
    sys.stdout.flush()
    print("{}:{:.4f}".format(global_message, v1), flush=True)


def gdal_resample(pan_path, ms_path, temp_path):
    bbox = get_bbox(pan_path)
    inputrasfile = gdal.Open(ms_path, gdal.GA_ReadOnly)
    inputProj = inputrasfile.GetProjection()
    # 获取参考影像信息
    referencefile = gdal.Open(pan_path, gdal.GA_ReadOnly)
    referencefileProj = referencefile.GetProjection()
    # 创建重采样输出文件
    file_name = os.path.basename(ms_path)
    prefix, suffix = os.path.splitext(file_name)[0], os.path.splitext(file_name)[-1]
    out_name = prefix + '_resized.vrt'
    ms_img_resized = os.path.normpath(os.path.join(temp_path, out_name))
    options = gdal.WarpOptions(srcSRS=inputProj, dstSRS=referencefileProj, format='VRT',
                               resampleAlg='Bilinear',
                               xRes=bbox[4], yRes=bbox[5],
                               outputBounds=(bbox[0], bbox[1], bbox[2], bbox[3]))
    ds = gdal.Warp(destNameOrDestDS=ms_img_resized, srcDSOrSrcDSTab=inputrasfile, options=options)
    del inputrasfile, referencefile
    return ms_img_resized


def IHS(pan, hs, pan_stat, hs_stat, weights=(1, 1, 1, 1)):
    m, n, C = hs.shape
    hs_stat = np.mean(hs_stat, axis=0)
    _, _, pmean, pstd = pan_stat[0]
    _, _, hmean, hstd = hs_stat
    u_hs = hs
    # I = np.mean(u_hs, axis=-1, keepdims=True)
    I = np.average(u_hs, axis=-1, weights=weights)
    I = np.expand_dims(I, axis=-1)
    P = (pan - pmean) * (hstd / pstd) + hmean
    I_IHS = u_hs + np.tile(P - I, (1, 1, C))
    return np.float32(I_IHS)


def Wavelet(pan, hs, pan_stat, hs_stat):
    m, n, C = hs.shape
    u_hs = hs

    hs_stat = np.mean(hs_stat, axis=0)
    _, _, pmean, pstd = pan_stat[0]
    _, _, hmean, hstd = hs_stat

    pan = np.squeeze(pan)
    pc = pywt.wavedec2(pan, 'haar', level=2)

    rec = []
    for i in range(C):
        temp_dec = pywt.wavedec2(u_hs[:, :, i], 'haar', level=2)

        pc[0] = temp_dec[0]

        temp_rec = pywt.waverec2(pc, 'haar')
        temp_rec = np.expand_dims(temp_rec, -1)
        rec.append(temp_rec)

    I_Wavelet = np.concatenate(rec, axis=-1)

    return np.float32(I_Wavelet)


def estimation_alpha(pan, hs, mode='global'):
    alpha = 0
    if mode == 'global':
        IHC = np.reshape(pan, (-1, 1))
        ILRC = np.reshape(hs, (hs.shape[0] * hs.shape[1], hs.shape[2]))

        alpha = np.linalg.lstsq(ILRC, IHC, rcond=None)[0]

    elif mode == 'local':
        patch_size = 32
        all_alpha = []

        for i in range(0, hs.shape[0] - patch_size, patch_size):
            for j in range(0, hs.shape[1] - patch_size, patch_size):
                patch_pan = pan[i:i + patch_size, j:j + patch_size, :]
                patch_hs = hs[i:i + patch_size, j:j + patch_size, :]

                IHC = np.reshape(patch_pan, (-1, 1))
                ILRC = np.reshape(patch_hs, (-1, hs.shape[2]))

                local_alpha = np.linalg.lstsq(ILRC, IHC)[0]
                all_alpha.append(local_alpha)

        all_alpha = np.array(all_alpha)

        alpha = np.mean(all_alpha, axis=0, keepdims=False)

    return alpha


def GSA(pan, hs):
    M, N, c = pan.shape
    m, n, C = hs.shape
    u_hs = hs
    # remove means from u_hs
    means = np.mean(u_hs, axis=(0, 1))
    image_lr = u_hs - means
    # remove means from hs
    image_lr_lp = hs - np.mean(hs, axis=(0, 1))
    # sintetic intensity
    image_hr = pan - np.mean(pan)
    image_hr0 = np.expand_dims(image_hr, -1)
    alpha = estimation_alpha(image_hr0, np.concatenate((image_lr_lp, np.ones((m, n, 1))), axis=-1), mode='global')
    I = np.dot(np.concatenate((image_lr, np.ones((M, N, 1))), axis=-1), alpha)
    I0 = I - np.mean(I)
    # computing coefficients
    g = [1]
    for i in range(C):
        temp_h = image_lr[:, :, i]
        c = np.cov(np.reshape(I0, (-1,)), np.reshape(temp_h, (-1,)), ddof=1)
        g.append(c[0, 1] / np.var(I0))
    g = np.array(g)
    # detail extraction
    delta = image_hr - I0
    deltam = np.tile(delta, (1, 1, C + 1))
    # fusion
    V = np.concatenate((I0, image_lr), axis=-1)
    g = np.expand_dims(g, 0)
    g = np.expand_dims(g, 0)
    g = np.tile(g, (M, N, 1))
    V_hat = V + g * deltam
    I_GSA = V_hat[:, :, 1:]
    I_GSA = I_GSA - np.mean(I_GSA, axis=(0, 1)) + means
    return np.float32(I_GSA)


def GS(pan, hs, pan_stat, hs_stat, weights=(1, 1, 1, 1)):
    M, N, c = pan.shape
    m, n, C = hs.shape
    u_hs = hs
    # remove means from u_hs
    means = np.mean(u_hs, axis=(0, 1))
    image_lr = u_hs - means

    hs_stat = np.mean(hs_stat, axis=0)
    _, _, pmean, pstd = pan_stat[0]
    _, _, hmean, hstd = hs_stat

    # sintetic intensity
    I = np.expand_dims(np.average(u_hs, axis=2, weights=weights), axis=2)
    # I0 = I - np.mean(I)
    I0 = I - hmean

    image_hr = (pan - pmean) * (hstd / pstd) + hmean
    # computing coefficients
    g = [1]

    for i in range(C):
        temp_h = image_lr[:, :, i]
        c = np.cov(np.reshape(I0, (-1,)), np.reshape(temp_h, (-1,)), ddof=1)
        g.append(c[0, 1] / np.var(I0))
    g = np.array(g)

    # detail extraction
    delta = image_hr - I0
    deltam = np.tile(delta, (1, 1, C + 1))

    # fusion
    V = np.concatenate((I0, image_lr), axis=-1)

    g = np.expand_dims(g, 0)
    g = np.expand_dims(g, 0)
    g = np.tile(g, (M, N, 1))
    V_hat = V + g * deltam
    I_GS = V_hat[:, :, 1:]
    I_GS = I_GS - np.mean(I_GS, axis=(0, 1)) + means
    return np.float32(I_GS)


def fuse_pan_ms(pan_path, ms_path, output_path, Method='Brovey2', ms_win_size=4096, weights=(1, 1, 1, 1)):
    print('pansharpen start')
    ms = IMAGE2()
    pan = IMAGE2()
    ms.read_img(ms_path)
    pan.read_img(pan_path)
    ms_stat = ms.compute_statistics()
    pan_stat = pan.compute_statistics()
    ms_extents = GenExtents(ms.im_width, ms.im_height, win_size=ms_win_size)
    pan_extents = GenExtents(pan.im_width, pan.im_height, win_size=ms_win_size)
    print(ms.im_width, ms.im_height, ms.im_bands, flush=True)
    print(pan.im_width, pan.im_height, pan.im_bands, flush=True)
    fused_img = pan
    fused_img.create_img(filename=output_path, out_bands=3, datatype=gdal.GDT_Byte)
    total = len(pan_extents)
    finish = 0
    global global_message
    for ms_extent, pan_extent in zip(ms_extents, pan_extents):
        m = ms.get_extent(ms_extent)
        p = pan.get_extent(pan_extent)
        original_msi = m.transpose(1, 2, 0)
        original_pan = np.expand_dims(p, axis=0).transpose([1, 2, 0])

        used_pan = original_pan
        used_ms = original_msi

        fused_image = np.full_like(used_pan, 0)
        if Method == 'Brovey2':
            fused_image = Brovey(used_pan[:, :, :], used_ms[:, :, :],
                                 pan_stat, ms_stat,
                                 weights=weights)
        elif Method == 'IHS':
            fused_image = IHS(used_pan[:, :, :], used_ms[:, :, :],
                              pan_stat, ms_stat,
                              weights=weights)
        elif Method == 'GSA':
            fused_image = GSA(used_pan[:, :, :], used_ms[:, :, :])

        elif Method == 'GS':
            fused_image = GS(used_pan[:, :, :], used_ms[:, :, :],
                             pan_stat, ms_stat,
                             weights=weights)

        elif Method == 'Wavelet':
            fused_image = Wavelet(used_pan[:, :, :], used_ms[:, :, :], pan_stat, ms_stat)

        mask_f = fused_image[:, :, (2, 1, 0)] == 0
        norm_windata = stretch(hl, ll, fused_image)[:, :, (2, 1, 0)]
        norm_windata[mask_f] = 0
        tci = gamma_trans(norm_windata, 0.625)
        tci = np.transpose(tci, (2, 0, 1))[(2, 1, 0), :, :]
        fused_img.write_extent(im_data=tci, extent=pan_extent)
        finish = finish + 1

        print("{}:{:.4f}".format(global_message, finish / total - 0.0001), flush=True)

    fused_img.output_dataset.BuildOverviews('NEAREST', [4, 8, 16, 32, 64, 128])
    print("{}:{:.4f}".format(global_message, 1.), flush=True)
    del ms, pan, fused_img
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser('rpc and pansharpen for pan and ms image')
    parser.add_argument('--json_path', type=str,
                        default=r''
                        )
    parser.add_argument('--ms_path', type=str,
                        default=r'C:\LCJ\image_data\mosaic_img\GF2_PMS2_E110.3_N24.5_20220928_L1A0006781153\GF2_PMS2_E110.3_N24.5_20220928_L1A0006781153-MSS2.tiff',
                        # required=True,
                        )
    parser.add_argument('--pan_path', type=str,
                        default=r'C:\LCJ\image_data\mosaic_img\GF2_PMS2_E110.3_N24.5_20220928_L1A0006781153\GF2_PMS2_E110.3_N24.5_20220928_L1A0006781153-PAN2.tiff',
                        # required=True,
                        )
    parser.add_argument('--method', type=str,
                        default='Brovey',
                        # required=True,
                        help='Brovey, IHS, GSA, GS, Wavelet, Brovey2')
    parser.add_argument('--output_path', type=str,
                        default=r'C:\Users\DELL\Desktop\tmp',
                        # required=True,
                        )
    parser.add_argument('--temp_path', type=str, default=r'C:\Users\DELL\Desktop\tmp')
    parser.add_argument('--win_size', type=int, default=1024)
    parser.add_argument('--rpc_height', type=float, default=0.0)
    parser.add_argument('--rpc_dem', type=str, default=r'')
    parser.add_argument('--sensor', type=str, default='GF2')
    parser.add_argument('--qrj', type=str, default='')
    parser.add_argument('--dqsq', type=str, default='')
    parser.add_argument('--metaxml', type=str,
                        default=r'C:\LCJ\image_data\mosaic_img\GF2_PMS2_E110.3_N24.5_20220928_L1A0006781153\GF2_PMS2_E110.3_N24.5_20220928_L1A0006781153-MSS2.xml')
    parser.add_argument('--savepath', type=str, default=r'C:\Users\DELL\Desktop\tmp\opt2.tif')
    parser.add_argument('--CostTime', type=str, default='time')
    parser.add_argument('--message', type=str, default='EventProgress')
    opt = parser.parse_args()
    t1 = time.time()
    if os.path.exists(opt.json_path):
        with open(opt.json_path, 'r', encoding='utf-8-sig') as f:
            options = json.load(f)
        opt.ms_path = options['ms_path']
        opt.pan_path = options['pan_path']
        opt.method = options['method']
        try:
            opt.rpc_height = float(options['rpc_height'])
        except:
            pass
        try:
            opt.win_size = int(options['win_size'])
        except:
            pass
        if options['rpc_dem'] == '':
            pass
        else:
            opt.rpc_dem = options['rpc_dem']
        opt.sensor = options['sensor']

        opt.qrj = float(options['qrj'])
        opt.dqsq = float(options['dqsq'])
        opt.metaxml = options['metaxml']
        opt.savepath = options['savepath']
        opt.message = options['message']
        opt.CostTime = options['cost_time']

    global_message = opt.message
    sys.stdout.flush()
    print('--------options----------', flush=True)
    for k in list(vars(opt).keys()):
        print('%s: %s' % (k, vars(opt)[k]), flush=True)
    print('--------options----------\n', flush=True)

    # rpc
    print('rpc start')
    ms_rpc, _ = rpc_process(input_image=opt.ms_path, temp_path=opt.temp_path,
                            rpc_height=opt.rpc_height, rpc_dem=opt.rpc_dem)
    pan_rpc, _ = rpc_process(input_image=opt.pan_path, temp_path=opt.temp_path,
                             rpc_height=opt.rpc_height, rpc_dem=opt.rpc_dem)
    # pansharpen
    Methods = ['Brovey', 'IHS', 'GSA', 'GS', 'Wavelet', 'Brovey2']
    try:
        assert opt.method in Methods
    except:
        raise (str(opt.method) + '不在备选方法中，参考：' + str(Methods))

    image_sensor = sensors.getsensor(opt.sensor)(opt.metaxml)

    try:
        used_weights = weights[opt.sensor]
    except:
        used_weights = [1, 1, 1, 1]
    print(used_weights)

    outbands = 3
    cols, rows, ipt = dl.info(opt.ms_path)
    # hl, ll = arcgisopt(ipt)
    hl = [749, 693, 658, 658]
    ll = [245, 136, 110, 110]
    if opt.method == 'Brovey':
        print('gdal', flush=True)

        # 测试
        basename = os.path.basename(opt.ms_path)
        preffix, _ = os.path.splitext(basename)
        tmp_name = preffix + '_pansharpen.vrt'
        opt.output_path = os.path.join(opt.output_path, tmp_name)

        gdal_pansharpen(('pansharpen', pan_rpc, ms_rpc, opt.output_path,
                         '-w', str(used_weights[0]),
                         '-w', str(used_weights[1]),
                         '-w', str(used_weights[2]),
                         '-w', str(used_weights[3]),
                         '-r', 'near', '-of', 'VRT'),
                        )

        col, row, ds, os = dl.io_info(opt.output_path, opt.savepath,
                                      opt_bands=3,
                                      opt_driver='GTiff',
                                      opt_dtype=gdalconst.GDT_Byte)
        frames = dl.block_tif(col, row, 16)
        len_frames = len(frames)
        finish = 1
        for index, wins in enumerate(frames):
            windata = dl.block_reader(ds, wins)
            mask = windata == 0
            dl.block_writer(os, wins, gamma_trans(stretch(hl, ll, windata)[:, :, (2, 1, 0)], 0.825))
            print('{}:{}'.format(global_message, finish / len_frames), flush=True)
            finish += 1
    else:
        ms_img_resized = gdal_resample(ms_path=ms_rpc, pan_path=pan_rpc, temp_path=opt.temp_path)
        fuse_pan_ms(pan_path=pan_rpc, ms_path=ms_img_resized,
                    output_path=opt.savepath,
                    Method=opt.method,
                    weights=used_weights,
                    ms_win_size=opt.win_size)
        os.remove(ms_img_resized)

    t2 = time.time()
    print("{}:{}".format(opt.CostTime, time.time() - t1), flush=True)
    # shutil.rmtree(opt.output_path)
    # shutil.rmtree(opt.temp_path)
