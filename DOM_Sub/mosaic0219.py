import os.path
import shutil
from osgeo import gdal
from utils import IMAGE2, make_file, GenExtents, coord_ras2geo, coord_geo2ras, raster_mosaic
import cv2
import numpy as np
from time import time
import argparse
import json

NUM = 0
CURRENT = 0
MESSAGE = 'mosaic'
ONE_FLAG = False


def generate_overview(img_path, work_dir, pct=None, size=None, name=None):
    make_file(work_dir)
    basename = os.path.basename(img_path)
    preffix, _ = os.path.splitext(basename)
    if name is None:
        out_path = os.path.join(work_dir, preffix + '_overview.tif')
    else:
        out_path = os.path.join(work_dir, preffix + '_{}.tif'.format(name))
    if pct is not None:
        options = gdal.TranslateOptions(
            format='GTiff',
            heightPct=pct, widthPct=pct
        )
    elif size is not None:
        options = gdal.TranslateOptions(
            format='GTiff',
            width=size[0],
            height=size[1],
        )
    else:
        raise KeyError('para error')
    gdal.Translate(
        destName=out_path,
        srcDS=img_path,
        options=options
    )
    return out_path


def get_intersec_extent(img_path1, img_path2):
    img1 = IMAGE2()
    img2 = IMAGE2()
    img1.read_img(img_path1)
    img2.read_img(img_path2)

    x1_res, y1_res = img1.im_geotrans[1], img1.im_geotrans[5]
    x2_res, y2_res = img2.im_geotrans[1], img2.im_geotrans[5]

    # 输出参数
    # # 左上角坐标
    lx1_geo = img1.im_geotrans[0]
    ly1_geo = img1.im_geotrans[3]
    lx2_geo = img2.im_geotrans[0]
    ly2_geo = img2.im_geotrans[3]

    # 右下角坐标
    rx1_geo = lx1_geo + img1.im_width * x1_res
    ry1_geo = ly1_geo + img1.im_height * y1_res
    rx2_geo = lx2_geo + img2.im_width * x2_res
    ry2_geo = ly2_geo + img2.im_height * y2_res

    # 获取重叠区的四至范围，重叠区不能四舍五入
    lx0, ly0 = max(lx1_geo, lx2_geo), min(ly1_geo, ly2_geo)
    rx0, ry0 = min(rx1_geo, rx2_geo), max(ry1_geo, ry2_geo)
    x_res, y_res = x1_res, y2_res  # 分辨率
    width_ol = int((rx0 - lx0) / x_res)
    height_ol = int((ry0 - ly0) / y_res)
    geotrans = (lx0, x_res, 0.0, ly0, 0.0, y_res)

    # 重叠区影像栅格extent
    x1, y1 = coord_geo2ras(img1.im_geotrans, [lx0, ly0])
    x2, y2 = coord_geo2ras(img2.im_geotrans, [lx0, ly0])

    extent1_ol = [x1, y1, width_ol, height_ol]
    extent2_ol = [x2, y2, width_ol, height_ol]
    lu_p = [lx0, ly0]
    res = [x_res, y_res]
    return extent1_ol, extent2_ol, lu_p, res, geotrans


def generate_mosaic_map(img_path_list, opt):
    make_file(opt.work_dir)
    weight_map_list = []
    for i, img_path in enumerate(img_path_list):
        make_file(os.path.join(opt.work_dir, 'weight'))
        weight_map_path = os.path.join(opt.work_dir, 'weight', os.path.basename(img_path))
        overview = generate_overview(img_path=img_path,
                                     work_dir=os.path.join(opt.work_dir, 'overview'),
                                     pct=opt.ratio)
        ov = IMAGE2()
        ov.read_img(overview)
        ov.create_img(filename=weight_map_path, out_bands=1, datatype=gdal.GDT_Float32)
        img_patch = ov.get_extent([0, 0, ov.im_width, ov.im_height])[0, ...]
        imgDist = cv2.distanceTransform(img_patch, distanceType=cv2.DIST_L2, maskSize=5) / 255.
        ov.write_extent([0, 0, ov.im_width, ov.im_height], imgDist)
        del ov.output_dataset
        weight_map_list.append(weight_map_path)
    return weight_map_list


def generate_ov_weight(weight_map_list, opt):
    tmp_list = []
    ov_wieght_list = []
    for i, weight in enumerate(weight_map_list):
        if i == 0:
            tmp_list.append(weight)
        else:
            tmp_list.append(weight)
            wm1 = IMAGE2()
            wm1.read_img(tmp_list[i - 1])
            wm2 = IMAGE2()
            wm2.read_img(weight)
            extent1, extent2, lu_p, res, _ = get_intersec_extent(tmp_list[i - 1], tmp_list[i])

            wm1_patch = wm1.get_extent(extent1)
            wm2_patch = wm2.get_extent(extent2)
            mask = wm1_patch / (wm1_patch + wm2_patch + np.finfo(np.float32).eps)

            geotrans = (lu_p[0], res[0] * opt.ratio, 0., lu_p[1], 0., res[1] * opt.ratio)
            outname = os.path.join(opt.work_dir, 'weight\ov_weight.tif')
            wm1.create_img(outname,
                           datatype=gdal.GDT_Float32,
                           im_width=extent1[2], im_height=extent1[3],
                           out_bands=1,
                           im_geotrans=geotrans)
            wm1.write_extent([0, 0, extent1[2], extent1[3]], mask)
            del wm1.output_dataset
            ov_wieght_list.append(outname)

    return ov_wieght_list


def generate_overlap(img1_path, img2_path, weight_map, opt):
    basename = os.path.basename(img1_path)
    preffix1, _ = os.path.splitext(basename)
    basename = os.path.basename(img2_path)
    preffix2, _ = os.path.splitext(basename)
    outname = preffix1 + '_' + preffix2 + '.tif'
    output_ol = os.path.join(opt.work_dir, outname)

    img1 = IMAGE2()
    img2 = IMAGE2()
    img1.read_img(img1_path)
    img2.read_img(img2_path)
    extent1_ol, extent2_ol, lu_p, res, geotrans = get_intersec_extent(img_path1=img1_path, img_path2=img2_path)
    width_ol, height_ol = extent1_ol[2], extent1_ol[3]
    new_weight_map_path = generate_overview(weight_map,
                                            work_dir=opt.work_dir,
                                            size=[width_ol, height_ol])

    wm = IMAGE2()
    wm.read_img(new_weight_map_path)
    wm.create_img(output_ol,
                  im_geotrans=geotrans,
                  im_width=width_ol, im_height=height_ol,
                  out_bands=3)
    # 增加滑窗
    # if wm.im_width * wm.im_height > 51200 * 51200:
    if wm:
        extents_wm = GenExtents(wm.im_width, wm.im_height, win_size=opt.win_size)
        for i, extent_wm in enumerate(extents_wm):
            if ONE_FLAG:
                print('{}: {}'.format(MESSAGE, (CURRENT / NUM + i / len(extents_wm) * (1 / NUM)) * 1 / 3 + 2 / 3),
                      flush=True)
            else:
                print('{}: {}'.format(MESSAGE, CURRENT / NUM + i / len(extents_wm) * (1 / NUM)), flush=True)
            x0_ras, y0_ras, width, height = extent_wm
            tem_geotrans = wm.im_geotrans[0], res[0], wm.im_geotrans[2], \
                           wm.im_geotrans[3], wm.im_geotrans[4], res[1],

            x_geo, y_geo = coord_ras2geo(tem_geotrans, [x0_ras, y0_ras])

            x1_ras, y1_ras = coord_geo2ras(img1.im_geotrans, [x_geo, y_geo])
            extent1 = [x1_ras, y1_ras, width, height]
            x2_ras, y2_ras = coord_geo2ras(img2.im_geotrans, [x_geo, y_geo])
            extent2 = [x2_ras, y2_ras, width, height]

            ex1 = img1.get_extent(extent1).astype(np.float32)
            ex2 = img2.get_extent(extent2).astype(np.float32)
            weight_map = wm.get_extent(extent_wm)
            weight_map = np.round(weight_map, 2)

            ov_out = np.full_like(ex1, 0)
            for i in range(3):
                ov_out[i, ...] = weight_map * ex1[i, ...] + (1. - weight_map) * ex2[i, ...]
            wm.write_extent(extent_wm, ov_out)
    else:
        # 重叠区域
        ex1 = img1.get_extent(extent1_ol).astype(np.float32)
        ex2 = img2.get_extent(extent2_ol).astype(np.float32)
        weight_map = wm.get_extent([0, 0, wm.im_width, wm.im_height])
        weight_map = np.round(weight_map, 2)

        ov_out = np.full_like(ex1, 0)
        for i in range(3):
            ov_out[i, ...] = weight_map * ex1[i, ...] + (1. - weight_map) * ex2[i, ...]

        wm.write_extent([0, 0, wm.im_width, wm.im_height], ov_out)
        if ONE_FLAG:
            print('{}:{}'.format(MESSAGE, CURRENT / NUM), flush=True)
        else:
            print('{}:{}'.format(MESSAGE, CURRENT / NUM / 3 + 2 / 3), flush=True)
    return output_ol


def run_one_seamless_mosaic(img_path_list, output_path, opt):
    weight_map_list = generate_mosaic_map(img_path_list, opt)
    ov_weight_list = generate_ov_weight(weight_map_list, opt)

    weight_map = ov_weight_list[0]
    img1_path = img_path_list[0]
    img2_path = img_path_list[1]
    output_ol_path = generate_overlap(img1_path, img2_path, weight_map, opt)

    raster_mosaic(
        file_path_list=img_path_list + [output_ol_path],
        output_path=output_path,
    )
    return output_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser('cloud removal for remote sensing images')
    parser.add_argument('--json_path', type=str, default=r'parameters_mos.json')
    parser.add_argument('--img_path_list', type=list, default=None)
    parser.add_argument('--output_path', type=str, default=None)
    parser.add_argument('--work_dir', type=str, default=None)
    parser.add_argument('--message', type=str, default=r'message')
    parser.add_argument('--win_size', type=int, default=1024)
    parser.add_argument('--ratio', type=int, default=10)
    parser.add_argument('--one_flag', type=bool, default=False)
    opt = parser.parse_args()
    t0 = time()

    if os.path.exists(opt.json_path):
        with open(opt.json_path, 'r', encoding='utf-8-sig') as f:
            args = json.load(f)
            opt.img_path_list = args['img_path_list']
            opt.output_path = args['output_path']
            opt.work_dir = args['work_dir']
            opt.message = args['message']
            opt.win_size = args['win_size']
            opt.ratio = args['ratio']

    print('--------options----------', flush=True)
    for k in list(vars(opt).keys()):
        print('%s: %s' % (k, vars(opt)[k]), flush=True)
    print('--------options----------\n', flush=True)

    base_img = 0
    img_path_list = opt.img_path_list
    ONE_FLAG = opt.one_flag
    NUM = len(img_path_list) - 1
    for i, img in enumerate(img_path_list):
        if i == 0:
            base_img = img
        else:
            if i != len(img_path_list) - 1:
                t = time()
                tmp_out = os.path.join(opt.work_dir, '{}.tif'.format(t))
            else:
                tmp_out = opt.output_path
            run_one_seamless_mosaic(
                img_path_list=[base_img, img],
                output_path=tmp_out,
                opt=opt
            )
            base_img = tmp_out
            CURRENT += 1

    shutil.rmtree(opt.work_dir)
    if ONE_FLAG:
        print('{}: {}'.format(MESSAGE, 1.), flush=True)
    else:
        print('{}: {}'.format(MESSAGE, 1.), flush=True)
        print("CostTime:{}".format(time() - t0), flush=True)
