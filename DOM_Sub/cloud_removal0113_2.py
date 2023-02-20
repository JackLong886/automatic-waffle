import time
from osgeo import ogr, gdal
import os
import numpy as np
import cv2
from numpy import einsum
from osgeo import gdalconst
import argparse
gdal.SetConfigOption('SHAPE_ENCODING', 'gbk')
import json
from utils import IMAGE2, make_file, GenExtents, coord_ras2geo, coord_geo2ras
global start, end
global start2, end2


# 生成结合表
def union_shp(shp_path_list, out_dir):
    shp_num = len(shp_path_list)
    assert shp_num > 1
    driver = ogr.GetDriverByName('ESRI Shapefile')
    dataset0 = driver.Open(shp_path_list[0])
    layer0 = dataset0.GetLayerByIndex(0)
    srs0 = layer0.GetSpatialRef()
    defn0 = layer0.GetLayerDefn()

    basename = os.path.basename(shp_path_list[-1])
    make_file(out_dir)
    out_path = os.path.join(out_dir, basename)

    outds = driver.CreateDataSource(out_path)
    outlayer = outds.CreateLayer(out_path, srs=srs0, geom_type=ogr.wkbPolygon)

    FieldDefns = []
    for field_n in range(defn0.GetFieldCount()):
        FieldDefns.append(defn0.GetFieldDefn(field_n))
        outlayer.CreateField(defn0.GetFieldDefn(field_n))

    for i, path in enumerate(shp_path_list):
        shp = driver.Open(path)
        layer = shp.GetLayerByIndex(0)

        layer.ResetReading()

        for feature in layer:
            geom = feature.GetGeometryRef()
            out_feat = ogr.Feature(outlayer.GetLayerDefn())
            out_feat.SetGeometry(geom)
            for FieldDefn in FieldDefns:
                out_feat.SetField(FieldDefn.GetName(), feature.GetField(FieldDefn.GetName()))

            outlayer.CreateFeature(out_feat)
        shp.Destroy()
    dataset0.Destroy()
    outds.Destroy()


# 擦除
def erase_cloud_mask(to_erase, erase_list, erase_out_dir, inter_out_dir, vaild_out_dir, image_path_list):
    make_file(erase_out_dir)
    make_file(inter_out_dir)
    # make_file(vaild_out_dir)
    # 开始批量擦除
    erase_path_list = []
    inter_path_list = []
    vaild_path_list = []
    new_img_path_list = []
    driver = ogr.GetDriverByName('ESRI Shapefile')
    for i, path in enumerate(erase_list):
        basename = os.path.basename(path)

        # 被擦除shp
        to_erase_shp = driver.Open(to_erase)
        to_erase_layer = to_erase_shp.GetLayer()
        num_feature = to_erase_layer.GetFeatureCount()

        if num_feature == 0:
            break

        dst_erase = os.path.join(erase_out_dir, str(i) + '_erase_' + basename)
        erase_path_list.append(dst_erase)
        dst_inter = os.path.join(inter_out_dir, str(i) + '_inter_' + basename)
        inter_path_list.append(dst_inter)

        test_shp = driver.Open(path, 0)
        test_layer = test_shp.GetLayer()
        test_srs = test_layer.GetSpatialRef()
        test_defn = test_layer.GetLayerDefn()

        outds_inter = driver.CreateDataSource(dst_inter)
        outlayer_inter = outds_inter.CreateLayer(dst_inter, srs=test_srs, geom_type=ogr.wkbPolygon)

        outds_erase = driver.CreateDataSource(dst_erase)
        outlayer_erase = outds_erase.CreateLayer(dst_erase, srs=test_srs, geom_type=ogr.wkbPolygon)

        for j in range(test_defn.GetFieldCount()):
            outlayer_inter.CreateField(test_defn.GetFieldDefn(j))
            outlayer_erase.CreateField(test_defn.GetFieldDefn(j))

        # 获取擦除剩余和擦除部分
        if i == 0:
            to_erase_layer.Erase(test_layer, outlayer_erase)
            to_erase_layer.Intersection(test_layer, outlayer_inter)

            to_erase_shp.Destroy()
        else:
            tmp_shp = driver.Open(erase_path_list[i - 1], 1)
            tmp_layer = tmp_shp.GetLayer()
            tmp_feat_count = tmp_layer.GetFeatureCount()
            if tmp_feat_count == 0:
                break
            tmp_layer.Erase(test_layer, outlayer_erase)
            tmp_layer.Intersection(test_layer, outlayer_inter)
            tmp_shp.Destroy()

        # 不相交的不输出
        if outlayer_inter.GetFeatureCount() != 0:
            new_img_path_list.append(image_path_list[i])
        else:
            inter_path_list.pop()

        # 擦除完毕
        if outlayer_erase.GetFeatureCount() == 0:
            break

        # # 添加原始影像路径字段
        # dirname = os.path.dirname(image_path_list[i])
        # outlayer_inter.CreateField(
        #     ogr.FieldDefn('Path', ogr.OFTString)
        # )
        # for feat in outlayer_inter:
        #     feat.SetField('Path', dirname)
        #     outlayer_inter.SetFeature(feat)

    return inter_path_list, new_img_path_list


# 获取云掩膜shp
def get_mask_shp(path_list, new_dir, gridcode):
    driver = ogr.GetDriverByName('ESRI Shapefile')

    new_inter_path_list = []
    make_file(new_dir)
    for path in path_list:
        shp_name = os.path.basename(path)
        new_inter_path = os.path.join(new_dir, shp_name)
        new_inter_path_list.append(new_inter_path)

        shp = driver.Open(path, 0)
        layer = shp.GetLayer()
        srs = layer.GetSpatialRef()
        defn = layer.GetLayerDefn()

        # 创建新shp
        new_inter = driver.CreateDataSource(new_inter_path)
        new_inter_layer = new_inter.CreateLayer(new_inter_path, srs=srs, geom_type=ogr.wkbPolygon)
        for j in range(defn.GetFieldCount()):
            new_inter_layer.CreateField(defn.GetFieldDefn(j))

        index = new_inter_layer.GetLayerDefn().GetFieldIndex('Shape_Area')  # 获取字段的索引值
        fld_defn = ogr.FieldDefn('Shape_Area', ogr.OFTString)  # 创建新属性的字段定义
        fld_defn.SetWidth(100)
        new_inter_layer.AlterFieldDefn(index, fld_defn, ogr.ALTER_WIDTH_PRECISION_FLAG)

        layer.ResetReading()
        for feature in layer:
            gd = feature.GetField('gridcode')
            if gd == gridcode:
                new_inter_layer.CreateFeature(feature)

    return new_inter_path_list


def shp2tif(shp_path, ref_tif_path, target_tif_path, attribute_field=''):
    ref_tif_file = IMAGE2()
    ref_tif_file.read_img(ref_tif_path)
    ref_tif_file.create_img(
        filename=target_tif_path,
        im_width=ref_tif_file.im_width, im_height=ref_tif_file.im_height,
        im_proj=ref_tif_file.im_proj, im_geotrans=ref_tif_file.im_geotrans,
        out_bands=1,
        datatype=gdal.GDT_Byte
    )

    shp_file = ogr.Open(shp_path)
    shp_layer = shp_file.GetLayer()
    gdal.RasterizeLayer(
        dataset=ref_tif_file.output_dataset,
        bands=[1],
        layer=shp_layer,
        # options=[f"ATTRIBUTE={attribute_field}"]
    )
    del ref_tif_file.output_dataset


def crop_img(image_path_list, shp_path_list, output_dir):
    assert len(image_path_list) == len(shp_path_list)
    make_file(output_dir)
    output_path_list = []
    for image_path, shp_path in zip(image_path_list, shp_path_list):
        # print('start crop {} using {}'.format(image_path, shp_path))
        basename = os.path.basename(image_path)
        preffix, _ = os.path.splitext(basename)
        name = 'crop_' + preffix + '.tif'
        output_path = os.path.join(output_dir, name)

        options = gdal.WarpOptions(
            format='GTiff',
            cutlineDSName=shp_path,
            dstNodata=0,
            cropToCutline=True
        )
        datasets = gdal.Warp(
            output_path,
            image_path,
            options=options
        )
        if datasets is None:
            raise KeyError('crop error')

        output_path_list.append(output_path)
    return output_path_list

    # 还没做重采样,没做滑窗


def run_one_blend(bg_path, source_path, mask_path, out_path, flag='pie'):
    bg = IMAGE2()
    s = IMAGE2()
    m = IMAGE2()
    bg.read_img(bg_path)
    s.read_img(source_path)
    m.read_img(mask_path)
    bg.copy_image(filename=out_path)

    bg_statis = bg.compute_statistics()
    s_statis = s.compute_statistics()

    width, height = int(m.im_width), int(m.im_height)
    # pie
    # if flag == 'pie':
    #     # 处理区域左上角地理坐标和分辨率
    #     x, x_res, _, y, _, y_res = m.im_geotrans
    #
    # extent_m = [0, 0, width, height]
    # # 计算bg 的栅格位置
    # x_bg, x_bg_res, _, y_bg, _, y_bg_res = bg.im_geotrans
    # dx = int(round((x - x_bg) / x_bg_res))
    # dy = int(round((y - y_bg) / y_bg_res))
    # extent_bg = [dx, dy, width, height]
    #
    # # 计算source的栅格位置
    # x_s, x_s_res, _, y_s, _, y_s_res = s.im_geotrans
    # dx2 = int(round((x - x_s) / x_s_res))
    # dy2 = int(round((y - y_s) / y_s_res))
    # extent_s = [dx2, dy2, width, height]
    #
    # bg_patch = bg.get_extent(extent_bg)
    # s_patch = s.get_extent(extent_s)
    # m_patch = m.get_extent(extent_m)
    #
    # # 输入影像块进行去云
    # out_patch = pie_blend(bg_patch, s_patch, m_patch)
    # # 写出
    # bg.write2copy_image(extent=extent_bg, im_data=out_patch)
    # elif flag == 'cv2':
    #     if width * height >= 100000000:
    #         extents_m = GenExtents(width, height, win_size=4096)
    #         for extent_m in tqdm(extents_m):
    #             x_ras, y_ras, width, height = extent_m
    #             x_geo, y_geo = coord_ras2geo(m.im_geotrans, [x_ras, y_ras])
    #
    #             # 计算bg 的栅格位置
    #             x_bg, y_bg = coord_geo2ras(bg.im_geotrans, [x_geo, y_geo])
    #             extent_bg = [x_bg, y_bg, width, height]
    #
    #             # 计算source的栅格位置
    #             x_s, y_s = coord_geo2ras(s.im_geotrans, [x_geo, y_geo])
    #             extent_s = [x_s, y_s, width, height]
    #
    #             bg_patch = bg.get_extent(extent_bg)
    #             s_patch = s.get_extent(extent_s)
    #             m_patch = m.get_extent(extent_m)
    #
    #
    #             # 输入影像块进行去云
    #             out_patch = cv2_blend(bg_patch, s_patch, m_patch)
    #             # 写出
    #             bg.write2copy_image(extent=extent_bg, im_data=out_patch)
    #
    #     else:
    #         # 处理区域左上角地理坐标和分辨率
    #         x, x_res, _, y, _, y_res = m.im_geotrans
    #
    #         extent_m = [0, 0, width, height]
    #         # 计算bg 的栅格位置
    #         x_bg, x_bg_res, _, y_bg, _, y_bg_res = bg.im_geotrans
    #         dx = int(round((x - x_bg) / x_bg_res))
    #         dy = int(round((y - y_bg) / y_bg_res))
    #         extent_bg = [dx, dy, width, height]
    #
    #         # 计算source的栅格位置
    #         x_s, x_s_res, _, y_s, _, y_s_res = s.im_geotrans
    #         dx2 = int(round((x - x_s) / x_s_res))
    #         dy2 = int(round((y - y_s) / y_s_res))
    #         extent_s = [dx2, dy2, width, height]
    #
    #         bg_patch = bg.get_extent(extent_bg)
    #         s_patch = s.get_extent(extent_s)
    #         m_patch = m.get_extent(extent_m)
    #
    #         # 输入影像块进行去云
    #         out_patch = cv2_blend(bg_patch, s_patch, m_patch)
    #         # 写出
    #         bg.write2copy_image(extent=extent_bg, im_data=out_patch)
    if flag == 'map':
        # 处理区域左上角地理坐标和分辨率
        x, x_res, _, y, _, y_res = m.im_geotrans

        extent_m = [0, 0, width, height]
        # 计算bg 的栅格位置
        x_bg, x_bg_res, _, y_bg, _, y_bg_res = bg.im_geotrans
        dx = int(round((x - x_bg) / x_bg_res))
        dy = int(round((y - y_bg) / y_bg_res))
        extent_bg = [dx, dy, width, height]

        # 计算source的栅格位置
        x_s, x_s_res, _, y_s, _, y_s_res = s.im_geotrans
        dx2 = int(round((x - x_s) / x_s_res))
        dy2 = int(round((y - y_s) / y_s_res))
        extent_s = [dx2, dy2, width, height]

        bg_patch = bg.get_extent(extent_bg)
        s_patch = s.get_extent(extent_s)
        m_patch = m.get_extent(extent_m)

        # for i in range(bg_patch.shape[0]):
        #     bg_min, bg_max, bg_mean, bg_std = bg_statis[i]
        #     s_min, s_max, s_mean, s_std = s_statis[i]
        #     s_patch[i, :, :] = (s_patch[i, :, :] - s_min) / (bg_std / s_std) + bg_min
        #
        # assert 1==2
        out_patch = map_blend(bg_patch, s_patch, m_patch)

        # 写出
        bg.write2copy_image(extent=extent_bg, im_data=out_patch)
    elif flag == 'easy':
        # 处理区域左上角地理坐标和分辨率
        x, x_res, _, y, _, y_res = m.im_geotrans

        extent_m = [0, 0, width, height]
        # 计算bg 的栅格位置
        x_bg, x_bg_res, _, y_bg, _, y_bg_res = bg.im_geotrans
        dx = int(round((x - x_bg) / x_bg_res))
        dy = int(round((y - y_bg) / y_bg_res))
        extent_bg = [dx, dy, width, height]

        # 计算source的栅格位置
        x_s, x_s_res, _, y_s, _, y_s_res = s.im_geotrans
        dx2 = int(round((x - x_s) / x_s_res))
        dy2 = int(round((y - y_s) / y_s_res))
        extent_s = [dx2, dy2, width, height]

        bg_patch = bg.get_extent(extent_bg)
        s_patch = s.get_extent(extent_s)
        m_patch = m.get_extent(extent_m)

        # 输入影像块进行去云
        out_patch = easy_blend(bg_patch, s_patch, m_patch)
        # 写出
        bg.write2copy_image(extent=extent_bg, im_data=out_patch)


def map_blend(bg_patch, s_patch, m_patch):
    bg = einsum('ijk->jki', bg_patch)
    s = einsum('ijk->jki', s_patch)
    if len(m_patch.shape) == 3:
        m = einsum('ijk->jki', m_patch)
    else:
        m = np.stack([m_patch, m_patch, m_patch], axis=-1)
    t = time.time()
    s_ori = s

    s = s.astype(np.float64)
    m = m.astype(np.float64)
    bg = bg.astype(np.float64)

    kernel_size = 30
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    m = cv2.dilate(m, kernel, iterations=5)

    kernel_size = 100
    kernel = np.ones((kernel_size, kernel_size)) / (kernel_size * kernel_size * 1.0)
    m = cv2.filter2D(m, -1, kernel, borderType=cv2.BORDER_ISOLATED) / 255.

    # s的nodata不参与计算
    m[s_ori == 0] = 0

    m2 = 1. - m
    s2 = s * m
    bg2 = bg * m2
    out = bg2 + s2

    # bg的nodata不输出
    out[bg == 0.] = 0.

    out = einsum('ijk->kij', out)
    return out


def easy_blend(bg_patch, s_patch, m_patch):
    bg = einsum('ijk->jki', bg_patch)
    s = einsum('ijk->jki', s_patch)
    if len(m_patch.shape) == 3:
        m = einsum('ijk->jki', m_patch)
    else:
        m = np.stack([m_patch, m_patch, m_patch], axis=-1)

    s = s.astype(np.float64)
    m = m.astype(np.float64) / 255.
    bg = bg.astype(np.float64)

    m2 = 1. - m
    s2 = s * m
    bg2 = bg * m2
    out = bg2 + s2

    out = einsum('ijk->kij', out)

    return out


def cloud_removal(shp_bg, img_bg, image_path_list, shp_path_list, work_dir, method):
    # 拆分shp 为有效区和云区
    shp_path_list = get_mask_shp(path_list=shp_path_list,
                                 new_dir=os.path.join(work_dir, 'source_shp'),
                                 gridcode=1)

    shp_vaild_bg = get_mask_shp(
        path_list=[shp_bg],
        new_dir=os.path.join(work_dir, 'bg_vaild_shp'),
        gridcode=1
    )[0]

    shp_bg = get_mask_shp(path_list=[shp_bg],
                          new_dir=os.path.join(work_dir, 'bg_cloud_shp'),
                          gridcode=2
                          )[0]

    # 判断是否相交

    # 开始擦除
    inter_path_list, new_img_path_list = erase_cloud_mask(
        to_erase=shp_bg,
        erase_list=shp_path_list,
        erase_out_dir=os.path.join(work_dir, 'erase'),
        inter_out_dir=os.path.join(work_dir, 'inter'),
        vaild_out_dir=os.path.join(work_dir, 'vaild'),
        image_path_list=image_path_list)

    if len(inter_path_list) == 0:
        return 0

    # 生成结合表
    union_shp_list = inter_path_list[:]
    union_shp_list.append(shp_vaild_bg)

    union_shp(
        shp_path_list=union_shp_list,
        out_dir=os.path.join(work_dir, 'union')
    )

    # 裁剪公共区域
    crop_path_list = crop_img(
        new_img_path_list,
        inter_path_list,
        output_dir=os.path.join(work_dir, 'crop'),
    )

    # 获取栅格mask
    assert len(crop_path_list) == len(inter_path_list)
    mask_path_list = []
    for crop_path, inter_path in zip(crop_path_list, inter_path_list):
        name = 'mask_' + os.path.basename(inter_path)[:1] + '.tif'
        output_mask_path = os.path.join(work_dir, name)
        shp2tif(
            shp_path=inter_path,
            ref_tif_path=crop_path,
            target_tif_path=output_mask_path
        )
        mask_path_list.append(output_mask_path)
    # print(mask_path_list)

    outpath_list = []
    outpath = None
    for j, (image_path, mask_path) in enumerate(zip(new_img_path_list, mask_path_list)):
        path = os.path.join(work_dir, 'result')
        preffix, _ = os.path.splitext(os.path.basename(image_path))
        t1 = time.time()
        name = method + '_' + str(t1) + '.tif'
        outpath = os.path.join(path, name)
        outpath_list.append(outpath)

        tmp_num = len(new_img_path_list)
        global start2, end2
        start2 = start + (j / tmp_num) * (end - start)
        end2 = start + ((j + 1) / tmp_num) * (end - start)

        if j == 0:
            run_one_blend2(bg_path=img_bg,
                           source_path=image_path,
                           mask_path=mask_path,
                           out_path=outpath,
                           flag=method,
                           )
        if j != 0:
            img_bg = outpath_list[j - 1]
            run_one_blend2(bg_path=img_bg,
                           source_path=image_path,
                           mask_path=mask_path,
                           out_path=outpath,
                           flag=method,
                           )
            os.remove(outpath_list[j - 1])
    return outpath_list[-1]


def map_blend2(bg_patch, s_patch, m_patch):
    bg = einsum('ijk->jki', bg_patch)
    s = einsum('ijk->jki', s_patch)
    if len(m_patch.shape) == 3:
        weight_map = einsum('ijk->jki', m_patch)
    else:
        weight_map = np.stack([m_patch, m_patch, m_patch], axis=-1)

    tmp = np.where(s == 0)
    if len(tmp[0]) != 0:
        weight_map[tmp] = 0

    weight_map = weight_map.astype(np.float64)
    s = s.astype(np.float64)
    bg = bg.astype(np.float64)

    m2 = 1. - weight_map
    s2 = s * weight_map
    bg2 = bg * m2
    out = bg2 + s2

    # bg的nodata不输出
    out[bg == 0.] = 0.
    # out = out * np.array(bg, dtype=bool)
    out = einsum('ijk->kij', out)
    return out


def run_one_blend2(bg_path, source_path, mask_path, out_path, win_size=1024, flag='pie'):
    bg = IMAGE2()
    s = IMAGE2()
    m = IMAGE2()
    bg.read_img(bg_path)
    s.read_img(source_path)
    m.read_img(mask_path)
    bg.copy_image(filename=out_path)

    extent_m = [0, 0, m.im_width, m.im_height]
    m_patch = m.get_extent(extent_m)
    weight_map = generate_map(m_patch)

    if m.im_width * m.im_height >= 20480 * 20480:
        extents_m = GenExtents(m.im_width, m.im_height, win_size=win_size)
        for i, extent_m in enumerate(extents_m):
            x_ras, y_ras, width, height = extent_m
            x_geo, y_geo = coord_ras2geo(m.im_geotrans, [x_ras, y_ras])
            # 计算bg 的栅格位置
            x_bg, y_bg = coord_geo2ras(bg.im_geotrans, [x_geo, y_geo])
            extent_bg = [x_bg, y_bg, width, height]
            # 计算source的栅格位置
            x_s, y_s = coord_geo2ras(s.im_geotrans, [x_geo, y_geo])
            extent_s = [x_s, y_s, width, height]

            bg_patch = bg.get_extent(extent_bg)
            s_patch = s.get_extent(extent_s)

            m_patch = weight_map[y_ras:y_ras + height, x_ras:x_ras + width]
            # 输入影像块进行去云
            out_patch = map_blend2(bg_patch, s_patch, m_patch)
            # 写出
            bg.write2copy_image(extent=extent_bg, im_data=out_patch)

            current = start2 + (i / len(extents_m)) * (end2 - start2)
            print('{}:{:.4f}'.format(message, current), flush=True)
    else:
        # 处理区域左上角地理坐标和分辨率
        x, x_res, _, y, _, y_res = m.im_geotrans

        extent_m = [0, 0, m.im_width, m.im_height]
        # 计算bg 的栅格位置
        x_bg, x_bg_res, _, y_bg, _, y_bg_res = bg.im_geotrans
        dx = int(round((x - x_bg) / x_bg_res))
        dy = int(round((y - y_bg) / y_bg_res))
        extent_bg = [dx, dy, m.im_width, m.im_height]

        # 计算source的栅格位置
        x_s, x_s_res, _, y_s, _, y_s_res = s.im_geotrans
        dx2 = int(round((x - x_s) / x_s_res))
        dy2 = int(round((y - y_s) / y_s_res))
        extent_s = [dx2, dy2, m.im_width, m.im_height]

        bg_patch = bg.get_extent(extent_bg)
        s_patch = s.get_extent(extent_s)
        m_patch = m.get_extent(extent_m)
        out_patch = map_blend(bg_patch, s_patch, m_patch)

        # 写出
        bg.write2copy_image(extent=extent_bg, im_data=out_patch)

        current = end2
        print('{}:{}'.format(message, current), flush=True)


def generate_map(m_patch):
    kernel_size = 10
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    weight_map = cv2.dilate(m_patch, kernel, iterations=5)

    kernel_size = 60
    kernel = np.ones((kernel_size, kernel_size)) / (kernel_size * kernel_size * 1.0)
    weight_map = cv2.filter2D(weight_map, -1, kernel, borderType=cv2.BORDER_REPLICATE) / 255.
    return weight_map


if __name__ == '__main__':
    parser = argparse.ArgumentParser('cloud removal for remote sensing images')
    parser.add_argument('--json_path', type=str, default=r'parameters.json')
    opt = parser.parse_args()
    if os.path.exists(opt.json_path):
        with open(opt.json_path, 'r', encoding='utf-8-sig') as f:
            args = json.load(f)
            input_image_path_list = args['input_image_path_list']
            input_shp_path_list = args['input_shp_path_list']
            work_dir = args['work_dir']
            message = args['message']

    t0 = time.time()

    if len(input_image_path_list) == 0 or len(input_shp_path_list) == 0:
        raise KeyError('请输入影像或shp文件')
    if len(input_image_path_list) != len(input_shp_path_list):
        raise KeyError('影像文件与矢量文件数目不一致')

    num = len(input_image_path_list)
    make_file(work_dir)
    method = 'map'
    # 去云
    cloud_removal_path_list = []
    for i in range(num):

        start = i / num
        end = (i + 1) / num

        shp_path_list = input_shp_path_list[:]
        shp_bg = shp_path_list.pop(i)
        image_path_list = input_image_path_list[:]
        img_bg = image_path_list.pop(i)

        img_cloud_removal = cloud_removal(
            shp_bg=shp_bg,
            img_bg=img_bg,
            shp_path_list=shp_path_list,
            image_path_list=image_path_list,
            method=method,
            work_dir=work_dir
        )
        if img_cloud_removal != 0:
            cloud_removal_path_list.append(img_cloud_removal)

    print("CostTime:{}".format(time.time() - t0))
