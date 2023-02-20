'''
删除临时文件，修复进度条错误
'''
import argparse
import json
import os
import shutil
from time import time
import cv2
import numpy as np
from osgeo import gdal
from osgeo import gdal_array
from osgeo import ogr
from tqdm import tqdm

MESSAGE = '1'
IMG_CURRENT = 0
IMG_NUM = 0
ONE_FLAG = False

LayerName = 'polygon'
DriverName = 'ESRI Shapefile'
FieldName = 'DN'

SCALE = 1000
SRC_NODATA = 0  # 源nodata
DST_NODATA = 0  # 目标nodata

ALL_NODATA = 0  # 全部是nodata
HAS_NODATA = 1  # 有部分nodata
NOT_NODATA = 2  # 没有nodata


# 配置文件
class ConfigObject:
    def __init__(self, d):
        self.__dict__ = d


# 测试掩膜数据标签
def test_flag(mask):
    height, width = mask.shape
    count = np.count_nonzero(mask)
    if count == 0:
        return NOT_NODATA
    elif count == height * width:
        return ALL_NODATA
    else:
        return HAS_NODATA


# 计算nodata
class NodataDet:
    def __init__(self, nodata):
        self.nodata = nodata

    def __call__(self, data):
        masks = data == self.nodata
        ret_mask = None
        for mask in masks:
            if ret_mask is None:
                ret_mask = mask
            else:
                ret_mask &= mask
        return ret_mask


# 读取器
class Reader:
    def __init__(self, dataset, bands, transform):
        self.dataset = dataset
        self.bands = bands
        self.nodet = NodataDet(SRC_NODATA)
        self.transform = transform

    def read(self, extent, outdat=None):
        x, y, xsize, ysize = extent
        if outdat is None:
            bandcount = len(self.bands)
            outdat = np.zeros((bandcount, ysize, xsize), dtype=np.float32)

        for i, band in enumerate(self.bands):
            band.ReadAsArray(x, y, xsize, ysize, buf_obj=outdat[i])

        mask = self.nodet(outdat)
        flag = test_flag(mask)
        if flag != ALL_NODATA:
            self.transform(outdat)

        return outdat, mask, flag


class Dilate:
    def __init__(self, kernel_size):
        self.kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (kernel_size, kernel_size))

    def __call__(self, img):
        cv2.dilate(img, self.kernel, dst=img, iterations=1)
        return img


class PostProcess:
    def __init__(self, kernel_size=3, threshold=0.5):
        self.dilate = Dilate(kernel_size)
        self.threshold = threshold

    def __call__(self, image):
        img = image[1]
        hieght, width = img.shape
        cls = img.view(np.uint8)
        cls = cls[0:hieght, 0:width]
        np.greater(img, self.threshold, out=cls)
        self.dilate(cls)
        cls += 1
        return cls


class BaseWriter:
    def write(self, image, extent, mask, flag):
        pass

    def end_write(self):
        pass

    def write_batch(self, images, extents, masks, flags):
        batchsize = len(extents)
        for i in range(batchsize):
            extent = extents[i]
            mask = masks[i]
            flag = flags[i]
            image = images[i]
            self.write(image, extent, mask, flag)


# 写入器
class Writer(BaseWriter):
    def __init__(self, dataset, bands):
        self.dataset = dataset
        self.bands = bands
        self.pprocess = PostProcess(kernel_size=3, threshold=0.5)

    def write(self, image, extent, mask, flag):
        data = self.pprocess(image)
        x, y, xsize, ysize = extent

        if flag == HAS_NODATA:
            data[mask] = DST_NODATA
        band = self.bands[0]
        band.WriteArray(data, x, y)


# 写入器
class MeanWriter(BaseWriter):
    def __init__(self, dataset, bands):
        self.dataset = dataset
        self.bands = bands
        rows = dataset.RasterYSize
        cols = dataset.RasterXSize
        self.prob = np.zeros((rows, cols), dtype=np.uint16)
        self.count = np.zeros((rows, cols), dtype=np.uint8)

    def write(self, image, extent, mask, flag):
        data = image[1]
        x, y, xsize, ysize = extent
        prob = self.prob[y:y + ysize, x:x + xsize]
        count = self.count[y:y + ysize, x:x + xsize]
        temp = data.view(np.uint16)[0:ysize, 0:xsize]
        np.multiply(data, SCALE, out=temp, casting='unsafe')
        # data *= SCALE
        # temp = data.astype(np.uint16)
        prob += temp
        count += 1
        if flag != NOT_NODATA:
            count[mask] = 0

    def end_write(self):
        mask = self.count.view(bool)
        prob = np.floor_divide(self.prob,
                               self.count, out=self.prob, where=mask, dtype=None)
        mid = SCALE / 2 - 1
        np.greater(prob, mid, out=prob)
        prob += 1
        np.equal(mask, False, out=mask)
        prob[mask] = DST_NODATA
        band = self.bands[0]
        band.WriteArray(prob, 0, 0)


# 归一化
class Normalize:
    def __init__(self, mean, std):
        mean = np.array(mean, dtype=np.float32)
        std = np.array(std, dtype=np.float32)
        self.mean = mean[:, None, None]
        self.std = std[:, None, None]

    def __call__(self, image):
        image -= self.mean
        image /= self.std
        return image


# opencv dnn
class OpenCVModel:
    def __init__(self, net):
        self.net = net

    def postprocess(self, outputs, batchsize):
        images = []
        batchsize = batchsize
        for i in range(batchsize):
            data = outputs[i]
            images.append(data)
        return images

    def __call__(self, blob, batchsize):
        self.net.setInput(blob)
        outputs = self.net.forward()
        return self.postprocess(outputs, batchsize)


def make_file(path):
    if not os.path.exists(path):
        os.makedirs(path)


# 通过文件后缀获取驱动名
def get_raster_drivername(filename):
    ext = os.path.splitext(filename)[1]
    ext = ext.lower()
    driver = {'.tif': 'GTiff', '.img': 'HFA', '.png': 'PNG', '.jpg': 'JPEG', '.gif': 'GIF', '.dat': 'ENVI',
              '.hdr': 'ENVI'}
    if ext in driver.keys():
        return driver[ext]
    else:
        raise Exception('Not supported format: ' + ext)


# 创建栅格数据集
def create_raster(out_file, width, height, bandcount, datatype):
    driver_name = get_raster_drivername(out_file)
    driver = gdal.GetDriverByName(driver_name)
    out_dataset = driver.Create(out_file, width, height, bandcount, datatype)
    if not out_dataset:
        raise Exception('can not create dataset: ' + out_file)

    return out_dataset


# 创建矢量
def create_shp(shpfile, srs, field_name):
    drv = gdal.GetDriverByName(DriverName)
    if drv is None:
        raise Exception('%s driver not available.' % DriverName)

    # ds = drv.CreateDataSource(shpfile)
    ds = drv.Create(shpfile, 0, 0, 0, gdal.GDT_Unknown)
    if ds is None:
        raise Exception('Creation of output file failed.')

    lyr = ds.CreateLayer(field_name, geom_type=ogr.wkbPolygon, srs=srs)

    if lyr is None:
        raise Exception('Creating %s layer failed.' % field_name)

    fd = ogr.FieldDefn(FieldName, ogr.OFTInteger)
    lyr.CreateField(fd)

    # 必须把数据集一起返回，不然会被释放
    return ds


# 生成处理范围
def gen_extents(width, height, tile_width, tile_height, x_stride=0, y_stride=0, is_vaild=False):
    if x_stride > tile_width:
        raise Exception(f'x_stride({x_stride}) > tile_width({tile_width})')
    if x_stride > tile_width:
        raise Exception(f'y_stride({y_stride}) > tile_height({tile_height})')
    if tile_width > width:
        raise Exception(f'tile_width({tile_width}) > width({width})')
    if tile_height > height:
        raise Exception(f'tile_height({tile_height}) > height({height})')

    if x_stride == 0:
        x_stride = tile_width
    if y_stride == 0:
        y_stride = tile_height

    extents = []
    y = 0
    x = 0
    y_begin = 0
    x_begin = 0
    x_vaild = 0
    y_vaild = 0
    x_end = False
    y_end = False
    while y < height:
        if y + tile_height >= height:
            if is_vaild:
                y_begin = y
                y_vaild = height - y
            else:
                y_begin = height - tile_height
                y_vaild = tile_height
            y_end = True
        else:
            y_begin = y
            y_vaild = tile_height
            y_end = False
        while x < width:
            if x + tile_width >= width:
                if is_vaild:
                    x_begin = x
                    x_vaild = width - x
                else:
                    x_begin = width - tile_width
                    x_vaild = tile_width
                x_end = True
            else:
                x_begin = x
                x_vaild = tile_width
                x_end = False
            extents.append((x_begin, y_begin, x_vaild, y_vaild))
            x += x_stride
            if x_end: break
        y += y_stride
        x = 0
        if y_end: break
    return extents


# 进行规划
def compute_batchs(batchsize, extents):
    batchs = []
    batch = []
    for extent in extents:
        if len(batch) == batchsize:
            batchs.append(batch)
            batch = [extent]
        else:
            batch.append(extent)

    if len(batch) > 0:
        batchs.append(batch)
    return batchs


# 尝试进行纠正
def try_warp(dataset, cellsize):
    geot = dataset.GetGeoTransform()
    xres = abs(geot[1])
    if abs(cellsize - xres) < cellsize / 4:
        return dataset
    opts = {}
    opts['targetAlignedPixels'] = True
    opts['multithread'] = True  # 是否多线程
    # opts['resampleAlg'] = 'bilinear'
    opts['errorThreshold'] = cellsize
    opts['copyMetadata'] = True
    opts['format'] = 'VRT'
    opts['xRes'] = cellsize
    opts['yRes'] = cellsize
    newds = gdal.Warp('', dataset, **opts)
    if newds is None:
        raise Exception('create warp dataset error.')
    return newds


# 创建读取器
def create_reader(file, mean, std, bandlist, cellsize=4):
    dataset = gdal.Open(file)
    if dataset is None:
        raise Exception('dataset open error: ' + file)

    srs = dataset.GetSpatialRef()
    if srs.IsProjected() == False:
        raise Exception('dataset.SpatialRef is not Projected.')

    dataset = try_warp(dataset, cellsize)
    bands = []
    for index in bandlist:
        bands.append(dataset.GetRasterBand(index))

    transform = Normalize(mean=mean, std=std)
    return Reader(dataset, bands, transform)


# 创建写入器
def create_writer(file, ref_dataset, bandcount, datatype=gdal.GDT_Byte, writer_type='Writer'):
    width = ref_dataset.RasterXSize
    height = ref_dataset.RasterYSize
    out_dataset = create_raster(file, width, height, bandcount, datatype)
    geotransform = ref_dataset.GetGeoTransform()
    srs = ref_dataset.GetSpatialRef()  # 原始空间参考
    if geotransform:
        out_dataset.SetGeoTransform(geotransform)
    if srs:
        out_dataset.SetSpatialRef(srs)
    bands = []
    for index in range(bandcount):
        bands.append(out_dataset.GetRasterBand(index + 1))
    if writer_type == 'Writer':
        return Writer(out_dataset, bands)
    elif writer_type == 'MeanWriter':
        return MeanWriter(out_dataset, bands)
    else:
        raise Exception('unknow writer type: ' + writer_type)


# 使用opencv dnn模块加载onnx模型
def load_cv_net(model_file, backendId, targetId):
    if os.path.isfile(model_file) == False:
        raise Exception('model file is not exist: ' + model_file)
    net = cv2.dnn.readNet(model_file)
    net.setPreferableBackend(backendId)
    net.setPreferableTarget(targetId)
    return net


# 构建推理引擎
def create_engine(model_file, backend=0, target=0):
    if model_file.endswith('.onnx'):
        net = load_cv_net(model_file, backend, target)
        model = OpenCVModel(net)
    else:
        raise Exception('unknow model file ' + model_file)
    return model


def create_buffer(batchsize, channels, height, width):
    buffer = np.zeros((batchsize, channels, height, width), dtype=np.float32)
    buffers = []
    for i in range(batchsize):
        buffers.append(buffer[i])
    return buffer, buffers


def read_data(reader, batchsize, windows, tile_height, tile_width):
    channels = reader.dataset.RasterCount
    blob1, buffers1 = create_buffer(batchsize, channels, tile_height, tile_width)
    blob2, buffers2 = create_buffer(batchsize, channels, tile_height, tile_width)

    use_first = True
    count = len(windows)
    it = iter(windows)

    ibatch = 0
    iextent = 0
    buffers = None
    blob = None

    while True:
        if use_first:
            blob = blob1
            buffers = buffers1
            use_first = False
        else:
            blob = blob2
            buffers = buffers2
            use_first = True

        ibatch = 0
        masks = []
        flags = []
        extents = []

        while ibatch < batchsize and iextent < count:
            extent = next(it)
            iextent += 1
            if ONE_FLAG:
                print('{}:{:.4f}'.format(MESSAGE, (iextent / count / IMG_NUM + IMG_CURRENT / IMG_NUM) / 3), flush=True)
            else:
                print('{}:{:.4f}'.format(MESSAGE, iextent / count / IMG_NUM + IMG_CURRENT / IMG_NUM), flush=True)
            imgdat, mask, flag = reader.read(extent, buffers[ibatch])
            if flag != ALL_NODATA:
                masks.append(mask)
                flags.append(flag)
                extents.append(extent)
                ibatch += 1

        if ibatch == 0:
            break
        else:
            yield blob, extents, masks, flags
    yield None


# 推理
def infer(engine, read_gen, writer):
    while True:
        result = next(read_gen)
        if result is None:
            break
        blob, extents, masks, flags = result
        batchsize = len(extents)
        outputs = engine(blob, batchsize)
        writer.write_batch(outputs, extents, masks, flags)
    writer.end_write()


# 栅格转矢量
def polygonize(outshp, data, srs, geot, nodata, field_name):
    # 创建有效范围数据集
    img_ds = gdal_array.OpenNumPyArray(data, True)
    band = img_ds.GetRasterBand(1)

    # 设置数据集参数
    img_ds.SetSpatialRef(srs)
    img_ds.SetGeoTransform(geot)
    band.SetNoDataValue(nodata)

    # 创建矢量，有效范围矢量和云范围矢量
    shpds = create_shp(outshp, srs, field_name)
    lyr = shpds.GetLayer(0)

    # 栅格转矢量
    maskband = None
    dst_field = 0
    options = ['8CONNECTED=8']
    err = gdal.Polygonize(band, maskband, lyr, dst_field, options, callback=None)
    if err != 0:
        raise Exception('polygonize error.')

    # 删除nodata的矢量
    sql = field_name + '=0'
    delete_features(shpds, sql)
    lyr = None

    return shpds


# 转矢量并删除背景图斑
def to_vector(outshp, img_data, srs, geot, nodata):
    vector_ds = polygonize(outshp, img_data, srs, geot, nodata, FieldName)
    sql = FieldName + '=0'
    delete_features(vector_ds, sql)
    return vector_ds


# 创建有效区图层
def create_vaild(outshp, src_data, srs, geot, nodata):
    img_data = (src_data != 0).astype(np.uint8)
    return to_vector(outshp, img_data, srs, geot, nodata)


# 创建云层图层
def create_cloud(outshp, src_data, srs, geot, nodata):
    img_data = (src_data == 2).astype(np.uint8)
    return to_vector(outshp, img_data, srs, geot, nodata)


# 简化图层
def simplify(dataset, tolerance=30):
    layer = dataset.GetLayer(0)
    layer.ResetReading()
    layer.SetAttributeFilter(None)
    for feature in layer:
        geometry = feature.GetGeometryRef()
        geometry = geometry.Simplify(tolerance)
        feature.SetGeometryDirectly(geometry)
        layer.SetFeature(feature)


# 根据面积删除
def delete_by_area(dataset, min_area=30):
    layer = dataset.GetLayer(0)
    layer.ResetReading()
    layer.SetAttributeFilter(None)
    delete_list = []
    for feature in layer:
        geometry = feature.GetGeometryRef()
        if geometry is None:
            delete_list.append(feature.GetFID())
            continue
        if geometry.GetArea() < min_area:
            fid = feature.GetFID()
            delete_list.append(fid)

    delete_list.reverse()
    for fid in delete_list:
        layer.DeleteFeature(fid)


# 缓存区分析
def buffer_analysis(dataset, distance, quadsecs=30):
    layer = dataset.GetLayer(0)
    layer.ResetReading()
    layer.SetAttributeFilter(None)
    for feature in layer:
        geometry = feature.GetGeometryRef()
        geometry = geometry.Buffer(distance, quadsecs)
        feature.SetGeometryDirectly(geometry)
        layer.SetFeature(feature)


# 消除空洞
def eliminate(dataset, area=0, percent=0):
    if area <= 0 and percent <= 0:
        return
    layer = dataset.GetLayer(0)
    layer.ResetReading()
    layer.SetAttributeFilter(None)
    remove_list = []
    for feature in layer:
        geometry = feature.GetGeometryRef()
        count = geometry.GetGeometryCount()
        if count == 1:
            continue
        remove_list.clear()
        total_area = 0
        for i in range(count):
            if i == 0:
                first_geom = geometry.GetGeometryRef(i)
                continue
            sub_geom = geometry.GetGeometryRef(i)
            sub_area = sub_geom.GetArea()
            if area > 0 and percent <= 0:
                if sub_area < area:
                    remove_list.append(i)
            elif area <= 0 and percent > 0:
                if total_area == 0:
                    total_area = first_geom.GetArea()
                if sub_area / total_area < percent:
                    remove_list.append(i)
            else:
                if sub_area < area:
                    remove_list.append(i)
                    continue
                if total_area == 0:
                    total_area = first_geom.GetArea()
                if sub_area / total_area < percent:
                    remove_list.append(i)

        if len(remove_list) > 0:
            remove_list.reverse()
            for i in remove_list:
                geometry.RemoveGeometry(i)
            layer.SetFeature(feature)


# 多部件消除小部件
def eliminate_small_part(dataset, area=0):
    if area <= 0:
        return
    layer = dataset.GetLayer(0)
    layer.ResetReading()
    layer.SetAttributeFilter(None)
    remove_list = []
    for feature in layer:
        geometry = feature.GetGeometryRef()
        count = geometry.GetGeometryCount()
        if count == 1:
            continue
        remove_list.clear()
        for i in range(count):
            sub_geom = geometry.GetGeometryRef(i)
            sub_area = sub_geom.GetArea()
            if sub_area < area:
                remove_list.append(i)

        if len(remove_list) > 0:
            remove_list.reverse()
            for i in remove_list:
                geometry.RemoveGeometry(i)
            layer.SetFeature(feature)


# 将多部件中的小部件移动至另一个图层
def move_part(from_ds, to_ds, area=0):
    if area <= 0:
        return
    from_layer = from_ds.GetLayer(0)
    from_layer.ResetReading()
    from_layer.SetAttributeFilter(None)

    to_layer = to_ds.GetLayer(0)
    to_layer.ResetReading()
    to_layer.SetAttributeFilter(None)
    feature_defn = to_layer.GetLayerDefn()

    remove_list = []
    for feature in from_layer:
        geometry = feature.GetGeometryRef()
        count = geometry.GetGeometryCount()
        if count == 1:
            continue
        remove_list.clear()
        for i in range(count):
            sub_geom = geometry.GetGeometryRef(i)
            sub_area = sub_geom.GetArea()
            if sub_area < area:
                remove_list.append((i, sub_geom))

        if len(remove_list) > 0:
            remove_list.reverse()
            for i, geom in remove_list:
                new_feature = ogr.Feature(feature_defn)
                new_feature.SetGeometry(geom)  #
                to_layer.CreateFeature(new_feature)

            for i, _ in remove_list:
                geometry.RemoveGeometry(i)
            from_layer.SetFeature(feature)


# 删除要素
def delete_features(dataset, where=None):
    layer = dataset.GetLayer(0)
    layer.ResetReading()
    layer.SetAttributeFilter(where)
    for feature in layer:
        fid = feature.GetFID()
        layer.DeleteFeature(fid)

    # sql = 'REPACK ' + layer.GetName()
    # dataset.ExecuteSQL(sql)


# 裁切分析
def erase_analysis(src_ds, method_ds, result_ds):
    src_layer = src_ds.GetLayer(0)
    method_layer = method_ds.GetLayer(0)
    result_layer = result_ds.GetLayer(0)
    option = ["PROMOTE_TO_MULTI=YES"]
    src_layer.Erase(method_layer, result_layer, option)


# 云检测生成云掩膜栅格
def cloudmask(cfg):
    if isinstance(cfg, dict):
        cfg = ConfigObject(cfg)
    engine = create_engine(cfg.model_file)
    reader = create_reader(cfg.in_raster, cfg.mean, cfg.std, cfg.bandlist, cfg.cellsize)

    x_stride = cfg.x_stride
    y_stride = cfg.y_stride
    tile_width = cfg.tile_width
    tile_height = cfg.tile_height

    if x_stride != tile_width or y_stride != tile_height:
        writer_type = 'MeanWriter'
    else:
        writer_type = 'Writer'
    writer = create_writer(cfg.out_label, reader.dataset, 1, writer_type=writer_type)

    raster_width = writer.dataset.RasterXSize
    raster_height = writer.dataset.RasterYSize
    batchsize = cfg.batchsize
    windows = gen_extents(raster_width, raster_height,
                          tile_width, tile_height, x_stride, y_stride)

    if cfg.quiet:
        read_gen = read_data(reader, batchsize, windows, tile_height, tile_width)
        infer(engine, read_gen, writer)
    else:
        with tqdm(windows, desc="infer", ascii=True, ncols=110) as pbar:
            read_gen = read_data(reader, batchsize, pbar, tile_height, tile_width)
            infer(engine, read_gen, writer)


# 对云掩膜栅格的进行转矢量等操作
def shpprocess(cfg):
    if isinstance(cfg, dict):
        cfg = ConfigObject(cfg)

    cloud_img = cfg.raster
    valid_shp = cfg.valid
    cloud_shp = cfg.cloud
    result_shp = cfg.result

    ds = gdal.Open(cfg.raster)
    if ds is None:
        raise Exception('can not open raster dataset.')

    # 读取原始栅格数据
    geot = ds.GetGeoTransform()
    srs = ds.GetSpatialRef()
    srcband = ds.GetRasterBand(1)
    nodata = srcband.GetNoDataValue()
    if nodata is None: nodata = 0
    nrows = ds.RasterYSize
    ncols = ds.RasterXSize
    src_data = srcband.ReadAsArray(0, 0, ncols, nrows)

    # 栅格转矢量数据
    vaild_ds = create_vaild(valid_shp, src_data, srs, geot, nodata)
    cloud_ds = create_cloud(cloud_shp, src_data, srs, geot, nodata)

    # 简化面
    simplify(vaild_ds, cfg.v_tolerance)
    simplify(cloud_ds, cfg.c_tolerance)

    # 删除小面积图斑
    delete_by_area(vaild_ds, cfg.v_min_area)
    delete_by_area(cloud_ds, cfg.c_min_area)

    # 缓冲区分析
    buffer_analysis(vaild_ds, cfg.v_buffer, cfg.v_quadsecs)
    buffer_analysis(cloud_ds, cfg.c_buffer, cfg.c_quadsecs)

    # 填补空洞
    eliminate(cloud_ds, cfg.hole_area, cfg.hole_percent)

    # 云层擦除有效区
    result_ds = create_shp(result_shp, srs, FieldName)
    erase_analysis(vaild_ds, cloud_ds, result_ds)

    # eliminate_small_part(result_ds, 50000)

    # 转移小部件
    move_part(result_ds, cloud_ds, cfg.min_part)


# 云检测，生成云范围矢量和有效区范围矢量
def detcloud(cfg):
    if isinstance(cfg, dict):
        cfg = ConfigObject(cfg)
    cloudmask(cfg)
    shpprocess(cfg)


# 初始化后处理的参数
def shpprocess_args(parser):
    parser.add_argument('--raster', type=str, help='input cloud raster which DN value in (0,1,2) ')
    parser.add_argument('--valid', type=str, help='output valid shapefile ')
    parser.add_argument('--cloud', type=str, help='output cloud shapefile ')
    parser.add_argument('--result', type=str, help='output result shapefile ')
    parser.add_argument('--v_tolerance', type=float, default=30, help='vaild layer simplify polygon tolerance')
    parser.add_argument('--c_tolerance', type=float, default=6, help='cloud layer simplify polygon tolerance')
    parser.add_argument('--v_min_area', type=float, default=10000, help='min area of polygon in vaild layer.')
    parser.add_argument('--c_min_area', type=float, default=10000, help='min area of polygon in cloud layer.')
    parser.add_argument('--v_buffer', type=float, default=-15, help='buffer size of valid layer.')
    parser.add_argument('--c_buffer', type=float, default=20, help='buffer size of cloud layer.')
    parser.add_argument('--v_quadsecs', type=float, default=5, help='quadsecs valid layer')
    parser.add_argument('--c_quadsecs', type=float, default=5, help='quadsecs cloud layer')
    parser.add_argument('--hole_area', type=float, default=8000, help='min hole area')
    parser.add_argument('--hole_percent', type=float, default=0.15, help='min percent of hole.')
    parser.add_argument('--min_part', type=float, default=20000, help='min part to move.')


# 初始化云检测的参数
def cloudmask_args(parser):
    parser.add_argument('--model_file', type=str, default=r'model/cloud_hr18.onnx', help='onnx model file path.')
    parser.add_argument('--in_raster', type=str, help='input image rasater file path')
    parser.add_argument('--out_label', type=str, help='output raster label file path.')
    parser.add_argument('--quiet', default=0, type=int, help='0: show the console progress, 1: not')
    parser.add_argument('--backend', default=0, type=int, help='opencv backend: ')
    parser.add_argument('--target', default=1, type=int, help='0 CPU, 1: OPENCL, 2: OPENCL_FP16')
    parser.add_argument('--tile_height', default=640, type=int, help=' ')
    parser.add_argument('--tile_width', default=640, type=int, help=' ')
    parser.add_argument('--x_stride', default=640, type=int, help=' ')
    parser.add_argument('--y_stride', default=640, type=int, help=' ')
    parser.add_argument('--cellsize', default=4, type=int, help=' ')
    parser.add_argument('--batchsize', default=4, type=int, help=' ')
    parser.add_argument('--bandlist', default=[1, 2, 3], nargs='+', type=int, help=' ')
    parser.add_argument('--mean', default=[58.65, 69.44, 59.07], nargs='+', type=float, help=' ')
    parser.add_argument('--std', default=[32.24, 28.16, 26.86], nargs='+', type=float, help=' ')


def main_args(parser):
    parser.add_argument('--json_path', type=str, default='parameters_det.json', help='input para json path')
    parser.add_argument('--image_list', type=list, help='input image')
    parser.add_argument('--work_dir', type=str, help='pocess work dir')
    parser.add_argument('--cost_time', type=str, help='cost_time')
    parser.add_argument('--message', type=str, help='cost_time')
    parser.add_argument('--one_flag', type=bool, default=False)


def json2args(args):
    if os.path.exists(args.json_path):
        with open(args.json_path, 'r', encoding='utf-8-sig') as f:
            js = json.load(f)
            args.in_raster = js['in_raster']
            args.work_dir = js['work_dir']

            basename = os.path.basename(args.in_raster)
            preffix, suffix = os.path.splitext(basename)

            name = preffix + '_label' + suffix
            args.out_label = os.path.join(args.work_dir, name)
            args.raster = args.out_label
            name = preffix + '_temp.shp'
            args.valid = os.path.join(args.work_dir, name)

            name = preffix + '_cloud.shp'
            args.cloud = os.path.join(args.work_dir, name)

            name = preffix + '_valid.shp'
            args.result = os.path.join(args.work_dir, name)
            return args


# 运行后处理
def run_shpprocess():
    parser = argparse.ArgumentParser()
    shpprocess_args(parser)
    args = parser.parse_args()
    shpprocess(args)


# 运行云掩膜生成
def run_cloudmask():
    parser = argparse.ArgumentParser()
    cloudmask_args(parser)
    args = parser.parse_args()
    cloudmask(args)


# 运行云检测
def run_detcloud():
    parser = argparse.ArgumentParser()
    cloudmask_args(parser)
    shpprocess_args(parser)
    args = parser.parse_args()
    detcloud(args)


def run_detcloud_js():
    # 云检测
    t0 = time()
    parser = argparse.ArgumentParser()
    main_args(parser)
    cloudmask_args(parser)
    shpprocess_args(parser)
    args = parser.parse_args()
    args.model_file = os.path.join(os.getcwd(), args.model_file)
    if os.path.exists(args.json_path):
        with open(args.json_path, 'r', encoding='utf-8-sig') as f:
            js = json.load(f)
            args.image_list = js['image_list']
            args.work_dir = js['work_dir']
            args.cost_time = js['cost_time']
            args.message = js['message']
    global MESSAGE, ONE_FLAG
    MESSAGE = args.message
    ONE_FLAG = args.one_flag

    tmp_dir = os.path.join(args.work_dir, 'tmp')
    make_file(tmp_dir)
    valid_list = []
    cloud_list = []
    temp_list = []
    for i, image in enumerate(args.image_list):
        global IMG_CURRENT
        global IMG_NUM
        IMG_NUM = len(args.image_list)
        IMG_CURRENT = i
        basename = os.path.basename(image)
        preffix, suffix = os.path.splitext(basename)
        args.in_raster = image
        name = preffix + '_label' + suffix
        args.out_label = os.path.join(tmp_dir, name)
        args.raster = args.out_label
        name = preffix + '_temp.shp'
        args.valid = os.path.join(tmp_dir, name)
        name = preffix + '_cloud.shp'
        args.cloud = os.path.join(tmp_dir, name)
        name = preffix + '_valid.shp'
        args.result = os.path.join(tmp_dir, name)
        temp_list.append(args.valid)
        cloud_list.append(args.cloud)
        valid_list.append(args.result)
        detcloud(args)

    # crop out of imgs
    work_dir = args.work_dir
    make_file(work_dir)
    driver = ogr.GetDriverByName('ESRI Shapefile')
    for valid, cloud, temp in zip(valid_list, cloud_list, temp_list):
        ds_temp = driver.Open(temp)
        layer_temp = ds_temp.GetLayerByIndex(0)

        ds_valid = driver.Open(valid, 1)
        layer_valid = ds_valid.GetLayerByIndex(0)
        srs_valid = layer_valid.GetSpatialRef()
        defn_valid = layer_valid.GetLayerDefn()
        dst_valid = os.path.join(work_dir, os.path.basename(valid))
        outds_valid = driver.CreateDataSource(dst_valid)
        outlayer_valid = outds_valid.CreateLayer(dst_valid, srs=srs_valid, geom_type=ogr.wkbPolygon)
        for i in range(defn_valid.GetFieldCount()):
            outlayer_valid.CreateField(defn_valid.GetFieldDefn(i))
        layer_valid.Intersection(layer_temp, outlayer_valid)

        ds_cloud = driver.Open(cloud, 1)
        layer_cloud = ds_cloud.GetLayerByIndex(0)
        srs_cloud = layer_cloud.GetSpatialRef()
        defn_cloud = layer_cloud.GetLayerDefn()
        dst_cloud = os.path.join(work_dir, os.path.basename(cloud))
        outds_cloud = driver.CreateDataSource(dst_cloud)
        outlayer_cloud = outds_cloud.CreateLayer(dst_cloud, srs=srs_cloud, geom_type=ogr.wkbPolygon)
        for i in range(defn_cloud.GetFieldCount()):
            outlayer_cloud.CreateField(defn_cloud.GetFieldDefn(i))
        layer_cloud.Intersection(layer_temp, outlayer_cloud)

        ds_temp.Destroy()
        ds_cloud.Destroy()
        ds_valid.Destroy()
    shutil.rmtree(tmp_dir)
    if ONE_FLAG:
        print('{}:{}'.format(MESSAGE, 1. / 3), flush=True)
    else:
        print('{}:{}'.format(MESSAGE, 1.), flush=True)
    if not ONE_FLAG:
        print("{}:{}".format(args.cost_time, time() - t0), flush=True)
    return 0


if __name__ == '__main__':
    run_detcloud_js()
