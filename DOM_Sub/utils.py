import os
from osgeo import gdal, gdalconst

def coord_ras2geo(im_geotrans, coord):
    x0, x_res, _, y0, _, y_res = im_geotrans
    x = x0 + x_res * coord[0]
    y = y0 + y_res * coord[1]
    return x, y


def coord_geo2ras(im_geotrans, coord):
    x0, x_res, _, y0, _, y_res = im_geotrans
    x = int(round((coord[0] - x0) / x_res))
    y = int(round((coord[1] - y0) / y_res))
    return x, y


class IMAGE2:
    # 读图像文件
    def read_img(self, filename, ):
        self.in_file = filename
        self.dataset = gdal.Open(self.in_file)  # 打开文件
        self.im_width = self.dataset.RasterXSize  # 栅格矩阵的列数
        self.im_height = self.dataset.RasterYSize  # 栅格矩阵的行数
        self.im_bands = self.dataset.RasterCount  # 波段数
        self.im_geotrans = self.dataset.GetGeoTransform()  # 仿射矩阵，左上角像素的大地坐标和像素分辨率
        self.im_proj = self.dataset.GetProjection()  # 地图投影信息，字符串表示
        del self.dataset

    def get_extent(self, extent):
        x, y, s_size, y_size = extent
        dataset = gdal.Open(self.in_file)
        extent_img = dataset.ReadAsArray(x, y, s_size, y_size)
        return extent_img

    def create_img(self, filename, out_bands, im_width=0, im_height=0, im_proj=0, im_geotrans=0,
                   datatype=gdal.GDT_Byte):
        self.datatype = datatype
        self.out_bands = out_bands
        # 创建文件
        driver = gdal.GetDriverByName("GTiff")
        if im_width != 0 and im_height != 0:
            self.output_dataset = driver.Create(filename, im_width, im_height, out_bands, datatype)
        else:
            self.output_dataset = driver.Create(filename, self.im_width, self.im_height, out_bands, datatype)
        if im_geotrans != 0:
            self.output_dataset.SetGeoTransform(im_geotrans)
        else:
            self.output_dataset.SetGeoTransform(self.im_geotrans)  # 写入仿射变换参数
        if im_proj != 0:
            self.output_dataset.SetProjection(im_proj)
        else:
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
            datasets = gdal.Open(self.in_file)
            s = datasets.GetRasterBand(i + 1).ComputeStatistics(True)
            statis.append(s)
        return statis

    def copy_image(self, filename):
        dirname = os.path.dirname(filename)
        make_file(dirname)
        self.copy_image_file = filename
        # 判断栅格数据的数据类型
        self.dataset = gdal.Open(self.in_file)
        im_data = self.dataset.ReadAsArray(0, 0, self.im_width, self.im_height)
        if 'int8' in im_data.dtype.name:
            datatype = gdal.GDT_Byte
        elif 'int16' in im_data.dtype.name:
            datatype = gdal.GDT_UInt16
        else:
            datatype = gdal.GDT_Float32
        # 判读数组维数
        if len(im_data.shape) == 3:
            im_bands, im_height, im_width = im_data.shape
        else:
            im_bands, (im_height, im_width) = 1, im_data.shape

        # 创建文件
        driver = gdal.GetDriverByName("GTiff")
        dataset = driver.Create(filename, im_width, im_height, im_bands, datatype)

        dataset.SetGeoTransform(self.im_geotrans)  # 写入仿射变换参数
        dataset.SetProjection(self.im_proj)  # 写入投影
        if im_bands == 1:
            dataset.GetRasterBand(1).WriteArray(im_data)  # 写入数组数据
        else:
            for i in range(im_bands):
                dataset.GetRasterBand(i + 1).WriteArray(im_data[i])

        del dataset
        del self.dataset

    def write2copy_image(self, extent, im_data):
        dataset = gdal.Open(self.copy_image_file, gdal.GA_Update)
        x, y, s_size, y_size = extent
        bands = dataset.RasterCount
        if bands == 1:
            dataset.GetRasterBand(1).WriteArray(im_data, xoff=x, yoff=y)  # 写入数组数据
        else:
            for i in range(bands):
                dataset.GetRasterBand(i + 1).WriteArray(im_data[i], xoff=x, yoff=y)
        del dataset


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


def make_file(path):
    if not os.path.exists(path):
        os.makedirs(path)


def raster_mosaic(file_path_list, output_path):
    print("raster mosaic")
    assert len(file_path_list) > 1
    ds_list = []
    reference_file_path = file_path_list[0]
    input_file1 = gdal.Open(reference_file_path, gdal.GA_ReadOnly)
    input_proj1 = input_file1.GetProjection()

    for path in file_path_list:
        ds = gdal.Open(path, gdal.GA_ReadOnly)
        ds_list.append(ds)

    options = gdal.WarpOptions(
        # srcSRS=input_proj1,
        # dstSRS=input_proj1,
        format='GTiff',
        srcNodata=0,
        dstNodata=0,
        resampleAlg=gdalconst.GRA_Bilinear
    )
    gdal.Warp(output_path, ds_list, options=options)