import numpy as np
import xml.dom.minidom

# hello world!
class other:
    datatime = None
    weights = None  # 全色融合权重

    def __init__(self, metaxml=None):
        self.weights = np.array([1, 1, 1, 1])


class GF1:
    datatime = '2020-11-27 15:08:08'
    gain = None
    offset = None
    solarirr = None
    solar_elevation = None
    sct = [3.27, 2.06, 1, 0.43]  # 瑞利散射比
    weights = None  # 全色融合权重

    def __init__(self, metaxml, sensors='auto'):
        """
        定义传感器类型，可选‘PMS1’或‘PMS2’或'auto'
        Args:
            sensors:PMS1 or PMS2 or B/C/D
            metaxml:影像的xml
        """
        dom = xml.dom.minidom.parse(metaxml)
        self.solar_elevation = 90- float(dom.getElementsByTagName('SolarZenith')[0].firstChild.data)
        if sensors == 'auto':
            self.gain = np.array([0.0731, 0.149, 0.1328, 0.1311, 0.1217])
            self.offset = np.array([0, 0, 0, 0, 0])
            self.solarirr = np.array([1371.79, 1945.29, 1854.10, 1542.9, 1080.77])
            self.weights = np.array([0.08159729, 0.14798217, 0.25725194, 0.5131686])


class GF2:
    datatime = '2020-11-27 15:08:08'
    gain = None
    offset = None
    solarirr = None
    solar_elevation = None
    sct = [3.24, 1.99, 1, 0.44]  # 瑞利散射比
    weights = None  # 全色融合权重

    def __init__(self, metaxml, sensors='auto'):
        """
        定义传感器类型，可选‘PMS1’或‘PMS2’或'auto'
        Args:
            sensors:PMS1 or PMS2
            metaxml:影像的xml
        """
        dom = xml.dom.minidom.parse(metaxml)
        self.solar_elevation = 90 - float(dom.getElementsByTagName('SolarZenith')[0].firstChild.data)
        if sensors == 'auto':
            sensors = dom.getElementsByTagName('SensorID')[0].firstChild.data
        if sensors == 'PMS1':
            self.gain = np.array([0.1884, 0.1374, 0.1784, 0.1723, 0.1894])
            self.offset = np.array([0, 0, 0, 0, 0])
            self.solarirr = np.array([1364.26, 1941.76, 1853.73, 1541.79, 1086.47])
            self.weights = np.array([0.09507853, 0.15278158, 0.23041838, 0.52172152])
        if sensors == 'PMS2':
            self.gain = np.array([0.1959, 0.1641, 0.1830, 0.1705, 0.1878])
            self.offset = np.array([0, 0, 0, 0, 0])
            self.solarirr = np.array([1362.16, 1941.22, 1853.61, 1541.70, 1086.53])
            self.weights = np.array([0.09507853, 0.15278158, 0.23041838, 0.52172152])


class GF6:
    datatime = '2020-11-27 15:19:20'
    gain = None
    offset = None
    solarirr = None
    solar_elevation = None
    sct = [3.24, 1.99, 1, 0.44]
    weights = None

    def __init__(self, metaxml, sensors='PMS'):
        """
        定义传感器类型，可选‘PMS’或‘WFV’
        Args:
            sensors:PMS or WFV
            metaxml:影像的xml
        """
        dom = xml.dom.minidom.parse(metaxml)
        self.solar_elevation = 90 - float(dom.getElementsByTagName('SolarZenith')[0].firstChild.data)
        if sensors == 'PMS':
            self.gain = np.array([0.0577, 0.0821, 0.0671, 0.0518, 0.031])
            self.offset = np.array([0, 0, 0, 0, 0])
            self.solarirr = np.array([1497.79, 1945.50, 1832.38, 1558.18, 1090.77])
            self.weights = np.array([0.1819032, 0.23793505, 0.19763335, 0.3825284])
        if sensors == 'WFV':
            self.gain = np.array([0.0633, 0.0532, 0.0508, 0.0325, 0.0523, 0.0463, 0.067, 0.0591])
            self.offset = np.array([0, 0, 0, 0, 0, 0, 0, 0])
            self.solarirr = np.array([1952.16, 1847.43, 1554.76, 1074.06, 1412.00, 1267.39, 1792.64, 1736.92])


class GF7:
    datatime = '2020-11-27 15:22:15'
    gain = None
    offset = None
    solarirr = None
    solar_elevation = None
    sct = [3.11, 1.84, 1, 0.46]
    weights = None

    def __init__(self, metaxml):
        """
        默认使用后视
        Args:
            metaxml:影像的xml
        """
        dom = xml.dom.minidom.parse(metaxml)
        elevations = dom.getElementsByTagName('SunAltitude')
        print(elevations)
        elevation_mean = 0
        for i in range(elevations.length):
            elevation_mean += float(elevations[i].firstChild.data)
        self.solar_elevation = elevation_mean / elevations.length
        self.gain = np.array([0.0879, 0.0914, 0.0981, 0.0759, 0.0925])
        self.offset = np.array([0, 0, 0, 0, 0])
        self.solarirr = np.array([1314.92, 1929.44, 1843.61, 1554.83, 1081.34])
        self.weights = np.array([0.12032595, 0.17687117, 0.23341639, 0.46938649])


class BJ02:
    datatime = '2020-11-27 15:19:20'
    gain = None
    offset = None
    solarirr = None
    solar_elevation = None
    sct = [3.19, 1.80, 1, 0.34]
    weights = None

    def __init__(self, metaxml):
        """
        定义传感器类型，可选‘PMS’或‘WFV’
        Args:
            metaxml:影像的xml
        """
        dom = xml.dom.minidom.parse(metaxml)
        self.solar_elevation = float(dom.getElementsByTagName('SUN_ELEVATION')[0].firstChild.data)
        self.gain = np.array([0.0625, 0.0625, 0.0625, 0.0625, 0.0625])
        self.offset = np.array([0, 0, 0, 0, 0])
        self.solarirr = np.array([1834.03, 2005.35, 1829.56, 1617.65, 1042.30])
        self.weights = np.array([0.31459283, 0.41627895, 0.26912535, 0.00000287])


class BJ3A:
    datatime = '2020-11-27 15:19:20'
    gain = None
    offset = None
    solarirr = None
    solar_elevation = None
    sct = [3.30, 1.953, 1, 0.419]
    weights = None

    def __init__(self, metaxml):
        """
        定义传感器类型，可选‘PMS’或‘WFV’
        Args:
            metaxml:影像的xml
        """
        dom = xml.dom.minidom.parse(metaxml)
        self.solar_elevation = float(dom.getElementsByTagName('SUN_ELEVATION')[0].firstChild.data)
        self.gain = np.array([0.0576, 0.0682, 0.0628, 0.0514, 0.0639])
        self.offset = np.array([-1.226, -1.241, -0.747, -0.381, -0.474])
        self.solarirr = np.array([1758.93, 1966.13, 1820.94, 1535.48, 1052.89])
        self.weights = np.array([0.35972915, 0.35085062, 0.28703749, 0.00238273])


class SV1:
    datatime = '2020-11-27 15:08:08'
    gain = None
    offset = None
    solarirr = None
    solar_elevation = None
    sct = [3.217, 1.2, 1, 0.443]  # 瑞利散射比
    weights = None  # 全色融合权重

    def __init__(self, metaxml, sensors='auto'):
        """
        定义传感器类型，可选‘PMS1’或‘PMS2’或'auto'
        Args:
            sensors:PMS1 or PMS2 or B/C/D
            metaxml:影像的xml
        """
        dom = xml.dom.minidom.parse(metaxml)
        self.solar_elevation = 90 - float(dom.getElementsByTagName('SolarZenith')[0].firstChild.data)
        if sensors == 'auto':
            self.gain = np.array([0.0995, 0.1435, 0.1138, 0.1082, 0.0807])
            self.offset = np.array([0, 0, 0, 0, 0])
            self.solarirr = np.array([1371.79, 1936.84, 1853.78, 1541.46, 1088.06])
            self.weights = np.array([0.08656488, 0.15225663, 0.23880252, 0.52237598])


def getsensor(name):
    try:
        sensorid = globals()[name]
    except KeyError:
        sensorid = globals()['other']
        print('没有对应的传感器，已选择通用模式')
    return sensorid


if __name__ == '__main__':
    ser = getsensor('other')()
    if ser.datatime:
        print('有传感器')




