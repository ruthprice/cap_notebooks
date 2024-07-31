import datetime as dt
import iris

class Sonde:
    fn_fmt = '%Y%m%d_%H%M%S'
    def __init__(self, filename):
        self.filename = filename
        # Assuming a given format of filename:
        i0 = filename.index('v1_')+3
        i1 = i0 + len('YYYYmmdd_HHMMSS')
        self.release_time = dt.datetime.strptime(filename[i0:i1], self.fn_fmt)

    def load(self):
        cubes = iris.load(self.filename)
        return cubes

    def load_by_name(self, name, lazy=False):
        cubes = iris.load(self.filename, name)
        if not lazy:
            for cube in cubes:
                cube.data
        return cubes

    def load_altitude(self):
        cube = self.load_by_name('altitude above MSL')[0]
        self.altitude = cube.data
        return

    def get_altitude(self):
        try:
            return self.altitude
        except AttributeError:
            self.load_altitude()
            return self.altitude

    def load_temp(self):
        cube = self.load_by_name('air_potential_temperature')[0]
        self.pot_temp = cube.data
        cube = self.load_by_name('air_temperature')[0]
        self.temperature = cube.data
        return

    def load_humidity(self):
        cube = self.load_by_name('mixing ratio')[0]
        self.humidity = cube.data
        return

    def load_pressure(self):
        cube = self.load_by_name('pressure')[0]
        self.pressure = cube.data
        return

    def load_lat(self):
        cube = self.load_by_name('reference latitude')[0]
        self.latitude = cube.data.data
        return

    def get_lat(self):
        try:
            return self.latitude
        except AttributeError:
            self.load_lat()
            return self.latitude

    def load_lon(self):
        cube = self.load_by_name('reference longitude')[0]
        self.longitude = cube.data.data
        return

    def get_lon(self):
        try:
            return self.longitude
        except AttributeError:
            self.load_lon()
            return self.longitude
