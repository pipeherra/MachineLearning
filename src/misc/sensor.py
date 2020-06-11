
class Sensor:
    def __init__(self, _id, name):
        self.id = _id
        self.name = name

    @staticmethod
    def get_sensors():
        sensors = [
            Sensor.get_sensor_unterarm(),
            Sensor.get_sensor_oberarm(),
            Sensor.get_sensor_oberschenkel(),
            Sensor.get_sensor_unterschenkel(),
            Sensor.get_sensor_ruecken()
        ]
        return sensors

    @staticmethod
    def get_sensor_unterarm():
        return Sensor(1324180, "Unterarm")

    @staticmethod
    def get_sensor_oberarm():
        return Sensor(1324185, "Oberarm")

    @staticmethod
    def get_sensor_oberschenkel():
        return Sensor(1324184, "Oberschenkel")

    @staticmethod
    def get_sensor_unterschenkel():
        return Sensor(1324187, "Unterschenkel")


    @staticmethod
    def get_sensor_ruecken():
        return Sensor(1324186, "Rücken")
