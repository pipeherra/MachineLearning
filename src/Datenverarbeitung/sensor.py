
class Sensor:
    def __init__(self, id, name):
        self.id = id
        self.name = name

    @staticmethod
    def get_sensors():
        sensors = [
            Sensor(1324180, "Unterarm"),
            Sensor(1324185, "Oberarm"),
            Sensor(1324184, "Oberschenkel"),
            Sensor(1324187, "Unterschenkel"),
            Sensor(1324186, "Rücken")
        ]
        return sensors
