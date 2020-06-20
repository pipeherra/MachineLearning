class P9Data:
    def __init__(self, id, start, end, ruhe_features, gehen_features, huepfen_features):
        self.id = id
        self.start = start
        self.end = end
        self.ruhe_features = ruhe_features
        self.gehen_features = gehen_features
        self.huepfen_features = huepfen_features
        self.ruhe_label = 'ruhe'
        self.gehen_label = 'gehen'
        self.huepfen_label = 'huepfen'