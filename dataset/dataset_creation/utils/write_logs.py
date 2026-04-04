class DatasetCreationLogs:
    def __init__(self):
        self.lines = []

    def add_line(self, line):
        self.lines.append(line + "\n")

    def add_line_distinction(self):
        self.lines.append("#############################\n")

    def flush(self, path):
        pass