import io
import csv
import logging


formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')


class CsvFormatter(logging.Formatter):
    def __init__(self, columns):
        super().__init__()
        self.header = columns
        self.written = False
        self.output = io.StringIO()
        self.writer = csv.writer(self.output, quoting=csv.QUOTE_NONNUMERIC)

    def _csv(self, rowlist) -> str:
        self.writer.writerow(rowlist)
        data = self.output.getvalue()
        self.output.truncate(0)
        self.output.seek(0)
        return data.strip()

    def format(self, record):
        rowline = list(map(record.msg.__getitem__, self.header))
        if not self.written:
            self.written = True
            return "\n".join([self._csv(self.header), self._csv(rowline)])
        return self._csv(rowline)
