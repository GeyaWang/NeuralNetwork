class ProgressBar:
    def __init__(self, length: int = 50, decimals: int = 1, fill: str = '█', prefix: str = '', suffix: str = ''):
        self.length = length
        self.decimals = decimals
        self.fill = fill
        self.prefix = prefix
        self.suffix = suffix

    def _print(self, i, total):
        percent = ("{0:." + str(self.decimals) + "f}").format(100 * (i / float(total)))
        filledLength = int(self.length * i // total)
        bar = self.fill * filledLength + '-' * (self.length - filledLength)

        print(f'\r{self.prefix}|{bar}| {percent}%{self.suffix}', end='')

    def __call__(self, iterable, prefix: str = ' ', suffix: str = ''):
        total = len(iterable)
        for i, val in enumerate(iterable):
            if i == 0:
                self._print(0, total)

            yield val
            self._print(i + 1, total)

        print()
