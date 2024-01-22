from io import StringIO
from pathlib import Path
from shutil import copyfileobj


class Logger:
    """
    Logging class to util the experiments.

    Attributes
    ----------
    filename : string
        Name of the file to write
    log_buffer : StringIO
        Text buffer containing results
    verbosity : boolean
        True if strings should be printed to the standard output too
    """

    def __init__(self,
                 filename: Path,
                 verbosity: bool) -> None:
        self.filename = filename
        self.log_buffer = StringIO()
        self.verbosity = verbosity

    def header(self,
               info: str) -> None:
        """
        Stores the header string in the log buffer.

        Parameters
        ----------
        info : str
            String to log
        """
        self.log_buffer.write(info)

    def store(self,
              info: str) -> None:
        """
        Stores the string in the log buffer.

        Parameters
        ----------
        info : str
            String to log
        """
        self.log_buffer.write(info + "\n")
        if self.verbosity:
            print(f'Result: {info}')

    def finish(self) -> None:
        """
        Writes the log buffer to a file.
        """
        with open(file=self.filename, mode="w") as logfile:
            self.log_buffer.seek(0)
            copyfileobj(self.log_buffer, logfile)
