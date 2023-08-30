import time
import inspect
import os

class Timer:
    def __init__(self, title=None, to_file=False, filename='timing_results.txt'):
        self.start = None
        self.interval = None
        self.title = title
        self.to_file = to_file
        self.filename = filename

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.interval = time.time() - self.start

        # Get line number and line content
        frame = inspect.currentframe().f_back
        line_no = frame.f_lineno
        with open(frame.f_globals["__file__"], "r") as f:
            line_content = f.readlines()[line_no - 1].strip()

        # Construct the message
        msg = f"{self.title},{line_no},'{line_content}',{self.interval:.4f}"
        
        print(msg)  # Print to console in a CSV-like format

        if self.to_file:
            self._write_to_csv(msg)

    def _write_to_csv(self, msg):
        # Check if file exists
        file_exists = os.path.exists(self.filename)
        
        # Append results to file
        with open(self.filename, 'a') as f:
            # Write header if file is newly created
            if not file_exists:
                f.write("Title,Line Number,Line Content,Time (s)\n")
            f.write(msg + "\n")

# # Usage
# with Timer(title="Generating fakeB", to_file=True, filename='timing_data.csv') as t:
#     fakeB = self.genX(imgA)  # Sample line you provided
