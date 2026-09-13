import os
import struct


class FileLineIndexer:
    """
    A class to provide fast random-access reads for large text files
    by building (and using) a byte-offset index.
    """

    def __init__(self, file_path):
        """
        Initialize the indexer for 'file_path'.
          - If an index file doesn't exist, build it.
          - Otherwise, read the index file to allow random line access.
        """
        self.file_path = file_path
        # We'll store the index in the same directory, adding ".idx" to the filename.
        self.index_path = f"{file_path}.idx"

        # If the index file doesn't exist, build it.
        if not os.path.exists(self.index_path):
            self._build_index()

        # Determine how many lines we have (by file size of the index).
        self.line_count = self._calc_line_count()

    def _build_index(self):
        """
        Build a binary index file that stores 64-bit offsets for each line
        in 'self.file_path'.

        Concurrency-safe: many Slurm array jobs may open the same dataset at once.
        The index is written to a temp file and atomically renamed into place under
        an exclusive lock; other processes wait on the lock and then find it complete.
        """
        import fcntl
        lock_path = f"{self.index_path}.lock"
        with open(lock_path, 'w') as lock_f:
            fcntl.flock(lock_f, fcntl.LOCK_EX)
            try:
                if os.path.exists(self.index_path):
                    return  # another process built it while we waited
                tmp_path = f"{self.index_path}.tmp.{os.getpid()}"
                offset = 0
                with open(self.file_path, 'rb') as f_in, open(tmp_path, 'wb') as f_idx:
                    for line in f_in:
                        # Write the current offset as an 8-byte (64-bit) integer.
                        f_idx.write(struct.pack('Q', offset))
                        offset += len(line)
                os.replace(tmp_path, self.index_path)
                print(f"Index built at {self.index_path}.")
            finally:
                fcntl.flock(lock_f, fcntl.LOCK_UN)

    def _calc_line_count(self):
        """
        Calculate the total number of lines by the size of the index file.
        Each line corresponds to one 8-byte offset in the index.
        """
        idx_size = os.path.getsize(self.index_path)  # in bytes
        return idx_size // 8  # each offset = 8 bytes

    def get_line(self, line_number):
        """
        Retrieve the specified (1-based) line from the file.
        Returns the line as a string or None if out of range.
        """
        # Validate line_number
        if line_number < 1 or line_number > self.line_count:
            return None

        with open(self.index_path, 'rb') as f_idx, open(self.file_path, 'rb') as f_data:
            # Seek to the correct offset in the index file
            f_idx.seek((line_number - 1) * 8)
            offset_bytes = f_idx.read(8)
            offset = struct.unpack('Q', offset_bytes)[0]

            # Now seek to that position in the main data file
            f_data.seek(offset)
            line_bytes = f_data.readline()

        # Decode and return
        return line_bytes.decode('utf-8', errors='replace')


if __name__ == "__main__":
    pass