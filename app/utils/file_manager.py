"""
Gestion des dossiers In/Out.
"""

class FileManager:
    def __init__(self, input_dir: str, output_dir: str):
        self.input_dir = input_dir
        self.output_dir = output_dir

    def get_pending_files(self):
        pass

    def move_to_processed(self, file_path: str):
        pass
