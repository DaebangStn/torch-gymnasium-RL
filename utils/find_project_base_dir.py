import os


def find_project_base_dir():
    current_dir = os.path.abspath(os.getcwd())

    while True:
        if os.path.isfile(os.path.join(current_dir, 'config.ini')):
            return current_dir
        parent_dir = os.path.dirname(current_dir)

        # If we've reached the root directory, the marker file is not found
        if current_dir == parent_dir:
            raise FileNotFoundError("Project base directory not found.")

        current_dir = parent_dir
