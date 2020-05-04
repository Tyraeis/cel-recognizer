# Deployment

A distributable package for this application can be built using PyInstaller.
PyInstaller can be installed by running `pip install pyinstaller`.
Then, the command to build the package is:

```
pyinstaller --windowed --name CelRecognizer ./src/main.py
```

This will create a folder containing an executable file along with all of the dependencies of the
project, including a Python installation so that end users are not required to have Python installed.

By default, PyInstaller generates many separate files, which may make it hard to find the executable.
PyInstaller can instead pack everything into a single file if the `--onefile` argument is given in
addition to the arguments listed above. This means only a single file is generated, but this approach
increases the startup time of the application significantly.