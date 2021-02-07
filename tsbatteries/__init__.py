from pbr.version import VersionInfo

all = ("__version__",)
__version__ = VersionInfo("tsbatteries").release_string()
