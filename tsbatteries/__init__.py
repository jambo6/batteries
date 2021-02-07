from pbr.version import VersionInfo

all = ("__version__",)
__version__ = VersionInfo("batteries").release_string()
