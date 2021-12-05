from distutils.core import setup 

setup(name="cnn_radar_classifier",
      version="0.0.1",
      description="Radar Modulation Classifier",
      author="Brian Egolf",
      author_email="egolfbr@miamioh.edu",
      url="https://github.com/ObeyedSky622/cnn-classifier",
      package_dir = {'':'src'},
      py_modules = ["chirp","cnn","datasetcreation"]
  
)
