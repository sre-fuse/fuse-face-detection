from setuptools import  setup

setup(
    name="fuse_face_detection",
    version="0.1.0",
    description="desc here",   
    install_requires=['tensorflow==2.4.1','opencv-python==4.5.1.48','numpy==1.19.5','scikit-learn==0.24.1', 'keras', 'tqdm'],
    zip_safe=False,
)

