Usage:
```
!pip install -e git+https://github.com/sre-fuse/fuse-face-detection#egg=fuse-face-detection
```

Load dataset (in google colab):
```
import site
site.main()
from fuse_face_detection import data_loader
train_data,test_data,train_target,test_target = data_loader.load_face_data()
``` 

Capture image in google colab:
```
from fuse_face_detection import media
from IPython.display import Image
try:
  filename = media.take_photo()
  print('Saved to {}'.format(filename))
  
  # Show the image which was just taken.
  display(Image(filename))
except Exception as err:
  # Errors will be thrown if the user does not have a webcam or if they do not
  # grant the page permission to access it.
  print(str(err))
```