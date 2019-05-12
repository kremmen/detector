# detector

Python3 object detection scripts for Raspberry Pi.

e.g.
```bash
python3 detect_image.py my-dog-and-sofa-pic.jpg  

# output:
dog: 65.27%  box: 1642, 866, 1911, 1087
sofa: 92.49%  box: 12, 234, 1512, 1074

my-dog-and-sofa-pic_output.jpg
```


These scripts need OpenCV3 to be installed. Adrian Rosebrock provides excellent detailed instructions for installing at  <https://www.pyimagesearch.com/2017/09/04/raspbian-stretch-install-opencv-3-python-on-your-raspberry-pi/>


The caffe implementation of MobileNet-SSD detection network is from <https://github.com/chuanqi305/MobileNet-SSD>.
