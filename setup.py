from setuptools import setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()


setup(
    name='photo_ocr',
    version='0.0.3-alpha',
    packages=['photo_ocr'],
    url='https://github.com/krasch/photo_ocr',
    license_files=('LICENSE.txt', 'LICENSE_detection.txt', 'LICENSE_recognition.txt'),
    author='krasch',
    author_email='dev@krasch.io',
    description='Read text in photos / images with complex backgrounds with this easy-to-use Python library.',
    long_description=long_description,
    long_description_content_type="text/markdown",
    python_requires=">=3.6,<3.9",
    install_requires=["torchvision>=0.7.0,<=0.10", "opencv-python>=3.4", "bbdraw"],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Software Development',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3'],
)