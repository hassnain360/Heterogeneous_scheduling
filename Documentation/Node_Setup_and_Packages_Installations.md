# Basic Node Setup

## Increasing available disk space by repartitioning
-   Using `sudo cfdisk`, delete the two partitions under the primary 16 GB partition to get the free space right under sda1.
-   Extend sda1 using the gui
-   Restart the pc. (not always needed though)
-   Run `sudo partprobe /dev/sda` to let it know to check changes in sda.
-   Run `resize2fs /dev/sda1`. This will take around 10 - 15 minutes. 
-   Try `df -h` , and see you have 960 gigs of space mounted in sda1.

## Commands to run:
-   `sudo apt-get update`
-   `sudo apt-get upgrade`

## MiniConda
- Use conda environment, not pip.
- Use MiniConda, specifically for the POWER PC, which can be downloaded from website. 
- pip will fail to download even the most basic packages, like sklearn, scipy, etc. Use conda for that.
- If you encounter uid gid error for miniconda, just chown the miniconda folder recursively. Haven't test this, but I think the fix for this is to do the chown before you execute the .sh when installing conda.
- When creating conda env, use python 3.6. 3.7 won't work.


## Nvidia Drivers

-  Go to NVIDIA Driver Download : https://www.ibm.com/links?url=https%3A%2F%2Fwww.nvidia.com%2FDownload%2Findex.aspx
-  Go to 'See all OS' and choose the one which says 'Ubuntu' on it specifically. You'll download a .deb file.
-  Then follow instructions here: https://www.ibm.com/support/knowledgecenter/SS5SF7_1.7.0/navigation/wmlce_setupUbuntu.html
-  Use `nvidia-smi` to see the intsalled version and related information.

## PowerAI and RAPIDS

-   Follow the guide here to install PowerAI as well as RAPIDS: https://www.ibm.com/support/knowledgecenter/SS5SF7_1.6.2/navigation/wmlce_install.htm#wmlce_install__helping

---
# ONNX + ONNXRUNTIME
1. '1st_Env' was used for Sklearn for HIGGS and IRIS. '2nd_Env' was used to install onnx because that couldn't run on python 3.9.
    - `conda install cmake`
    - `pip install onnx`
    - `pip3 install onnxruntime` 
    -  Onnxruntime apparenlty isn't supported for ppe64le.
    -  Conda won't have the "onnx" package, amongst few others. In this case, just use pip install. 
    -  When you try intalling onnx on the machine using pip, it will fail with and error saying something about 'PEP 517'
    -  You need to install cmake as well as do the command `sudo apt-get install protobuf-compiler libprotoc-dev`
---

# TVM 

1. First, Install Cmake.
    -   Download source. 
    -   do `tar -zxvf file_name_here.tar.gz`
    -   cd inside the extracted folder
    -   run `./configure`
    -   run `make`
    -   run `sudo make install`
    -   run `cmake --version` to make sure cmake is installed properly.

2. Go to installation page on documentation and use conda installation method that uses recipe.
---

# HummingBird

Can't install this on ppc64le due because PyTorch version not met.

---




## Current Work:

- Node 1 on cgpu005:
    -   Has 'PowerAI' conda environment which has PowerAI, RAPIDS and TVM installed. 












https://hub.docker.com/r/ibmcom/powerai
https://repo.anaconda.com/pkgs/
https://www.ibm.com/support/knowledgecenter/SS5SF7_1.6.2/navigation/wmlce_install.htm#wmlce_install__helping
https://medium.com/@alapha23/tensorflow-1-2-0-built-from-source-on-ppc64le-4a83ae1b2e27
https://www.ibm.com/support/knowledgecenter/SS5SF7_1.6.1/navigation/wmlce_getstarted_tensorflow.html