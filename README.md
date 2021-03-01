#  Performance Analysis of Different Hardware Architectures in a Heterogeneous Computing Platform for Machine Learning Workloads

## **Introduction**: 
Workloads in modern database settings are increasingly varying and dynamic in nature. Databases running such workloads and equipped with heterogeneous hardware like CPUs, GPUs and FPGAs need intelligent scheduling algorithms that can optimally decide what hardware to offload each task to.

This project aims to research and test some of such scheduling algorithms. 


---
 ##  System Setup

As a first step, this requires us to test different workloads individually on each of the available hardware in the heterogeneous systems.

We do this on the CloudLab Clemson ibm8335 node, which has:

|TYPE| Installed HW|
|:--:|:-----------:|
|**CPU**| Two ten-core (8 threads/core) IBM POWER8NVL CPUs at 2.86 GHz
|**RAM**| 256GB 1600MHz DDR4 memory
|**Disk**| Two Seagate 1TB 7200 RPM 6G SATA HDDs (ST1000NX0313)
|**NIC**| One Broadcom NetXtreme II BCM57800 1/10 GbE NIC
|**GPU**| NVIDIA GP100GL (Tesla P100 SMX2 16GB)
|**FPGA**| One ADM-PCIE-KU3 (Xilinx Kintex UltraScale)


---

## ML Models Under Consideration (Tentative): 

### Classical Machine Learning

-   Random Forests
-   TreeLite Models 
-   XGBoost / Gradient Boosted Ensemble Trees
-   Logistic Regression
-   State Vector Machines
-   K-Means and KNN

### Deep Learning Models



---

## ML Libraries for our Setup

|              | Header 1        | Header 2                       || Header 3                       ||
|              | Subheader 1     | Subheader 2.1  | Subheader 2.2  | Subheader 3.1  | Subheader 3.2  |
|==============|-----------------|----------------|----------------|----------------|----------------|
| Row Header 1 | 3row, 3col span                                 ||| Colspan only                   ||
| Row Header 2 |       ^                                         ||| Rowspan only   | Cell           |
| Row Header 3 |       ^                                         |||       ^        | Cell           |
| Row Header 4 |  Row            |  Each cell     |:   Centered   :| Right-aligned :|: Left-aligned  |
:              :  with multiple  :  has room for  :   multi-line   :    multi-line  :  multi-line    :
:              :  lines.         :  more text.    :      text.     :         text.  :  text.         :
|--------------|-----------------|----------------|----------------|----------------|----------------|
[Caption Text]

---
## Datasets Under Consideration: 

### For RF:
-   **IRIS**
-   **HIGGS**
---
## Todo:
- **2/25/2021**:
    * [ ] Explore Libraries for CPU/GPU/FPGAs and list them in a table.
    * [ ] Explore Use cases for these libraries.
    * [ ] Look into other datasets for RF and generally as well. See difference b/w typical RF datasets.
    * [ ] Look into XGBoost. Run some experiments and Compare with TreeLite and RAPIDS generated Models.
    * [ ] Look how RAPIDS works internally. Main question to answer: Why constant inference times for 1 - 1M input dataset sizes.
        -   https://www.youtube.com/watch?v=pXnEniQRAdQ&t=1265s
        -   https://www.youtube.com/watch?v=lV7rtDW94do
        -   https://www.youtube.com/watch?v=ZRIjLzsZ2Pc
    * [ ] Talk to Saiful regarding NVProf and HW performance counters.
        - HW Counters
        - Memory Divergence
        - Branch Divergence
        - Other Counters and Metrics that can be useful.
    * [ ] See if I can use Colab as a verification environment for replication of trends in results. 
        -   See what libraries can be used on Colab.

---

## Updates

**12/21/2020**: Awaiting Project Approval from CloudLab.

**01/05/2021**: Project Approved by CloudLab. Instantiated the IBM8335 Power 8 Machine. 

**2/25/2021**: 
-   Installed PowerAI and PowerAI Packages from PowerAI Early Access Conda Channel.
-   Libraries installed and tested : RAPIDS, SK-Learn, TVM (testing due), PyTorch, Hummingbird.
- 

---
## Resources for me
- To Log In: `ssh -p 22 hassnain@clgpu005.clemson.cloudlab.us`
- To Kill All GPU processes (and clear memory) `nvidia-smi | grep 'python' | awk '{ print $3 }' | xargs -n1 kill -9`
- Search and Replace in Vim: `:%s/old_text/new_text/g`
