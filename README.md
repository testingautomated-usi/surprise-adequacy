# 🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨
**This repository is now archived.**

If you are looking for a better implementation of surprise adequacy (which partially builds on this code here), 
head over to our [dnn-tip library](https://github.com/testingautomated-usi/dnn-tip), 
which is faster, better documented, easier to install and more widely applicable than the code in this repo.

# 🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨

# A Review and Refinement of Surprise Adequacy

This repository contains the code of the paper "A Review and Refinement of Surprise Adequacy" 
by M. Weiss, R. Chakraborty and P. Tonella - published at the ICSE Workshop *DeepTest 2021*.

## Implementation of Surprise Adequacy
Find our implementation of surprise adequacy in the folder ``surprise``, in particular in the file ``surprise_adequacy.py``.

The smart sampling strategies discussed in the paper can be found in ``smart_dsa_by_lsa.py`` (unsurprising-first sampling)
and ``smart_dsa_normdiffs.py`` (Neighbor-Free Sampling). 
For distribution-preserving (i.e., uniform) sampling, no class is provided - this type of sampling can easily be implemented
by passing a subset of the training set to the regular SA implementations in  ``surprise_adequacy.py``

**Known limitations:**
Currently, our implementation has to be considered as an early **beta**: 
Please use with care and note that you may have to adapt the implementation for your purposes. 
There are also features which are not yet implemented, as for example the use of SA for regression problems.
Please do not hesitate to submit pull requests if you identified and fixed problems with the implementation.

## Code of our empiricial study
The code used to generate the results of our thesis can be found in the folder ``case_studies``.
 Note that it relies on the mnist-c dataset being present on your file system (file ``mnist_corrupted.npy`` in the 
 DATASET_FOLDER, as defined in ``config.py``).

## Repository Structure

The repository is structured as follows:

<pre/>
- surprise
  | This folder contains our implementation of surprise adequacy
- case_study [archived!]
  | This folder contains the code to reproduce our results
- test
  | Unit Tests verifying that our implementation of SA is consistent with 
  | the original implementation by Kim et. al.
- scripts
  | Utilities for this repository
- Dockerfile: Definition of the environment used in our study
- requirements.txt: The dependencies of our SA implementation
- test_requirements.txt: The additional dependencies of the SA unit tests
- study_requirements.txt: The additional dependencies in our empirical study
</pre>

## Paper

If you use our code, please cite the following paper [(preprint)](https://arxiv.org/abs/2103.05939):

    @inproceedings{Weiss2021Surprise,  
      title={A Review and Refinement of Surprise Adequacy},  
      author={Weiss, Michael and Chakraborty, Rwiddhi and Tonella, Paolo},  
      booktitle={ICSEW'21: Proceedings of the IEEE/ACM 43nd International Conference on Software Engineering Workshops},  
      year={2021},  
      organization={IEEE},  
      note={forthcoming}  
    }  
    
Also, do not forget to cite the original proposition of surprise adequacy,
on which our work heavily relies [(preprint)](https://arxiv.org/abs/1808.08444):

    @inproceedings{Kim2019Surprise,
      title={Guiding deep learning system testing using surprise adequacy},
      author={Kim, Jinhan and Feldt, Robert and Yoo, Shin},
      booktitle={2019 IEEE/ACM 41st International Conference on Software Engineering (ICSE)},
      pages={1039--1049},
      year={2019},
      organization={IEEE}
    }
