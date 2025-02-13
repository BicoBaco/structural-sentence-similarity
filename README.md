# structural-sentence-similarity

## Overview

I implemented this model in order to use it in my master's thesis in NLP but discarded it because the similarity is not symmetrical.

The expected model inputs are two DRGs (Discourse Representation Graph) which can be obtained from the pipeline formed by the C&C parser and the Boxer system.
The script in "generate_drg.sh" generates a DRG given a sentence and the desired filename for the output.

## C&C and Boxer installation
Done by following this repo: https://github.com/valeriobasile/learningbyreading/blob/master/README.md

## Disclaimer

This repository contains a Python implementation of the structural similarity model described in:

- Mamdouh Farouk, Measuring text similarity based on structure and word embedding, Cognitive Systems Research, 2020
- Link: [[DOI]](https://doi.org/10.1016/j.cogsys.2020.04.002)

This implementation is my own work and is not affiliated with the original authors.

## Work in Progress
- Better classes definition
- Better file management
- Better DRG explanation
- To fix: Pandas DataFrame type hint doesn't allow custom columns and rows names
