# DRAGON

The model is a Pytorch implementation for "Enhancing Dyadic Relations with Homogeneous Graphs for Multimodal Recommendation" -ECAI'23 [arxiv](https://arxiv.org/pdf/2301.12097.pdf)
```
@article{zhou2023enhancing,
  title={Enhancing Dyadic Relations with Homogeneous Graphs for Multimodal Recommendation},
  author={Zhou, Hongyu and Zhou, Xin and Shen, Zhiqi},
  journal={arXiv preprint arXiv:2301.12097},
  year={2023}
}
```

## Data
For the visual data, please go here: https://drive.google.com/drive/folders/11N8jQ-FVNiez0lI8xzYgueCqg-qs3SzV
## Run the code
1. Download the data from the data link we provided above, then put the download data to the ./data folder
2. Use ```conda env create -f dragon.yml``` to create the enviroment with correct dependencies
2. Run generate-u-u-matrix.py on the dataset you want to generate the user co-occurrence graph
`python generate-u-u-matrix.py -d dataset_name`
3. Enter the src folder and run with
`python main.py -m DRAGON -d dataset_name`

