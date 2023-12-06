# Image Dataset Analysis and Quality Assurance
### Implementation of the paper "Content Moderation of Generative AI Prompts, Springer’s SN Computer Science, 2023"
This is the repository for checking the quality and content of any image dataset.
#### Details
Data content analysis and quality assurance methodology involve the systematic examination of data to assess its accuracy, consistency, and relevance. We proposed the framework for analyzing data based on image settings, demographic attributes, object counting and identification, image quality, activities involved, and ill-suited content. This approach helps ensure that the data is free from errors and meets the specific criteria and standards necessary for reliable analysis, dataset cleaning and decision-making for the model training or other use cases.

### Execution
```
python "main.py" --folder_name name_of_folder --threshold 20 --objectsFile "" --threshold_width 800 --threshold_height 600
```
* --folder_name: Name of the dataset folder that you want to analyze
* --threshold: Threshold for classifying the images based on CLIP model predictions
* --objectsFile: name of the object file that you want to find in the dataset
* --threshold_width: Image resolution minimum width
* --threshold_height: Image resolution minimum height

### Citation
If you find this code useful in your research, please consider citing it as:
```
@misc{akewarmayur/ImageDatasetAnalysis,
  author = {Mayur Akewar, Dr. Rashmi Welekar},
  title = {ImageDatasetAnalysis},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub Repository},
  howpublished = {\url{https://github.com/akewarmayur/ImageDatasetAnalysis}},
}
```
