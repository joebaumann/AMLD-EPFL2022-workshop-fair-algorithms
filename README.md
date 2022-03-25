# AMLD-EPFL2022-workshop-fair-algorithms

Either run the Jupyter Notebooks locally or using Google Colab & Google Drive. 

## Run Jupyter Notebooks using Google Colab & Google Drive

- Go to: http://colab.research.google.com/
- Log in with a Google account
- Create a new Notebook: File > New Notebook
- Connect the notebook to Google Drive
```
from google.colab import drive
drive.mount("/content/gdrive")
```
- Add a new code cell with: + Code
```
%cd gdrive/MyDrive
! git clone https://github.com/joebaumann/AMLD-EPFL2022-workshop-fair-algorithms.git
```

Now the repository has been saved to your Google Drive. Go to https://drive.google.com/, open the folder "AMLD-EPFL2022-workshop-fair-algorithms", and open the desired Jupyter Notebook: Open with > Google Colaboratory

### Synchronize GitHub repository
- Navigate to the desired folder
```
%cd AMLD-EPFL2022-workshop-fair-algorithms
```
- To synchronize the GitHub repository, run the following command on  Google Colab
```
! git pull
```
