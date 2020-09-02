The dictionary files are too large to upload to github. Follow these directions to create them and place them in the correct locations.

The 'Dictionaries' directory is for holding '.dic', '.inf', '.bin' files. It also holds two files, 'dlc' and 'dlf', inside the 'Internal_api_use' sub-directory.
There are two ways to get these files, either downloading them or creating them.

### Downloading files from Google Drive
1. Download the 'Dictionaries' folder: https://drive.google.com/drive/folders/179xBlST2qYNPy-hTdYa1Ipqq00cuP1Yn?usp=sharing

2. Place contents of downloaded 'Dictionaries' folder into this current folder 'ehrp-api-master/resources/Dictionaries', overriding if necessary


### Creating Files
1. These files can be created by running `main.py`, which can be found inside the 'UMLS_to_Unitex' directory.
     Note: `main.py` will need to be altered to point to your local MRCONSO and MRSYS files.
 
2. Once `main.py` is finished, it will have created 14 files in total, inside the 'Categorized_Dictionaries' sub-directory, 'UMLS_to_Unitex/Categorized_Dictionaries'.

3. From 'Categorized_Dictionaries', every '.dic', '.bin', and '.inf' file needs to be placed into this 'Dictionaries' directory. Make sure to exclude 'dlf' and 'dlc'.
 
4. Now, move 'dlf' and 'dlc' into the 'Internal_api_use' sub-directory, 'Dictionaries/Internal_api_use'.
 
Congrats! The dictionary files are now setup.
