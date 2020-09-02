The scripts in this directory, 'UMLS_to_Unitex', are for the conversion of UMLS information into Unitex format dictionaries.

### To use:
1. Edit `main.py`<br>
  On lines 11 and 12, change the `conso_path` and `types_path` to point to your local MRCONSO.RRF and MRSTY.RRF files, respectively.
2. Run `python3 ./main.py`<br>
  This will take a few minutes, since the files are relatively large.<br>
  Additionally, it might be helpful to close other programs, since this program can have high amounts of memory usage.

### Results:
Inside the 'UMLS_to_Unitex/Categorized_Dictionaries', there will now be 14 files.
1. 'drug.dic'
2. 'drug.bin'
3. 'drug.inf'
4. 'disorder.dic'
5. 'disorder.bin'
6. 'disorder.inf'
7. 'device.dic'
8. 'device.bin'
9. 'device.inf'
10. 'procedure.dic'
11. 'procedure.bin'
12. 'procedure.inf'
13. 'dlc'
14. 'dlf'

### Explanation
The '.dic' files are the raw Unitex-format dictionaries, divided by category.<br>
The '.bin' and '.inf' files are the result of compressing the '.dic' files, using the Unitex `Compress` program.<br>
'dlc' contains compound words, in Unitex-format.<br>
'dlf' contains simple words, in Unitex-format.
