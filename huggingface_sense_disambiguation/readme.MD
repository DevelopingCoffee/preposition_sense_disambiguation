(Details verfügbar auf https://github.com/kaushaltrivedi/fast-bert)


1. 	git clone https://github.com/DevelopingCoffee/preposition_sense_disambiguation.git
	cd huggingface_sense_disambiguation
	
Opt:	virtualenv venv
	source venv/bin/activate

2.	git clone git@github.com:kaushaltrivedi/fast-bert.git
 	cd fast-bert
	pip install --editable .
	pip install -r requirements.txt
	cd ..

ONLY if 2 didn't work:
	pip install git+https://github.com/kaushaltrivedi/fast-bert.git

ONLY if apex not installed: check via: pip3 list
3. 	git clone https://github.com/NVIDIA/apex
	cd apex
	pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
	cd ..


4.	python3 bert_modell.py
	

DONE! 
Output sollte entweder im output folder oder im selben Ordner gespeichert werden





