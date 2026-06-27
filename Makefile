.PHONY: create-env delete-env

create-env:
	@echo "Creating conda environment: rag-sandbox..."
	conda create -n rag-sandbox python=3.11 -y
	conda run -n rag-sandbox pip install -r requirements.txt
	@echo "Done. Activate with: conda activate rag-sandbox"

delete-env:
	@echo "Removing conda environment: rag-sandbox..."
	-conda deactivate
	conda env remove -n rag-sandbox -y
	@echo "Environment removed."
