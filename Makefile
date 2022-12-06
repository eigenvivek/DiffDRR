.PHONY: environment clean


dists:
	@echo "Building distributions..."
	rm -rf dist
	python -m build


testpypi: dists
	@echo "Uploading distributions..."
	python -m twine upload --repository testpypi dist/* --verbose
	

pypi: dists
	@echo "Uploading distributions..."
	python -m twine upload --repository pypi dist/* --verbose


conda:
	grayskull pypi --strict-conda-forge diffdrr


environment:
	mamba env create -f environment.yml


dev: environment
	mamba install build twine grayskull -n DiffDRR -y


clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

