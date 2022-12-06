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


environment:
	mamba env create -f environment.yml


clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

